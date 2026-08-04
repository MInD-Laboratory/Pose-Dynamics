"""
Inferential statistics for Case 2 (MOSAIC).

Deliberately a separate module that the package's ``__init__`` does **not** import.
Two reasons. These models are specific to this dataset rather than part of the
pipeline, so they do not belong alongside the preprocessing and recurrence code in
:mod:`~pose_dynamics.case_studies.mosaic.reproduce`. And they need ``statsmodels``,
which is an optional (``repro``) dependency rather than a core one -- importing
``pose_dynamics.case_studies.mosaic`` therefore stays free of that requirement, and
only ``...mosaic.stats`` pulls it in::

    from pose_dynamics.case_studies.mosaic.stats import fit_individual, to_trial_individual

Everything here operates on the tidy frames :func:`..reproduce.run_individual` and
:func:`..reproduce.run_reproduction` emit, and returns tidy coefficient tables. No
plotting, no I/O.

The models match the published Case 2 analysis: a fixed effect of background-noise
condition with Office as reference, random intercepts for pair and
individual-within-pair at the individual level and for pair alone at the dyadic level,
fit to **trial-level means** because condition is a property of the trial.
"""
from __future__ import annotations

import hashlib
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as _stats
from scipy.stats import studentized_range

#: See :data:`..reproduce._SOURCE_SHA`. This module is *not* part of the notebook's cache
#: fingerprint -- it does not shape the ROI signals -- so a stale copy here would otherwise
#: go entirely unnoticed.
_SOURCE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]

# Aliased CFG, not C: patsy resolves the ``C(participant)`` in vc_formula against
# this module's globals, so a name ``C`` here shadows patsy's own C() categorical
# function and the fit dies with "'module' object is not callable".
from . import config as CFG

ROIS = ["arms", "upper_body", "centre_face"]
METRICS_IND = ["rms", "mean_vel", "sd_vel"]
# Order follows Section 3.2.1: recurrence rate, determinism, laminarity, mean and
# maximum diagonal line length, entropy, trapping time -- plus the linear
# cross-correlation, which is not a recurrence measure but is modelled the same way.
METRICS_DYAD = ["xcorr_lag0", "cross_perc_recur", "cross_perc_determ",
                "cross_laminarity", "cross_mean_line_length", "cross_lmax",
                "cross_entropy", "cross_trapping_time"]
TERMS = [("C(condition, Treatment('Office'))[T.Cafe]", "Cafe"),
         ("C(condition, Treatment('Office'))[T.Food]", "Food"),
         ("C(condition, Treatment('Office'))[T.Party]", "Party")]
_TERM = "C(condition, Treatment('Office'))[T.%s]"

_FORMULA = "{metric} ~ C(condition, Treatment('Office'))"


def _smf():
    """Import ``statsmodels`` on first use, with an actionable message if absent."""
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:  # pragma: no cover - depends on the install extra
        raise ImportError(
            "Case 2's statistics need statsmodels, an optional dependency: "
            "pip install 'pose-dynamics[repro]'"
        ) from exc
    return smf


# ----------------------------------------------------------------------
# Row preparation / aggregation
# ----------------------------------------------------------------------
def _prep_individual(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the grouping keys the individual models need.

    ``session`` identifies the pair; ``session`` + ``camera`` (a dedicated webcam per
    partner) identifies one participant. Works on a copy so it is idempotent and
    independent of the order callers run things in.
    """
    f = frame.copy()
    f["session"] = f["session"].astype(str)
    f["participant"] = f["session"] + "_" + f["camera"]
    return f


def _prep_dyad(frame: pd.DataFrame) -> pd.DataFrame:
    """As :func:`_prep_individual`, but the dyadic models group on pair only."""
    f = frame.copy()
    f["session"] = f["session"].astype(str)
    return f


def to_trial_individual(frame: pd.DataFrame) -> pd.DataFrame:
    """Window rows -> one row per participant x trial x ROI.

    Condition is a property of the trial, so the trial is the level at which the fixed
    effect varies. Fitting the raw 60 s window rows instead would treat every window
    inside a trial as an independent replicate of that condition; with 50% overlap
    adjacent windows share half their samples, so their residuals cannot be independent
    and the condition standard errors would be deflated.
    """
    f = _prep_individual(frame)
    return (f.groupby(["session", "camera", "participant", "trial", "condition", "roi"],
                      observed=True)[METRICS_IND].mean().reset_index())


def to_trial_dyad(frame: pd.DataFrame) -> pd.DataFrame:
    """Window rows -> one row per dyad x trial x ROI (see :func:`to_trial_individual`)."""
    f = _prep_dyad(frame)
    return (f.groupby(["session", "trial", "condition", "roi"],
                      observed=True)[METRICS_DYAD].mean().reset_index())


# ----------------------------------------------------------------------
# Degrees of freedom
# ----------------------------------------------------------------------
def containment_df(n_obs: int, n_fixed: int, n_groups: int) -> int:
    """Between-within (containment) denominator df: ``n_obs - n_fixed - n_groups``.

    This is the convention the published Case 2 analysis used, and it is what
    ``statsmodels`` does *not* provide -- its ``df_resid`` is the naive
    ``n_obs - n_fixed``, which ignores the random-intercept levels entirely. Condition
    varies within participant (and within dyad), so one df is charged per level of the
    finest random grouping. On this dataset the rule returns 451 for the individual arms
    model and 220 for the dyadic arms model, matching the 451 and 219-221 reported in
    the manuscript.

    Note this is *not* the Satterthwaite approximation, which would need the sampling
    covariance of the variance components and is unavailable in ``statsmodels``. For
    these sample sizes the two agree to well within a rounded p-value, so the practical
    reason to prefer containment is consistency with the published values.
    """
    return int(n_obs - n_fixed - n_groups)


def _p_from_t(t: float, df: int) -> float:
    """Two-sided p for a t statistic on ``df`` denominator degrees of freedom."""
    return float(2.0 * _stats.t.sf(abs(t), df))


# ----------------------------------------------------------------------
# Reference contrasts (Office as baseline)
# ----------------------------------------------------------------------
# statsmodels emits this whenever *any* variance component falls below a hard-coded
# absolute 0.01 (mixed_linear_model.py: ``np.min(np.abs(vcomp)) < 0.01``). The threshold
# is on the raw variance, so it is a statement about the outcome's units, not about the
# fit: these metrics live on ~0.01-0.03 in normalised coordinates, so their variance
# components are ~1e-4 and trip it unavoidably. Multiplying any outcome by 1000 silences
# it and moves every t statistic by < 1e-13, which is what makes it safe to drop --
# see :func:`_fit_one`. The warning that *would* matter is a non-converged optimiser or
# a singular covariance, and both are escalated to exceptions below.
_BOUNDARY_MSG = "The MLE may be on the boundary of the parameter space."
_FAILURE_MARKERS = ("optimization failed", "singular", "not positive definite")


def _check_fit(caught: list, res, metric: str, kind: str) -> None:
    """Escalate genuine fit failures; drop the benign boundary notice; keep the rest.

    Anything unrecognised is re-emitted rather than swallowed, so this narrows the
    warning stream without hiding anything new that statsmodels might start saying.
    """
    if not getattr(res, "converged", True):
        raise RuntimeError(
            f"mixedlm did not converge for {kind}/{metric}; the reported coefficients "
            "cannot be trusted. Try a different optimizer or rescale the outcome."
        )
    for w in caught:
        msg = str(w.message)
        if _BOUNDARY_MSG in msg:
            continue
        if any(marker in msg.lower() for marker in _FAILURE_MARKERS):
            raise RuntimeError(f"mixedlm fit failed for {kind}/{metric}: {msg}")
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)


def _fit_one(sub: pd.DataFrame, metric: str, kind: str):
    """Fit one ROI x metric model.

    ``kind="individual"`` nests an individual-within-pair term inside a per-pair
    intercept, matching the paper. ``re_formula="1"`` is required alongside
    ``vc_formula`` or ``statsmodels`` silently drops the pair-level intercept. Note that
    with two partners per pair the two random terms are collinear by construction (the
    participant dummies within a pair sum to that pair's intercept), so the variance
    components are weakly identified and ``pair_var`` can settle at exactly zero. That
    does not affect the condition contrasts, which are estimated within participant,
    but the variance components themselves should not be interpreted.

    Warnings are handled here rather than left to the caller's global filter state,
    which in a notebook is neither predictable nor reproducible. See
    :data:`_BOUNDARY_MSG` for why the boundary notice is dropped, and
    :func:`_check_fit` for what is escalated instead.
    """
    smf = _smf()
    formula = _FORMULA.format(metric=metric)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if kind == "individual":
            res = smf.mixedlm(formula, sub, groups=sub["session"], re_formula="1",
                              vc_formula={"individual": "0 + C(participant)"}).fit()
        else:
            res = smf.mixedlm(formula, sub, groups=sub["session"]).fit()
    _check_fit(caught, res, metric, kind)
    return res


def _fit_reference(frame: pd.DataFrame, metrics: list[str], kind: str) -> pd.DataFrame:
    """Office-reference contrasts for every ROI x metric, with containment df."""
    group_col = "participant" if kind == "individual" else "session"
    rows = []
    for roi in ROIS:
        sub = frame[frame["roi"] == roi]
        if sub.empty:
            continue        # an ROI can be absent entirely, e.g. dropped for missingness
        for metric in metrics:
            s = sub.dropna(subset=[metric])
            m = _fit_one(s, metric, kind)
            df_c = containment_df(int(m.nobs), len(TERMS) + 1, s[group_col].nunique())
            # the Office mean. Kept so a condition's estimated marginal mean is
            # ``intercept + beta`` straight from this table -- reporting the shape of an
            # effect needs the level, not just the contrast, and a near-zero measure like
            # the cross-correlation can have a "reduction" that is really a sign change.
            base = {"roi": roi, "metric": metric, "n_obs": int(m.nobs), "df": df_c,
                    "converged": bool(getattr(m, "converged", True)),
                    "intercept": m.params["Intercept"]}
            if kind == "individual":
                base["pair_var"] = m.cov_re.iloc[0, 0]
                base["individual_var"] = m.vcomp[0]
            for term, label in TERMS:
                t_stat = m.params[term] / m.bse[term]
                rows.append({**base, "vs_Office": label, "beta": m.params[term],
                             "SE": m.bse[term], "t": t_stat,
                             "p": _p_from_t(t_stat, df_c)})
    return pd.DataFrame(rows).round(4)


def fit_individual(frame: pd.DataFrame) -> pd.DataFrame:
    """Individual-level reference contrasts. Accepts trial-level or window-level rows."""
    return _fit_reference(frame, METRICS_IND, "individual")


def fit_dyadic(frame: pd.DataFrame) -> pd.DataFrame:
    """Dyadic reference contrasts. Accepts trial-level or window-level rows."""
    return _fit_reference(frame, METRICS_DYAD, "dyad")


# ----------------------------------------------------------------------
# All-pairwise comparisons, Tukey-adjusted
# ----------------------------------------------------------------------
def tukey_pairwise(frame: pd.DataFrame, metrics: list[str], kind: str) -> pd.DataFrame:
    """All six pairwise condition contrasts with Tukey-adjusted p-values.

    Not a supplementary check: the paper's significance claims are pairwise and
    adjusted ("Food Court and Party always exceeded Cafe"), so the Office-reference
    contrasts alone are not comparable to them. Contrasts are formed from the
    treatment-coded fit -- Office's coefficient is structurally 0, and each contrast's
    standard error comes from the model's own parameter covariance via
    ``Var(b_X - b_Y) = Var(b_X) + Var(b_Y) - 2 Cov(b_X, b_Y)`` -- then adjusted with the
    studentized range over the four condition means, the same adjustment ``emmeans``
    applies. Signed louder-minus-quieter, so a positive difference means the noisier
    condition scored higher.
    """
    conds = CFG.CONDITION_ORDER
    k = len(conds)
    group_col = "participant" if kind == "individual" else "session"
    rows = []
    for roi in ROIS:
        by_roi = frame[frame["roi"] == roi]
        if by_roi.empty:
            continue        # as in _fit_reference
        for metric in metrics:
            s = by_roi.dropna(subset=[metric])
            m = _fit_one(s, metric, kind)
            # same denominator df as the reference contrasts, so the two tables agree
            df_c = containment_df(int(m.nobs), len(TERMS) + 1, s[group_col].nunique())
            cov = m.cov_params()
            term = {c: (None if c == "Office" else _TERM % c) for c in conds}
            for quieter, louder in combinations(conds, 2):  # CONDITION_ORDER is quiet->loud
                tl, tq = term[louder], term[quieter]
                b = ((0.0 if tl is None else m.params[tl])
                     - (0.0 if tq is None else m.params[tq]))
                v = ((0.0 if tl is None else cov.loc[tl, tl])
                     + (0.0 if tq is None else cov.loc[tq, tq])
                     - 2 * (0.0 if (tl is None or tq is None) else cov.loc[tl, tq]))
                se = float(np.sqrt(max(v, 0.0)))
                t = b / se if se > 0 else np.nan
                # studentized range takes q = |t| * sqrt(2) for a pairwise mean difference
                p_tukey = (float(studentized_range.sf(abs(t) * np.sqrt(2), k, df_c))
                           if se > 0 and np.isfinite(t) else np.nan)
                rows.append({"roi": roi, "metric": metric,
                             "contrast": f"{louder} - {quieter}",
                             "diff": b, "SE": se, "t": t, "df": df_c,
                             "p_tukey": p_tukey})
    return pd.DataFrame(rows).round(4)


def check_tukey_conservative(tukey: pd.DataFrame, reference: pd.DataFrame) -> int:
    """Assert Tukey p >= unadjusted p on the reference contrasts; return how many.

    Tukey can only ever be more conservative than the unadjusted contrasts. If that
    ordering inverts, the adjustment is being applied wrongly.
    """
    merged = tukey.merge(
        reference.assign(contrast=lambda d: d["vs_Office"] + " - Office"),
        on=["roi", "metric", "contrast"], suffixes=("_tk", "_raw"))
    if not (merged["p_tukey"] >= merged["p"] - 1e-9).all():
        raise AssertionError("Tukey p below unadjusted p")
    return len(merged)


# ----------------------------------------------------------------------
# Strategy comparison
# ----------------------------------------------------------------------
def compare_alignment(aligned: pd.DataFrame, noalign: pd.DataFrame,
                      keys: tuple[str, ...] = ("roi", "metric", "vs_Office")
                      ) -> pd.DataFrame:
    """Two coefficient tables side by side, flagging terms whose significance flips."""
    merged = aligned.merge(noalign, on=list(keys), suffixes=("_aligned", "_noalign"))
    merged["beta_diff"] = (merged["beta_aligned"] - merged["beta_noalign"]).round(4)
    merged["sig_changed"] = (merged["p_aligned"] < 0.05) != (merged["p_noalign"] < 0.05)
    return merged
