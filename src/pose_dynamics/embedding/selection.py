"""
Study-level embedding selection: evidence, proposal, and human commitment.

This is the first human-in-the-loop stage (build plan §3). The framework:

1. computes AMI and FNN across many signals (all trials, or a logged random
   subset),
2. presents per-signal curves plus their spread,
3. **proposes** ``(τ, m)`` with an explicit justification, using relative-shape /
   plateau robustness for ``τ`` (not a naive first local minimum) and the FNN knee
   erring high for ``m``,
4. but does **not** commit. The researcher inspects the evidence and commits a
   single ``(τ, m)`` — :meth:`EmbeddingEvidence.commit` — which is recorded in the
   config and applied fixed across the whole study.

Multivariate RQA skips this stage entirely (it uses the observed dimensions
directly); :class:`EmbeddingParams` records that as ``multivariate=True``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from ..data.pose_sequence import PoseSequence
from .ami import AmiCurve, ami_curve
from .fnn import FnnCurve, fnn_curve


# ----------------------------------------------------------------------
# Signals
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Signal:
    """A named 1-D signal with optional grouping metadata for diagnostics."""

    label: str
    values: np.ndarray
    group: dict[str, Any] = field(default_factory=dict)


def _as_signal(item: Any, i: int) -> Signal:
    if isinstance(item, Signal):
        return item
    if isinstance(item, tuple) and len(item) == 2:
        return Signal(label=str(item[0]), values=np.asarray(item[1], float))
    return Signal(label=f"signal{i}", values=np.asarray(item, float))


def coordinate_channels(seq: PoseSequence, group: dict[str, Any] | None = None) -> list[Signal]:
    """One signal per keypoint-axis channel (dimension-agnostic)."""
    base = dict(group or {})
    out: list[Signal] = []
    for k in range(seq.n_keypoints):
        for d in range(seq.dims):
            g = dict(base, keypoint=seq.keypoint_names[k], axis="xyz"[d])
            out.append(Signal(f"{seq.keypoint_names[k]}_{'xyz'[d]}", seq.coords[:, k, d], g))
    return out


def magnitude_channels(seq: PoseSequence, group: dict[str, Any] | None = None) -> list[Signal]:
    """One signal per keypoint: per-frame speed magnitude ``|Δx|`` (any dims)."""
    base = dict(group or {})
    speed = np.linalg.norm(np.diff(seq.coords, axis=0), axis=-1)  # (T-1, K)
    out: list[Signal] = []
    for k in range(seq.n_keypoints):
        g = dict(base, keypoint=seq.keypoint_names[k])
        out.append(Signal(f"{seq.keypoint_names[k]}_speed", speed[:, k], g))
    return out


def pool_signals(
    sequences: Iterable[PoseSequence],
    extractor: Callable[[PoseSequence, dict[str, Any]], list[Signal]] = coordinate_channels,
) -> list[Signal]:
    """Pool signals across many sequences, tagging each with its trial group."""
    pooled: list[Signal] = []
    for seq in sequences:
        group = {"trial": seq.source_file, **seq.meta}
        pooled.extend(extractor(seq, group))
    return pooled


# ----------------------------------------------------------------------
# Committed parameters
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class EmbeddingParams:
    """The human-confirmed embedding, applied fixed across a study.

    Attributes
    ----------
    tau : int
        Embedding delay in frames.
    m : int
        Embedding dimension.
    multivariate : bool
        If ``True``, no delay embedding is used (multivariate RQA); ``tau``/``m``
        are ignored.
    chosen_by : str
        Provenance of the choice (``"human_confirmed"`` by default).
    proposed_tau_range : tuple of int or None
        The delay interval the evidence supported. This is the framework's actual
        proposal for ``τ``: the AMI rules disagree on quasi-periodic pose signals, so
        the defensible output is an interval rather than a point. ``None`` when the
        params were built directly rather than via
        :meth:`EmbeddingEvidence.commit`.
    proposed_tau, proposed_m : int or None
        What the framework proposed, for the record. ``proposed_tau`` is a single-point
        reading of the aggregate curve and one member of ``proposed_tau_range``; prefer
        the range when reporting. ``m`` has no equivalent range because its selection is
        directional -- under-embedding is riskier than over-embedding -- so a lower
        bound is the meaningful quantity.
    n_signals : int
        How many signals the evidence was based on.
    notes : str
        Free-text justification recorded alongside the choice.
    """

    tau: int
    m: int
    multivariate: bool = False
    chosen_by: str = "human_confirmed"
    proposed_tau_range: tuple[int, int] | None = None
    proposed_tau: int | None = None
    proposed_m: int | None = None
    n_signals: int = 0
    notes: str = ""

    @property
    def theiler_window(self) -> int:
        """Default Theiler window for auto-RQA (the delay ``τ``)."""
        return self.tau

    @property
    def max_interp_gap(self) -> int:
        """The principled interpolation cap ``(m-1)·τ`` implied by this choice."""
        return (self.m - 1) * self.tau

    def to_dict(self) -> dict[str, Any]:
        return {
            "tau": self.tau,
            "m": self.m,
            "multivariate": self.multivariate,
            "chosen_by": self.chosen_by,
            # the range is the proposal; proposed_tau is a single-point reading of it
            "proposed_tau_range": self.proposed_tau_range,
            "proposed_tau": self.proposed_tau,
            "proposed_m": self.proposed_m,
            "n_signals": self.n_signals,
            "notes": self.notes,
        }


# ----------------------------------------------------------------------
# Proposal heuristics (suggest, never commit)
# ----------------------------------------------------------------------
def _smooth(y: np.ndarray, w: int = 7) -> np.ndarray:
    """Centered moving average with reflected edges (odd window)."""
    if w < 3 or y.size < w:
        return y
    w = w if w % 2 == 1 else w + 1
    pad = w // 2
    padded = np.pad(y, pad, mode="reflect")
    kernel = np.ones(w) / w
    return np.convolve(padded, kernel, mode="valid")


def _suggest_tau(lags: np.ndarray, ami: np.ndarray, rel_frac: float) -> dict[str, Any]:
    """Suggest tau from an AMI curve using relative-shape / plateau logic.

    Pose AMI curves are noisy and quasi-periodic (manuscript Fig. AMI archetypes):
    they rarely fall to ``1/e`` and often carry noise-induced micro-minima. The
    curve is therefore lightly smoothed, and the *primary* proposal is the first
    **prominent** local minimum of the smoothed curve (a genuine turning point),
    falling back to the plateau onset (diminishing returns) and then the relative
    crossing. All candidates are returned so the justification can show them.
    """
    ami = np.asarray(ami, float)
    valid = np.isfinite(ami)
    lags, ami = lags[valid], ami[valid]
    result: dict[str, Any] = {
        "relative": None, "first_local_min": None, "plateau": None, "primary": None
    }
    if ami.size < 5 or ami[0] <= 0:
        return result

    s = _smooth(ami, w=7)
    total_range = float(s.max() - s.min())

    # Relative crossing (may be absent for strongly periodic signals).
    rel = ami / ami[0]
    below = np.flatnonzero(rel <= rel_frac)
    result["relative"] = int(lags[below[0]]) if below.size else None

    # First *prominent* local minimum of the smoothed curve: a turning point whose
    # subsequent rise is a non-trivial fraction of the curve's range.
    if total_range > 0:
        for i in range(1, s.size - 1):
            if s[i] < s[i - 1] and s[i] <= s[i + 1]:
                following_max = s[i:].max()
                if (following_max - s[i]) >= 0.05 * total_range:
                    result["first_local_min"] = int(lags[i])
                    break

    # Plateau onset: first lag where the smoothed decrease per step falls below
    # 10% of the mean decrease over the initial descent (diminishing returns).
    drops = -np.diff(s)
    initial = drops[: max(1, drops.size // 5)]
    ref = float(np.mean(initial[initial > 0])) if np.any(initial > 0) else 0.0
    if ref > 0:
        small = np.flatnonzero(drops < 0.10 * ref)
        result["plateau"] = int(lags[small[0]]) if small.size else None

    # Primary proposal: prominent first minimum -> plateau -> relative.
    result["primary"] = (
        result["first_local_min"]
        or result["plateau"]
        or result["relative"]
    )
    return result


def _suggest_m(dims: np.ndarray, pct: np.ndarray, tol: float) -> dict[str, Any]:
    """Suggest m from an FNN curve using the diminishing-returns knee.

    Pose FNN curves often plateau at a non-negligible noise floor rather than
    reaching zero (manuscript §Embedding Dimension), so the strict "first below
    tolerance" rule chases the floor to an over-large ``m``. The primary proposal
    is instead the **knee**: the dimension after which each added dimension removes
    only a small fraction of the largest single-step reduction. The tolerance
    crossing is reported alongside it.
    """
    dims = np.asarray(dims)
    pct = np.asarray(pct, float)
    finite = np.isfinite(pct)
    dims, pct = dims[finite], pct[finite]
    result: dict[str, Any] = {"knee": None, "tol_cross": None, "primary": None}
    if pct.size == 0:
        return result

    # Tolerance crossing (reported).
    below = np.flatnonzero(pct <= tol)
    result["tol_cross"] = int(dims[below[0]]) if below.size else None

    # Knee: after the largest single-step drop, the first dimension where the
    # marginal improvement falls below ~30% of that largest drop.
    if pct.size >= 3:
        drops = -np.diff(pct)  # improvement going d -> d+1
        if drops.max() > 0:
            peak = int(np.argmax(drops))
            thresh = 0.30 * drops.max()
            knee = None
            for i in range(peak, drops.size):
                if drops[i] < thresh:
                    knee = int(dims[i])  # adding dims[i]->dims[i]+1 barely helps
                    break
            result["knee"] = knee if knee is not None else int(dims[-1])
    if result["knee"] is None:
        result["knee"] = result["tol_cross"]

    result["primary"] = result["knee"] or result["tol_cross"]
    return result


def _clamp(value: int | None, lo: int, hi: int) -> tuple[int, bool]:
    """Clamp into [lo, hi]; return (clamped, was_outside)."""
    if value is None:
        return lo, True
    clamped = max(lo, min(hi, value))
    return clamped, clamped != value


def _tau_range(tau_info: dict[str, Any], per_tau: np.ndarray,
               tau_grid: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int] | None]:
    """Span of the delay estimators that produced a value, clamped into the grid.

    The three AMI rules answer genuinely different questions -- where the curve first
    turns, where its descent flattens, and where it falls to a fixed fraction of its
    initial value -- so on quasi-periodic pose signals they routinely disagree by a
    factor of two or more. Reporting their span, rather than picking one, states what
    the evidence actually supports: any delay inside the interval is defensible, and
    the width is itself the honest measure of how weakly the data constrain the choice.

    The per-signal median is included so between-signal disagreement widens the
    interval too, not just disagreement between rules on the aggregate curve.

    Returns ``(clamped, raw)``, where ``raw`` is the unclamped span and is ``None``
    when clamping changed nothing. Both are reported: clamping to the presentation
    grid would otherwise make the interval look like a property of the grid rather
    than of the data.
    """
    lo_g, hi_g = tau_grid
    cands = [tau_info.get(k) for k in ("first_local_min", "plateau", "relative")]
    finite = per_tau[np.isfinite(per_tau)]
    if finite.size:
        cands.append(int(round(float(np.median(finite)))))
    vals = [int(c) for c in cands if c is not None]
    if not vals:                       # nothing resolved: fall back to the whole grid
        return (lo_g, hi_g), None
    raw = (min(vals), max(vals))
    clamped = (max(lo_g, min(hi_g, raw[0])), max(lo_g, min(hi_g, raw[1])))
    return clamped, (raw if raw != clamped else None)


# ----------------------------------------------------------------------
# Evidence object
# ----------------------------------------------------------------------
@dataclass
class EmbeddingEvidence:
    """Framework-computed evidence for choosing ``(τ, m)`` — inspectable, not committed."""

    labels: list[str]
    groups: list[dict[str, Any]]
    ami_lags: np.ndarray
    ami_curves: np.ndarray        # (n_signals, L), NaN-padded
    fnn_dims: np.ndarray
    fnn_curves: np.ndarray        # (n_signals, D), NaN-padded
    fnn_tau: int
    per_signal_tau: np.ndarray    # (n_signals,) float (NaN where undetermined)
    per_signal_m: np.ndarray      # (n_signals,) float
    tau_grid: tuple[int, int]
    m_grid: tuple[int, int]
    proposed_tau: int
    #: Span of the delay estimators that returned a value, clamped into ``tau_grid``.
    #: AMI-based selection does not identify a unique delay -- the local-minimum,
    #: plateau and relative-crossing rules routinely disagree by a factor of two or
    #: more on pose signals -- so the defensible output is an interval and any delay
    #: inside it is supportable. :attr:`proposed_tau` is one point in this interval,
    #: kept for the commit record; prefer reporting the range.
    proposed_tau_range: tuple[int, int]
    proposed_m: int
    justification: str
    n_signals_total: int
    n_signals_used: int
    subset_seed: int | None
    rel_frac: float
    fnn_tol: float

    # ---- aggregate curves -------------------------------------------------
    def ami_summary(self) -> dict[str, np.ndarray]:
        return _band(self.ami_curves)

    def fnn_summary(self) -> dict[str, np.ndarray]:
        return _band(self.fnn_curves)

    def summary(self) -> dict[str, Any]:
        return {
            "n_signals_used": self.n_signals_used,
            "n_signals_total": self.n_signals_total,
            "subset_seed": self.subset_seed,
            "proposed_tau": self.proposed_tau,
            "proposed_tau_range": self.proposed_tau_range,
            "proposed_m": self.proposed_m,
            "fnn_tau": self.fnn_tau,
            "tau_grid": self.tau_grid,
            "m_grid": self.m_grid,
            "per_signal_tau_median": float(np.nanmedian(self.per_signal_tau)),
            "per_signal_m_median": float(np.nanmedian(self.per_signal_m)),
            "justification": self.justification,
        }

    # ---- the human commits ------------------------------------------------
    def commit(self, tau: int, m: int, notes: str = "") -> EmbeddingParams:
        """Record the researcher's chosen ``(τ, m)`` as :class:`EmbeddingParams`.

        Warns (does not block) if the choice sits outside the presentation grid or
        below the proposed dimension, since over-embedding is safer than under.
        """
        lo_t, hi_t = self.tau_grid
        lo_m, hi_m = self.m_grid
        if not (lo_t <= tau <= hi_t):
            warnings.warn(
                f"committed tau={tau} is outside the presented grid [{lo_t}, {hi_t}]; "
                "make sure the evidence supports it.",
                stacklevel=2,
            )
        elif not (self.proposed_tau_range[0] <= tau <= self.proposed_tau_range[1]):
            # Inside the plotted grid but outside what the estimators actually spanned:
            # worth saying, since the grid is a presentation choice and the range is not.
            warnings.warn(
                f"committed tau={tau} is inside the presented grid but outside the "
                f"range the delay estimators supported, {list(self.proposed_tau_range)}; "
                "the plot should show why.",
                stacklevel=2,
            )
        if m < self.proposed_m:
            warnings.warn(
                f"committed m={m} is below the proposed m={self.proposed_m}; "
                "under-embedding is riskier than over-embedding.",
                stacklevel=2,
            )
        note = notes or self.justification
        return EmbeddingParams(
            tau=int(tau),
            m=int(m),
            chosen_by="human_confirmed",
            proposed_tau_range=tuple(self.proposed_tau_range),
            proposed_tau=self.proposed_tau,
            proposed_m=self.proposed_m,
            n_signals=self.n_signals_used,
            notes=note,
        )

    def __repr__(self) -> str:
        return (
            f"EmbeddingEvidence(n={self.n_signals_used}, "
            f"tau in {list(self.proposed_tau_range)}, m={self.proposed_m})"
        )


def _band(curves: np.ndarray) -> dict[str, np.ndarray]:
    """Median / mean / 10-90 percentile band across signals (ignoring NaN)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        return {
            "median": np.nanmedian(curves, axis=0),
            "mean": np.nanmean(curves, axis=0),
            "p10": np.nanpercentile(curves, 10, axis=0),
            "p90": np.nanpercentile(curves, 90, axis=0),
        }


def _stack_curves(per_signal: list[np.ndarray | None], axis_values: list[np.ndarray | None]):
    """Align a list of (possibly None / ragged) curves onto a common axis prefix."""
    present = [(a, c) for a, c in zip(axis_values, per_signal) if a is not None and c is not None]
    if not present:
        return np.array([]), np.empty((0, 0))
    common_len = min(len(a) for a, _ in present)
    axis = present[0][0][:common_len]
    stacked = np.full((len(per_signal), common_len), np.nan)
    j = 0
    for a, c in zip(axis_values, per_signal):
        if a is not None and c is not None:
            stacked[j, :] = c[:common_len]
        j += 1
    return axis, stacked


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def select_embedding(
    signals: Sequence[Any],
    *,
    ami_min_lag: int = 1,
    ami_max_lag: int = 140,
    fnn_min_dim: int = 1,
    fnn_max_dim: int = 10,
    fnn_tau: int | None = None,
    tau_grid: tuple[int, int] = (10, 25),
    m_grid: tuple[int, int] = (3, 6),
    rel_frac: float = 1.0 / np.e,
    fnn_tol: float = 10.0,
    subset: int | None = None,
    seed: int = 0,
) -> EmbeddingEvidence:
    """Compute AMI/FNN evidence across signals and propose ``(τ, m)``.

    Parameters
    ----------
    signals : sequence
        1-D arrays, ``(label, array)`` tuples, or :class:`Signal` objects. Pool
        across trials with :func:`pool_signals`.
    ami_min_lag, ami_max_lag, fnn_min_dim, fnn_max_dim : int
        Ranges for the AMI and FNN computations.
    fnn_tau : int, optional
        Delay at which to compute FNN. Defaults to the proposed ``τ`` from AMI.
    tau_grid, m_grid : tuple
        The bounded presentation grid to shade and clamp proposals into (defaults
        ``τ ∈ [10, 25]``, ``m ∈ [3, 6]`` per the build plan).
    rel_frac : float
        Relative-AMI fraction for the ``τ`` heuristic (default ``1/e``).
    fnn_tol : float
        FNN percentage tolerance for the ``m`` heuristic (default 10%).
    subset : int, optional
        If given and smaller than the signal count, use a random subset of this
        many signals (size and ``seed`` are logged in the evidence).
    seed : int
        RNG seed for subsampling.

    Returns
    -------
    EmbeddingEvidence
    """
    sigs = [_as_signal(s, i) for i, s in enumerate(signals)]
    n_total = len(sigs)
    if n_total == 0:
        raise ValueError("select_embedding needs at least one signal.")

    used_seed: int | None = None
    if subset is not None and subset < n_total:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_total, size=subset, replace=False))
        sigs = [sigs[i] for i in idx]
        used_seed = seed

    # --- AMI across signals ---
    ami_objs: list[AmiCurve | None] = [
        ami_curve(s.values, ami_min_lag, ami_max_lag) for s in sigs
    ]
    ami_lags, ami_curves = _stack_curves(
        [a.ami if a else None for a in ami_objs],
        [a.lags if a else None for a in ami_objs],
    )
    if ami_lags.size == 0:
        raise ValueError(
            "AMI could not be computed for any signal (all too short or constant)."
        )

    # Per-signal tau suggestions + aggregate proposal from the median curve.
    per_tau = np.array(
        [
            (_suggest_tau(a.lags, a.ami, rel_frac).get("primary") or np.nan)
            if a else np.nan
            for a in ami_objs
        ],
        dtype=float,
    )
    agg_ami = _band(ami_curves)["median"]
    agg_tau_info = _suggest_tau(ami_lags, agg_ami, rel_frac)
    raw_tau = agg_tau_info["primary"]
    proposed_tau, tau_clamped = _clamp(raw_tau, *tau_grid)
    tau_range, tau_range_raw = _tau_range(agg_tau_info, per_tau, tau_grid)

    # --- FNN across signals at the proposed tau ---
    tau_for_fnn = int(fnn_tau if fnn_tau is not None else proposed_tau)
    fnn_objs: list[FnnCurve | None] = [
        fnn_curve(s.values, tau_for_fnn, fnn_min_dim, fnn_max_dim) for s in sigs
    ]
    fnn_dims, fnn_curves = _stack_curves(
        [f.pct_false if f else None for f in fnn_objs],
        [f.dims if f else None for f in fnn_objs],
    )
    per_m = np.array(
        [(_suggest_m(f.dims, f.pct_false, fnn_tol).get("primary") or np.nan)
         if f else np.nan for f in fnn_objs],
        dtype=float,
    )
    agg_fnn = _band(fnn_curves)["median"] if fnn_curves.size else np.array([])
    agg_m_info = _suggest_m(fnn_dims, agg_fnn, fnn_tol) if agg_fnn.size else {"primary": None}
    raw_m = agg_m_info.get("primary")
    proposed_m, m_clamped = _clamp(raw_m, *m_grid)

    justification = _build_justification(
        agg_tau_info, proposed_tau, tau_clamped, tau_grid, tau_range, tau_range_raw,
        agg_m_info, proposed_m, m_clamped, m_grid, fnn_tol, tau_for_fnn,
        n_used=len(sigs), n_total=n_total, per_tau=per_tau, per_m=per_m,
    )

    return EmbeddingEvidence(
        labels=[s.label for s in sigs],
        groups=[dict(s.group) for s in sigs],
        ami_lags=ami_lags,
        ami_curves=ami_curves,
        fnn_dims=fnn_dims,
        fnn_curves=fnn_curves,
        fnn_tau=tau_for_fnn,
        per_signal_tau=per_tau,
        per_signal_m=per_m,
        tau_grid=tau_grid,
        m_grid=m_grid,
        proposed_tau=proposed_tau,
        proposed_tau_range=tau_range,
        proposed_m=proposed_m,
        justification=justification,
        n_signals_total=n_total,
        n_signals_used=len(sigs),
        subset_seed=used_seed,
        rel_frac=rel_frac,
        fnn_tol=fnn_tol,
    )


def _build_justification(
    tau_info, proposed_tau, tau_clamped, tau_grid, tau_range, tau_range_raw,
    m_info, proposed_m, m_clamped, m_grid, fnn_tol, tau_for_fnn,
    n_used, n_total, per_tau, per_m,
) -> str:
    def _median(a):
        a = a[np.isfinite(a)]
        return float(np.median(a)) if a.size else float("nan")

    tau_med, m_med = _median(per_tau), _median(per_m)
    lines = [
        f"Evidence from {n_used} of {n_total} signals.",
        "",
        "Delay (tau) — relative-shape / plateau heuristic (proposal, confirm from the plot):",
        f"  - first prominent local minimum (smoothed) at lag {tau_info.get('first_local_min')}.",
        f"  - plateau onset (diminishing returns) at lag {tau_info.get('plateau')}.",
        f"  - relative 1/e crossing at lag {tau_info.get('relative')}.",
        f"  - per-signal median suggestion: {tau_med:.1f}.",
        f"  => supported range: tau in [{tau_range[0]}, {tau_range[1]}]"
        + (f" (estimators spanned [{tau_range_raw[0]}, {tau_range_raw[1]}], "
           f"clamped to grid {list(tau_grid)})" if tau_range_raw else "") + ".",
        "     The rules above answer different questions, so they disagree; their span "
        "is what the evidence supports,",
        "     and its width measures how weakly the data constrain the delay. Any tau "
        "in the interval is defensible.",
        f"     Single-point reading of the aggregate curve, for the record: {proposed_tau}.",
        "",
        f"Dimension (m) — FNN diminishing-returns knee at tau={tau_for_fnn}:",
        f"  - knee (elbow of the FNN curve): {m_info.get('knee')}.",
        f"  - first dimension with FNN <= {fnn_tol:.0f}% (noise floor): {m_info.get('tol_cross')}.",
        f"  - per-signal median: {m_med:.1f}.",
        f"  => proposed m = {proposed_m}"
        + (f" (clamped into grid {m_grid})" if m_clamped else "") + ".",
        "",
        "NOTE: these are proposals, not decisions. Inspect the curves and their "
        "spread, then commit a single (tau, m) with evidence.commit(tau, m). The "
        "analysis needs one delay; the range says which choices the data permit, not "
        "that the choice can be left open.",
    ]
    return "\n".join(lines)
