"""
RQA parameter set — one explicit, validated, logged decision per knob.

All RQA parameters (embedding, radius mode, Theiler window, minimum line length,
rescaling, normalization) are fixed across a study by default, declared here, and
written into every result's provenance (build plan §5). There is exactly one
normalization decision (``norm``) and it is passed through to ``rqa-analysis``
explicitly — the framework never normalizes a second time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

RadiusMode = Literal["fixed_radius", "fixed_rrec"]

# Distance-rescaling codes understood by the rqa-analysis C++ core
# (rqa_utils.cpp: 1 -> divide by mean, 2 -> divide by max, 0 -> none).
_RESCALE_CODE = {"mean": 1, "max": 2, "none": 0}


@dataclass(frozen=True)
class RqaParams:
    """Resolved RQA parameters for one analysis.

    Parameters
    ----------
    eDim, tLag : int
        Embedding dimension and delay (from the committed :class:`EmbeddingParams`).
        Ignored when ``multivariate=True``.
    radius_mode : {"fixed_rrec", "fixed_radius"}
        ``"fixed_rrec"`` (default, usability): the framework solves for the radius
        that hits ``target_rec``. ``"fixed_radius"`` (reproduction): ``radius`` is
        supplied and %REC is an outcome.
    radius : float or None
        Recurrence threshold for ``fixed_radius`` mode (as a fraction of the
        rescaled distance, e.g. 0.2 of the mean pairwise distance).
    target_rec : float or None
        Target recurrence rate in percent for ``fixed_rrec`` mode (e.g. 2.5).
    norm : str
        The single normalization decision, passed to rqa-analysis
        (``"zscore"``, ``"minmax"``, ``"center"``, or ``"none"``).
    rescale : {"mean", "max", "none"}
        Distance rescaling before thresholding. ``"mean"`` (default) matches the
        manuscript.
    theiler : int or None
        Theiler window for auto-RQA. ``None`` resolves to ``tLag`` (the rule of
        thumb). Cross-RQA always uses 0.
    min_line : int
        Minimum diagonal/vertical line length (l_min).
    multivariate : bool
        If ``True``, no delay embedding (multivariate RQA uses observed dims).
    bisect_tol : float
        Convergence tolerance for the radius search, in percent %REC.
    bisect_max_iter : int
        Maximum bisection iterations before reporting non-convergence.
    radius_hi : float
        Initial upper bound for the radius search (expanded if needed).
    """

    eDim: int
    tLag: int
    radius_mode: RadiusMode = "fixed_rrec"
    radius: float | None = None
    target_rec: float | None = None
    norm: str = "zscore"
    rescale: str = "mean"
    theiler: int | None = None
    min_line: int = 2
    multivariate: bool = False
    bisect_tol: float = 0.05
    bisect_max_iter: int = 50
    radius_hi: float = 2.0

    def __post_init__(self) -> None:
        if self.radius_mode not in ("fixed_radius", "fixed_rrec"):
            raise ValueError(f"radius_mode must be 'fixed_radius' or 'fixed_rrec'; got {self.radius_mode!r}")
        if self.radius_mode == "fixed_radius":
            if self.radius is None:
                raise ValueError("fixed_radius mode requires a radius.")
            if self.target_rec is not None:
                raise ValueError("Provide radius OR target_rec, not both (fixed_radius mode).")
        else:  # fixed_rrec
            if self.target_rec is None:
                raise ValueError("fixed_rrec mode requires target_rec (percent).")
            if self.radius is not None:
                raise ValueError("Provide target_rec OR radius, not both (fixed_rrec mode).")
            if not (0 < self.target_rec < 100):
                raise ValueError(f"target_rec must be a percent in (0, 100); got {self.target_rec}.")
        if self.rescale not in _RESCALE_CODE:
            raise ValueError(f"rescale must be one of {list(_RESCALE_CODE)}; got {self.rescale!r}")

    # --- resolved values -------------------------------------------------
    @property
    def rescale_code(self) -> int:
        return _RESCALE_CODE[self.rescale]

    def theiler_for(self, analysis: str) -> int:
        """Theiler window to use for a given analysis type."""
        if analysis in ("cross", "multivariate_cross"):
            return 0  # no autocorrelation to suppress across two signals
        return self.tLag if self.theiler is None else self.theiler

    def lib_params(self, radius: float, theiler: int) -> dict[str, Any]:
        """Build the params dict for rqa-analysis, forcing side effects off."""
        return {
            "norm": self.norm,
            "eDim": self.eDim,
            "tLag": self.tLag,
            "rescaleNorm": self.rescale_code,
            "radius": float(radius),
            "tw": int(theiler),
            "minl": self.min_line,
            # The framework owns normalization, plotting, and output:
            "showMetrics": False,
            "plotMode": "none",
            "pointSize": 1,
            "saveFig": False,
            "doStatsFile": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "eDim": self.eDim,
            "tLag": self.tLag,
            "radius_mode": self.radius_mode,
            "radius": self.radius,
            "target_rec": self.target_rec,
            "norm": self.norm,
            "rescale": self.rescale,
            "theiler": self.theiler,
            "min_line": self.min_line,
            "multivariate": self.multivariate,
        }

    @classmethod
    def from_embedding(cls, embedding, **kwargs) -> "RqaParams":
        """Build from a committed :class:`EmbeddingParams`, overriding as needed."""
        return cls(
            eDim=embedding.m,
            tLag=embedding.tau,
            multivariate=getattr(embedding, "multivariate", False),
            **kwargs,
        )
