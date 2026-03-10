"""
Global configuration for the pose_dynamics analysis package.

This module centralises all analysis parameters so that values stay consistent
across the preprocessing, feature-extraction, and recurrence-quantification
pipelines.  Defaults reflect the parameter choices reported in the accompanying
methods paper and can be overridden at runtime via ``set_cfg``.

Usage
-----
    from pose_dynamics.config import get_cfg

    cfg = get_cfg()
    radius = cfg.RQA.radius        # recurrence radius
    fps    = cfg.SAMPLING.pose_fps  # pose sampling rate
"""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class RQAParams:
    """
    Parameters for Recurrence Quantification Analysis (RQA) and Cross-RQA (CRQA).

    These values control the phase-space reconstruction and recurrence-plot
    construction steps described in the methods section of the paper.

    Attributes
    ----------
    eDim : int
        Embedding dimension (m) for Takens' delay-embedding theorem.
        Chosen via False Nearest Neighbours (FNN) analysis.
    tLag : int
        Time delay (tau, in frames) for the delay embedding.
        Chosen via the first minimum of the Auto Mutual Information (AMI) function.
    radius : float
        Recurrence threshold: two embedded points are considered recurrent if
        their distance is <= radius.  Applied after z-score normalisation.
    tw : int
        Theiler window — the number of diagonals adjacent to the main diagonal
        that are excluded from line-structure statistics to avoid counting
        trivially correlated (temporally neighbouring) points.
    minl : int
        Minimum diagonal-line length (L_min) used when computing determinism,
        entropy, and related diagonal-line measures.
    norm : str
        Normalisation applied to each time series before embedding.
        ``"zscore"`` subtracts the mean and divides by the standard deviation.
    rescaleNorm : bool
        If True, rescale the distance matrix to [0, 1] before thresholding.

    ami_min_lag, ami_max_lag : int
        Lag range (in frames) over which AMI is evaluated during parameter estimation.
    fnn_min_dim, fnn_max_dim : int
        Embedding dimension range searched during FNN analysis.
    """
    eDim: int = 4
    tLag: int = 15
    radius: float = 0.15
    tw: int = 1
    minl: int = 2
    norm: str = "zscore"
    rescaleNorm: bool = True

    ami_min_lag: int = 1
    ami_max_lag: int = 140
    fnn_min_dim: int = 1
    fnn_max_dim: int = 10


@dataclass
class WindowingDefaults:
    """
    Default sliding-window parameters for time-series segmentation.

    Pose data are analysed in overlapping temporal windows so that
    recurrence measures can be tracked as they evolve over the course of
    an experimental session.

    Attributes
    ----------
    window_size_sec : float
        Duration of each analysis window in seconds.
    step_size_sec : float
        Step between successive window starts (i.e., window_size - overlap).
        A 50 % overlap corresponds to step_size_sec = window_size_sec / 2.
    min_coverage : float
        Minimum fraction of a window that must contain valid (non-NaN) data
        for the window to be included in downstream analyses.
    """
    window_size_sec: float = 60.0
    step_size_sec: float = 30.0
    min_coverage: float = 0.6


@dataclass
class SamplingRates:
    """
    Nominal sampling rates (Hz) for each data modality.

    Rates are used when converting between frame counts and seconds and
    when constructing Butterworth filters.

    Attributes
    ----------
    pose_fps : float
        Frame rate of the 3-D pose estimation output (e.g. ZED body-tracking).
    face_fps : float
        Frame rate of 2-D facial keypoint estimation (e.g. OpenPose).
    eyelink_raw_hz : float
        Native sampling rate of the EyeLink eye-tracker.
    eyelink_downsampled_hz : float
        Rate to which eye-tracking data are downsampled before analysis.
    """
    pose_fps: float = 30.0
    face_fps: float = 30.0
    eyelink_raw_hz: float = 1000.0
    eyelink_downsampled_hz: float = 60.0


@dataclass
class InterpolationConfig:
    """
    Parameters controlling NaN gap-filling by linear interpolation.

    Attributes
    ----------
    max_gap_frames_default : int
        Maximum consecutive missing frames that will be filled by linear
        interpolation.  Gaps longer than this are left as NaN.
        At 30 fps, the default of 60 frames corresponds to a 2-second gap.
    """
    max_gap_frames_default: int = 60


@dataclass
class GlobalConfig:
    """
    Top-level configuration container.

    Holds one instance of each parameter group.  Access sub-configs via
    attribute lookup, e.g. ``get_cfg().RQA.radius``.
    """
    RQA: RQAParams = field(default_factory=RQAParams)
    WINDOWING: WindowingDefaults = field(default_factory=WindowingDefaults)
    SAMPLING: SamplingRates = field(default_factory=SamplingRates)
    INTERP: InterpolationConfig = field(default_factory=InterpolationConfig)


_CFG = GlobalConfig()


def set_cfg(cfg: GlobalConfig) -> None:
    """Replace the global configuration with a custom ``GlobalConfig`` instance."""
    global _CFG
    _CFG = cfg

def get_cfg() -> GlobalConfig:
    """Return the active global configuration."""
    return _CFG
