# config.py
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class RQAParams:
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
    window_size_sec: float = 60.0
    step_size_sec: float = 30.0
    min_coverage: float = 0.6


@dataclass
class SamplingRates:
    pose_fps: float = 30.0
    face_fps: float = 30.0
    eyelink_raw_hz: float = 1000.0
    eyelink_downsampled_hz: float = 60.0


@dataclass
class InterpolationConfig:
    max_gap_frames_default: int = 60


@dataclass
class GlobalConfig:
    RQA: RQAParams = field(default_factory=RQAParams)
    WINDOWING: WindowingDefaults = field(default_factory=WindowingDefaults)
    SAMPLING: SamplingRates = field(default_factory=SamplingRates)
    INTERP: InterpolationConfig = field(default_factory=InterpolationConfig)


_CFG = GlobalConfig()

def get_cfg() -> GlobalConfig:
    return _CFG
