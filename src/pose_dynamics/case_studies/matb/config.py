"""
Case 1 (MATB) configuration — the recovered numeric definitions.

All indices are 0-based (the raw OpenPose export is 1-based; the converter and the
loader shift them). Values are from ``numeric_inventory.md`` §1, §7.
"""
from __future__ import annotations

# --- acquisition / preprocessing ---
FRAME_RATE = 60.0
IMG_WIDTH = 2560
IMG_HEIGHT = 1440
CONF_THRESHOLD = 0.30
INTERP_CAP = 60          # frames (1 s @ 60 Hz)
FILTER_CUTOFF = 10.0     # Hz
FILTER_ORDER = 4

# --- landmark indices (0-based; 1-based OpenPose value in comments) ---
PROCRUSTES_LANDMARKS = [29, 30, 36, 45]           # nose tip 30, side 31, eyelids 37, 46
LEFT_EYE_RING = [36, 37, 38, 39, 40, 41]          # 37-42
RIGHT_EYE_RING = [42, 43, 44, 45, 46, 47]         # 43-48
LEFT_PUPIL = 68                                    # 69
RIGHT_PUPIL = 69                                   # 70
MOUTH = [62, 66]                                   # 63, 67
BLINK_L_TOP = [37, 38]                             # 38, 39
BLINK_L_BOT = [40, 41]                             # 41, 42
BLINK_R_TOP = [43, 44]                             # 44, 45
BLINK_R_BOT = [46, 47]                             # 47, 48

# --- embedding (committed after AMI/FNN) ---
TAU = 20
M = 4

# --- windowing ---
WINDOW_S = 60.0
OVERLAP = 0.5

# --- RQA / CRQA (numeric_inventory 7.15-7.19) ---
# Auto-RQA: radius 0.2 mean-rescaled, Theiler 2, l_min 4, unit-interval norm.
AUTO_RADIUS = 0.2
AUTO_THEILER = 2
AUTO_MINL = 4
# CRQA: radius 0.3 mean-rescaled, l_min 2, Theiler 0, unit-interval norm.
CROSS_RADIUS = 0.3
CROSS_MINL = 2
NORM = "minmax"          # unit-interval normalization
RESCALE = "mean"


def feature_pipeline_config(template) -> list[dict]:
    """The Case 1 feature pipeline as a config list.

    Normalize to screen -> anisotropic Procrustes emitting BOTH streams
    (aligned geometry + head-motion parameters) -> pupil offset (two-eye) on the
    aligned coordinates.
    """
    return [
        {"step": "coordinate_normalization",
         "params": {"width": IMG_WIDTH, "height": IMG_HEIGHT, "mode": "unit"}},
        {"step": "procrustes",
         "params": {"template": template, "landmarks": PROCRUSTES_LANDMARKS,
                    "scale": "anisotropic", "emit": "both", "prefix": "head"}},
        {"step": "offset_feature",
         "params": {"name_out": "pupil_metric",
                    "point": [LEFT_PUPIL, RIGHT_PUPIL],
                    "center": [LEFT_EYE_RING, RIGHT_EYE_RING]}},
    ]
