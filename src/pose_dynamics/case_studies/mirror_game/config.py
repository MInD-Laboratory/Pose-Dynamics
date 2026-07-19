"""
Case 3 (Mirror Game) configuration — recovered numeric definitions.

ZED 38-keypoint 3D model, 0-based. Values from ``numeric_inventory.md`` §9.
"""
from __future__ import annotations

# --- acquisition / preprocessing ---
TARGET_RATE = 30.0       # resample the variable-rate ZED stream to a uniform grid
FILTER_CUTOFF = 5.0      # Hz (numeric_inventory 9.8)
FILTER_ORDER = 4

# --- keypoint indices (38-keypoint ZED model) ---
PELVIS = 0
# five anatomically-informative keypoints (numeric_inventory 9.9)
SUBSET = {"head": 5, "l_wrist": 16, "r_wrist": 17, "l_ankle": 22, "r_ankle": 23}
SUBSET_INDICES = list(SUBSET.values())
SUBSET_NAMES = list(SUBSET)

# --- embedding (committed after AMI/FNN) ---
TAU = 20
M = 4

# --- CRQA (numeric_inventory 9.11-9.14): per-keypoint magnitude, delay-embedded,
# z-scored, fixed 2.5% recurrence rate, l_min 2, Euclidean. ---
TARGET_REC = 2.5
MIN_LINE = 2
NORM = "zscore"
RESCALE = "mean"

# --- conditions ---
CONDITION_ORDER = ["b2b", "uni", "f2f"]
CONDITION_LABELS = {"b2b": "Back-to-back", "uni": "Unidirectional", "f2f": "Face-to-face"}
N_TRIALS = 12            # 2 blocks x 6 trials
