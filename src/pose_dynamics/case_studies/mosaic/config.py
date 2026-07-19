"""
Case 2 (MOSAIC) configuration — recovered numeric definitions.

OpenPose body+face export with NAME-based columns. ROIs are the manuscript's
BODY_25 groups (numeric_inventory §8.2-8.4), expressed here by keypoint name.
Values from ``numeric_inventory.md`` §8.
"""
from __future__ import annotations

# --- acquisition / preprocessing ---
FRAME_RATE = 60.0
TARGET_RATE = 30.0        # downsample after filtering
VIDEO_WIDTH = 720
VIDEO_HEIGHT = 720
CONF_THRESHOLD = 0.30
INTERP_CAP = 60           # frames (1 s @ 60 Hz)
FILTER_CUTOFF = 10.0      # Hz
FILTER_ORDER = 4

# --- ROIs (numeric_inventory §8.2-8.4) ---
# arms      = BODY_25 {2,3,4,5,6,7}; upper_body = {1,2,5,8};
# centre_face = body {0,15,16} + face-model landmarks.
ROI_EXACT = {
    "arms": ["RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist"],
    "upper_body": ["Neck", "RShoulder", "LShoulder", "MidHip"],
    "centre_face": ["Nose", "REye", "LEye"],
}
# centre_face also aggregates matching face landmarks (substring, like the
# prototype's name-based sets: Eye/Pupil/Chin/Nostril/Lip).
CENTRE_FACE_SUBSTRINGS = ["Eye", "Pupil", "Chin", "Nostril", "Lip"]

# --- embedding (committed after AMI/FNN, at 30 Hz) ---
TAU = 10
M = 4

# --- windowing ---
WINDOW_S = 60.0
OVERLAP = 0.5

# --- RQA / CRQA (numeric_inventory §8.13-8.18) ---
RADIUS = 0.2              # mean-rescaled
MIN_LINE = 2
NORM = "zscore"
RESCALE = "mean"
AUTO_THEILER = 2          # cross-RQA uses 0 (forced by the wrapper)

# --- conditions ---
CONDITION_ORDER = ["Office", "Cafe", "Food", "Party"]
CONDITION_DB = {"Office": 60, "Cafe": 70, "Food": 80, "Party": 87.5}
