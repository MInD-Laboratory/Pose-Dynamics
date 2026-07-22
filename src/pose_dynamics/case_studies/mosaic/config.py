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
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
CONF_THRESHOLD = 0.30
INTERP_CAP = 60           # frames (1 s @ 60 Hz)
FILTER_CUTOFF = 10.0      # Hz
FILTER_ORDER = 4
# The raw MOSAIC CSVs' "_offset" columns are already translated to a video-centre
# origin upstream of this pipeline (confirmed: face keypoints cluster near y=0 at
# eye/cheek height, with the rest of the seated body descending to large negative
# y -- consistent with a centre, not top-left, origin). So only the scaling half of
# the paper's normalization step happens here; re-translating already-centred input
# (mode="centered") would corrupt it. "scale_only" divides by the *full* 1920x1080
# (not half), matching cathy-dev's actual executed spatial.scale.method="screen"
# (src/pose_dynamics/preprocess/spatial.py) -- confirmed against the real dataset:
# global extremes across all 550 files are x in [-479.8, 1510.9], y in [-925.3,
# 481.0], comfortably inside [-1,1] under full-dimension division (would exceed it
# under half-dimension division, e.g. an occluded knee's y as low as -925).
NORMALIZE_MODE = "scale_only"

# --- windowed Procrustes alignment (paper: single global template, per-window fit) ---
TEMPLATE_SAMPLE = 24          # files used to build the global Procrustes template
MIN_KEYPOINTS_FOR_FIT = 4     # minimum valid keypoints to attempt a window's fit
MIN_VALID_FRAC_PER_KP = 0.20  # a keypoint needs finite coords in >= this fraction of a window's frames to be used

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

# --- curated alignment/feature keypoint set (cathy-dev's `selection.keypoints`,
# preprocess.yaml) --- deliberately excludes lower-body points (hips, knees,
# ankles, toes, heels): they're occluded/unreliable in a seated conversation and
# must not influence the Procrustes fit. In this design every entry here ends up
# in one of the three ROIs (the face landmarks all match a CENTRE_FACE_SUBSTRINGS
# substring), so this list and the ROI union happen to coincide exactly.
SELECTED_KEYPOINTS = [
    "Nose", "REye", "LEye", "rightChin", "leftChin",
    # right eyebrow
    "rightOuterEyeBrow", "rightOuter1EyeBrow", "rightPeakEyeBrow",
    "rightInner1EyeBrow", "rightInnerEyeBrow",
    # left eyebrow
    "leftInnerEyeBrow", "leftInner1EyeBrow", "leftPeakEyeBrow",
    "leftOuter1EyeBrow", "leftOuterEyeBrow",
    # nostrils
    "leftNostril", "leftInNostril", "centerNostril", "rightInNostril", "rightNostril",
    # right eye
    "rightEdgeEyeLeft", "rightTopEyeInner", "rightTopEyeOuter",
    "rightEdgeEyeRight", "rightLowerEyeRight", "rightLowerEyeLeft",
    # left eye
    "leftEdgeEyeLeft", "leftTopEyeOuter", "leftTopEyeInner",
    "leftEdgeEyeRight", "leftLowerEyeRight", "leftLowerEyeLeft",
    # outer lip
    "rightOuterLip", "rightUpperLip", "rightTopLip", "topCenterLip", "leftTopLip",
    "leftUpperLip", "leftOuterLip", "leftLowerLip", "leftBottomLip",
    "bottomCenterLip", "rightBottomLip", "rightLowerLip",
    # inner lip
    "rightInnerLip", "rightInnerTopLip", "innerTopCenterLip", "leftInnerTopLip",
    "leftInnerLip", "leftInnerBottomLip", "innerBottomCenterLip", "rightInnerBottomLip",
    # pupils
    "rightPupil", "leftPupil",
    # upper body
    "Neck", "RShoulder", "LShoulder", "MidHip",
    # arms
    "RElbow", "RWrist", "LElbow", "LWrist",
]

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
