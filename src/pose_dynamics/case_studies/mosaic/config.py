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
# Both axes are divided by the SAME constant. This is the single most important
# property of this step and it is not a typo: the published run called
# ``preprocess_mosaic_trial`` without overriding its ``video_width``/``video_height``
# arguments, so it took their defaults of 720x720 (upstream
# ``projects/MOSAIC/pipeline.py``, pinned at commit 127677e). Dividing x by 1920 and
# y by 1080 instead -- the nominal frame dimensions, which an earlier version of this
# file used -- compresses the pose 1.78x along one axis. That is not a change of
# units but a change of shape, and it breaks two things. First, ``windowed_align``
# fits rotation plus *uniform* scale, which structurally cannot undo per-axis
# compression, so the distortion survives alignment and lands in the aligned
# trajectories. Second, RQA z-scores each window, so
# an isotropic divisor cancels out of every recurrence measure whereas a per-axis one
# does not: it re-weights x against y inside the velocity magnitude and changes the
# signal's shape, not just its scale.
#
# The specific value 720 therefore matters only for the linear metrics, which are not
# z-scored and so carry its units; any single isotropic constant is equally
# shape-correct. 720 is used here to keep the linear numbers comparable with the
# published figures.
VIDEO_WIDTH = 720
VIDEO_HEIGHT = 720
CONF_THRESHOLD = 0.30
INTERP_CAP = 60           # frames (1 s @ 60 Hz)
FILTER_CUTOFF = 10.0      # Hz
FILTER_ORDER = 4
# The raw MOSAIC CSVs' "_offset" columns are already translated to a video-centre
# origin upstream of this pipeline (confirmed: face keypoints cluster near y=0 at
# eye/cheek height, with the rest of the seated body descending to large negative
# y -- consistent with a centre, not top-left, origin). So only the scaling half of
# the paper's normalization step happens here; re-translating already-centred input
# (mode="centered") would corrupt it. "scale_only" divides by the *full* dimension
# rather than half, matching upstream ``normalize_by_resolution``, which divides by
# ``width``/``height`` outright.
#
# Note this leaves normalized coordinates *unbounded*: with a 720 divisor the
# dataset's global extremes (x in [-479.8, 1510.9], y in [-925.3, 481.0]) map to
# roughly x in [-0.67, 2.10], y in [-1.29, 0.67]. The paper describes "frame edges
# mapped to [-1,1]", which would need half-dimension division -- another point where
# the manuscript's text and the executed prototype disagree. Nothing downstream
# assumes a bound (the recurrence radius is mean-rescaled and the signals are
# z-scored), and keypoints can legitimately be tracked beyond the visible frame
# (e.g. an occluded knee under a table), so the executed behavior is reproduced here.
NORMALIZE_MODE = "scale_only"

# --- windowed Procrustes alignment (paper: single global template, per-window fit) ---
# None = build the template from every file, pooling all valid frames across the
# whole dataset (paper: "averaging...across all valid frames in the dataset").
# Set an int to cap the file count for faster iteration during development.
TEMPLATE_SAMPLE = None
MIN_KEYPOINTS_FOR_FIT = 4     # minimum valid keypoints to attempt a window's fit
MIN_VALID_FRAC_PER_KP = 0.20  # a keypoint needs finite coords in >= this fraction of a window's frames to be used

# --- window inclusion rule ---
# Three modes, in decreasing strictness. The ROI centroid is a mean over its member
# keypoints, so partial membership is the hazard being managed: if a member drops in or
# out mid-window the centroid shifts between two different points and registers as a
# spurious velocity spike (measured at ~100x a normal frame). Because RMS squares before
# averaging, a handful of such frames inflated upper_body RMS by ~11%, arms by ~4%, and
# centre_face by ~1% -- worst for small ROIs, upper_body having only four members, one of
# them the occlusion-prone MidHip.
#
# "roi_complete" (default): an ROI's signal for a window requires all of *that ROI's own*
#   keypoints finite throughout it; otherwise that ROI alone is dropped for that window.
#   Removes the artifact by construction while confining the cost to the ROI responsible.
#   Retains ~94% of windows for arms, ~99% for upper_body, ~78% for centre_face.
#
# "all_keypoints": reproduces the prototype, which discarded a window outright for EVERY
#   ROI if any of the 62 selected keypoints held a missing value at any frame within it
#   (`window.isnull().any().any()`). Retains 72%. Note this lets a face landmark void the
#   arms signal, which never uses it -- defensible only as a fidelity choice, and the
#   dropout is condition-dependent (centre_face 17% in Office rising to 27% in Party), so
#   it discards disproportionately from the loudest condition. That is the retention bias
#   Section 2.3.1 warns about, so it is not used as the default.
#
# "roi_available": the original behaviour -- keep every window and average over whichever
#   members happen to be finite each frame. Retains ~100% but admits the artifact above.
#
# MIN_KEYPOINTS_FOR_FIT and MIN_VALID_FRAC_PER_KP above apply only to the Procrustes fit,
# and are inactive under "all_keypoints", which admits only fully-observed windows.
WINDOW_COMPLETENESS = "roi_complete"

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

# --- curated alignment/feature keypoint set --- deliberately excludes
# lower-body points (hips, knees, ankles, toes, heels): they're occluded/unreliable
# in a seated conversation and
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

# --- limb-length normalization (see reproduce.apply_fixed_limb_lengths) ---
# The published prototype forced each upper-limb segment to a constant length on
# every frame, after Procrustes and before differentiation. It is not described in
# the manuscript's methods, but it was executed, and omitting it leaves the arms ROI
# (and only the arms ROI) overstated -- roughly 15% on RMS and 1.3-1.75x on the
# CRQA condition effects. Set False to measure that contribution again.
APPLY_LIMB_RESCALE = True
# Order matters. Each side fixes shoulder->elbow first, then elbow->wrist from the
# *already corrected* elbow, so the chain composes distally -- matching the
# prototype's dict-insertion order.
LIMB_CHAIN = [("LShoulder", "LElbow"), ("LElbow", "LWrist"),
              ("RShoulder", "RElbow"), ("RElbow", "RWrist")]

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
