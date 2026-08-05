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

# Bone list for rendering a 38-keypoint pose as a stick figure (ZED BODY_38
# parent->child hierarchy). Used only for visualization (the PM figure).
SKELETON_EDGES = [
    # spine and head
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (5, 6), (5, 7), (6, 8), (7, 9),
    # left arm: neck -> clavicle -> shoulder -> elbow -> wrist
    (4, 10), (10, 12), (12, 14), (14, 16),
    # right arm
    (4, 11), (11, 13), (13, 15), (15, 17),
    # left hand (thumb, index, middle, pinky tips)
    (16, 30), (16, 32), (16, 34), (16, 36),
    # right hand
    (17, 31), (17, 33), (17, 35), (17, 37),
    # left leg: pelvis -> hip -> knee -> ankle -> heel/toes
    (0, 18), (18, 20), (20, 22), (22, 28), (22, 24), (24, 26),
    # right leg
    (0, 19), (19, 21), (21, 23), (23, 29), (23, 25), (25, 27),
]

# --- embedding (committed after AMI/FNN) ---
TAU = 20
M = 4

# --- CRQA (numeric_inventory 9.11-9.14): per-keypoint magnitude, delay-embedded,
# z-scored, fixed 2.5% recurrence rate, l_min 2, Euclidean. ---
TARGET_REC = 2.5
MIN_LINE = 2
NORM = "zscore"
RESCALE = "mean"

# --- MdCRQA (multidimensional cross-recurrence over the five-keypoint magnitude
# vector: each dimension z-scored, then delay-embedded with the shared (TAU, M),
# so the two bodies are compared as whole multivariate systems). ---
MD_TARGET_REC = 2.5
MD_MIN_LINE = 2

# Fixed-radius MdCRQA: %REC becomes an outcome rather than a pinned constant.
# The primary radius is the grand mean radius achieved under the fixed 2.5% REC
# solution, so the two modes are centred on the same recurrence density; the grid
# is the sensitivity check (0.61 anchors on the back-to-back baseline instead).
MD_RADIUS = 0.59
MD_RADIUS_GRID = [0.50, 0.55, 0.59, 0.61, 0.65]

# Per-keypoint (scalar-signal) CRQA at a fixed radius. The grid spans the range of
# radii the fixed-2.5%-REC solution converged on (0.146-0.409). 0.30 is the primary:
# it reproduces the originally reported Case-3 coefficients, and the condition
# effects hold at every radius on the grid (see the notebook sweep).
#
# Note the mean of the solved radii (0.297) is NOT a density-matched anchor: the
# per-keypoint radii are heterogeneous and %REC is steeply nonlinear in radius, so
# applying one mid-range radius yields far more than 2.5% REC. Radius reported as a
# density measure is a mean of radii, not the radius of the mean.
CROSS_RADIUS = 0.30
CROSS_RADIUS_GRID = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

# --- conditions ---
CONDITION_ORDER = ["b2b", "uni", "f2f"]
CONDITION_LABELS = {"b2b": "Back-to-back", "uni": "Unidirectional", "f2f": "Face-to-face"}
N_TRIALS = 12            # 2 blocks x 6 trials
