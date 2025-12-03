"""
MOSAIC Project Configuration

Dataset-specific constants, keypoint mappings, and analysis parameters
for the MOSAIC dyadic pose dynamics study.
"""

from dataclasses import dataclass
from pathlib import Path

# ============================================================================
# DATASET CONSTANTS
# ============================================================================

# Sampling rate
TARGET_RATE = 60.0  # Hz

# Video resolution
VIDEO_WIDTH = 720
VIDEO_HEIGHT = 720

# Preprocessing parameters
CONF_THRESHOLD = 0.4  # Minimum confidence for valid keypoints
MAX_INTERP_GAP = 60   # Maximum frames to interpolate (1 second at 60 fps)
FILTER_CUTOFF = 10.0  # Low-pass filter cutoff frequency (Hz)
FILTER_ORDER = 4      # Butterworth filter order

# ============================================================================
# KEYPOINT SETS
# ============================================================================

PREDEFINED_SETS = {
    'face': ["Eye", "Pupil", "Chin", "Jaw", "Cheek", "Nostril", "Lip", "Temple", "Nose"], 
    'center_face': ["Eye", "Pupil", "Chin", "Nostril", "Lip", "Nose"],
    'hand': ["wristBase", "Tip"],
    'arm': ["Shoulder", "Elbow", "Wrist"],
    'body': ["Shoulder", "MidHip", "Neck"],
    'temple': ["Temple"],   # strict match
    'nose': ["Nose"]        # strict match
}

# Default keypoint sets for analysis
DEFAULT_KEYPOINT_SETS = ["center_face", "body", "arm"]

# ============================================================================
# SKELETON CONNECTIONS FOR VISUALIZATION
# ============================================================================

SKELETON_CONNECTIONS = {
    "face": [    
        ("Nose", "LEye"),     
        ("Nose", "REye"),     
        ("Nose", "Neck"),     
    ],
    "body": [
        ("Neck", "RShoulder"),
        ("Neck", "LShoulder"),
        ("RShoulder", "RElbow"),
        ("RElbow", "RWrist"),
        ("LShoulder", "LElbow"),
        ("LElbow", "LWrist"),
        ("Neck", "MidHip"),
        ("MidHip", "RHip"),
        ("RHip", "RKnee"),
        ("RKnee", "RAnkle"),
        ("MidHip", "LHip"),
        ("LHip", "LKnee"),
        ("LKnee", "LAnkle"),
    ],
    "arm": [
        ("RShoulder", "RElbow"),
        ("RElbow", "RWrist"),
        ("LShoulder", "LElbow"),
        ("LElbow", "LWrist"),
    ]
}

# ============================================================================
# WINDOWING PARAMETERS
# ============================================================================

WINDOW_PARAMS = {
    "rqa": {
        "size_seconds": 60,      # 60-second windows for RQA/CRQA
        "size_frames": 60 * 60,  # 3600 frames at 60 fps
        "overlap": 0.5           # 50% overlap
    },
    "linear": {
        "size_seconds": 5,       # 5-second windows for linear metrics
        "size_frames": 5 * 60,   # 300 frames at 60 fps
        "overlap": 0.5           # 50% overlap
    }
}

# ============================================================================
# RQA/CRQA PARAMETERS
# ============================================================================

RQA_PARAMS = {
    "time_lags": [15],
    "embedding_dims": [4],
    "radii": [0.15],
    "minl": 2,           # Minimum line length
    "tw": 2,             # Theiler window (for RQA)
    "tw_crqa": 0,        # Theiler window for CRQA (dyadic analysis)
    "norm": 2,           # Normalization method (2 = z-score)
    "rescaleNorm": 1
}

# ============================================================================
# PCA PARAMETERS
# ============================================================================

N_COMPONENTS = 6  # Number of principal components to extract

# ============================================================================
# ALIGNMENT PARAMETERS
# ============================================================================

SYMMETRIZATION_MODES = ["none", "nose", "torso", "full"]
DEFAULT_SYMMETRIZATION = "none"  # No symmetrization by default
ALLOW_ROTATION = True            # Allow rotation in Procrustes alignment
USE_PROCRUSTES = True            # Use Procrustes alignment

# Reference points for alignment
REFERENCE_OPTIONS = ["Torso", "Nose"]
DEFAULT_REFERENCE = "Torso"

# ============================================================================
# CUSTOM FEATURES
# ============================================================================

CUSTOM_FEATURES = ["blink_dist", "lip_dist"]

# ============================================================================
# ANIMATION PARAMETERS
# ============================================================================

ANIMATION_PARAMS = {
    "fps": 40,
    "n_frames": 250,
    "scale": 2.0,
    "mode": "sine",  # "sine" or "subject"
    "width_openpose": 720,
    "height_openpose": 720,
    "width_video": 1920,
    "height_video": 1080,
    "zoom_factor": 1.0
}

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================

OUTPUT_DIRS = {
    "rqa_results": "rqa_results",
    "crqa_results": "crqa_results",
    "linear_results": "linear_results",
    "animations": "animations",
    "diagnostics": "diagnostics"
}

# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

@dataclass
class MOSAICConfig:
    """Main configuration for MOSAIC analysis."""
    
    # Data paths (to be set by user)
    data_path: str = "D:/mosaic analysis files"
    
    # Processing parameters
    fps: float = TARGET_RATE
    video_width: int = VIDEO_WIDTH
    video_height: int = VIDEO_HEIGHT
    conf_threshold: float = CONF_THRESHOLD
    max_interp_gap: int = MAX_INTERP_GAP
    filter_cutoff: float = FILTER_CUTOFF
    filter_order: int = FILTER_ORDER
    
    # Keypoint sets
    keypoint_sets: list = None
    
    # Windowing
    window_size_rqa: int = WINDOW_PARAMS["rqa"]["size_frames"]
    window_size_linear: int = WINDOW_PARAMS["linear"]["size_frames"]
    window_overlap: float = 0.5
    
    # PCA
    n_components: int = N_COMPONENTS
    
    # Alignment
    symmetrization_mode: str = DEFAULT_SYMMETRIZATION
    allow_rotation: bool = ALLOW_ROTATION
    use_procrustes: bool = USE_PROCRUSTES
    reference: str = DEFAULT_REFERENCE
    
    # Analysis flags
    use_velocity: bool = False  # Compute features on velocity instead of position
    skip_sessions: list = None  # Sessions to skip (e.g., [1])
    
    def __post_init__(self):
        if self.keypoint_sets is None:
            self.keypoint_sets = DEFAULT_KEYPOINT_SETS
        if self.skip_sessions is None:
            self.skip_sessions = []

# Global configuration instance
CFG = MOSAICConfig()
