"""
Mirror Game dataset configuration.

This module defines all dataset-specific constants for the Mirror Game case
study: sampling parameters, the 38-keypoint skeleton layout produced by the
ZED stereo-camera body-tracking system, body-region groupings used for
kinematic feature extraction, experimental condition labels, and the RQA
measures reported in the paper.

The Mirror Game is a joint-improvisation paradigm in which two participants
move together under three visual-feedback conditions that systematically vary
the coupling information available:
  - **Back-to-back (b2b)**: no visual feedback between partners.
  - **Unidirectional (uni)**: one designated leader can see the follower but
    not vice versa.
  - **Face-to-face (f2f)**: mutual visual feedback, both participants can see
    each other.

Pose data were recorded at 30 Hz using a ZED stereo camera and processed with
the ZED body-tracking SDK, yielding 3-D joint positions for 38 keypoints per
participant per frame.
"""
from pathlib import Path
import re
from collections import defaultdict

TARGET_RATE = 30.0  # Pose data sampling rate (Hz)

MG_RE = re.compile(r"^(P\d{3})_(T\d+)_((?:P1|P2))_pose_3d\.csv$", re.IGNORECASE)

KEYPOINT_MAPPING = {
    # Core body (0-4)
    0: 'pelvis',
    1: 'spine_1',
    2: 'spine_2',
    3: 'spine_3',
    4: 'neck',
    
    # Head (5-9)
    5: 'nose',
    6: 'left_eye',
    7: 'right_eye',
    8: 'left_ear',
    9: 'right_ear',
    
    # Shoulders/Clavicles (10-13)
    10: 'left_clavicle',
    11: 'right_clavicle',
    12: 'left_shoulder',
    13: 'right_shoulder',
    
    # Arms (14-17)
    14: 'left_elbow',
    15: 'right_elbow',
    16: 'left_wrist',
    17: 'right_wrist',
    
    # Hips/Legs (18-23)
    18: 'left_hip',
    19: 'right_hip',
    20: 'left_knee',
    21: 'right_knee',
    22: 'left_ankle',
    23: 'right_ankle',
    
    # Feet (24-29)
    24: 'left_big_toe',
    25: 'right_big_toe',
    26: 'left_small_toe',
    27: 'right_small_toe',
    28: 'left_heel',
    29: 'right_heel',
    
    # Left hand fingers (30-33)
    30: 'left_hand_thumb_4',
    31: 'right_hand_thumb_4',
    32: 'left_hand_index_1',
    33: 'right_hand_index_1',
    34: 'left_hand_middle_4',
    35: 'right_hand_middle_4',
    36: 'left_hand_pinky_1',
    37: 'right_hand_pinky_1',
}

SELECTED_KEYPOINTS = list(range(38))  # All keypoints

SKELETON_EDGES = [
    # Spine
    (0, 1), (1, 2), (2, 3), (3, 4),  # Pelvis → Spine → Neck
    
    # Head
    (4, 5),  # Neck → Nose
    (5, 6), (5, 7),  # Nose → Eyes
    (6, 8), (7, 9),  # Eyes → Ears
    
    # Shoulders/Clavicles
    (4, 10), (4, 11),  # Neck → Clavicles
    (10, 12), (11, 13),  # Clavicles → Shoulders
    
    # Left arm
    (12, 14), (14, 16),  # Shoulder → Elbow → Wrist
    
    # Right arm
    (13, 15), (15, 17),  # Shoulder → Elbow → Wrist
    
    # Hips
    (0, 18), (0, 19),  # Pelvis → Hips
    
    # Left leg
    (18, 20), (20, 22),  # Hip → Knee → Ankle
    (22, 24), (22, 26),  # Ankle → Toes
    (22, 28),  # Ankle → Heel
    
    # Right leg
    (19, 21), (21, 23),  # Hip → Knee → Ankle
    (23, 25), (23, 27),  # Ankle → Toes
    (23, 29),  # Ankle → Heel
    
    # Left hand fingers (from wrist)
    (16, 30), (16, 32), (16, 34), (16, 36),
    
    # Right hand fingers (from wrist)
    (17, 31), (17, 33), (17, 35), (17, 37),
]

BODY_REGIONS = {
    'head': {
        'indices': [5, 6, 7, 8, 9],
        'label': 'Head',
        'n_keypoints': 5
    },
    'arms': {
        'indices': [12, 13, 14, 15, 16, 17],
        'label': 'Arms', 
        'n_keypoints': 6
    },
    'legs': {
        'indices': [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        'label': 'Legs',
        'n_keypoints': 12
    },
    'torso': {
        'indices': [0, 1, 2, 3, 4],
        'label': 'Torso',
        'n_keypoints': 5
    }
}