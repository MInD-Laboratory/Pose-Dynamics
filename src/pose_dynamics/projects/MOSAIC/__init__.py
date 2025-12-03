"""MOSAIC project module for dyadic pose analysis."""

from .pipeline import *
from .alignment import *
from .features import *

__all__ = [
    # Data loading
    'load_mosaic_session',
    'load_all_sessions',
    'extract_keypoints',
    
    # Alignment utilities
    'build_symmetric_template',
    'compute_reference_limb_lengths',
    'batch_apply_fixed_lengths',
    'align_keypoints',
    
    # Feature extraction
    'add_custom_features',
    'compute_velocity',
    
    # Window utilities
    'get_window_indices',
]
