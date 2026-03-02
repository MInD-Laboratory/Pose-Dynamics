"""Preprocessing utilities for pose data."""

# Signal cleaning utilities
from .signal_cleaning import (
	# Pose-specific preprocessing
	normalize_by_resolution,
	mask_low_confidence,
	interpolate_nans,
	filter_data_safe_preserve_nans,
	# Generic signal processing
	resample_array,
	resample_dataframe,
	align_pair,
	interpolate_run_limited,
	interpolate_dataframe_nan_runs,
	detrend_array,
	detrend_dataframe,
	normalize_array,
	normalize_dataframe,
	butterworth_filter_array,
	butterworth_filter_dataframe,
	sliding_windows,
)

# Pose preprocessing utilities
from .pose_preprocessing import (
	order_xyz_triplets,
	df_to_pose_array,
	pose_array_to_df,
	compute_interpolation_limit,
	preprocess_pose_dataframe,
	align_keypoints_3d,
	extract_keypoint_subset,
	check_data_quality,
)

# Geometry utilities
from .geometry import (
	center_points,
	procrustes_align_sequence,
)

__all__ = [
	# Pose-specific preprocessing
	"normalize_by_resolution",
	"mask_low_confidence",
	"interpolate_nans",
	"filter_data_safe_preserve_nans",
	# Signal cleaning
	"resample_array",
	"resample_dataframe",
	"align_pair",
	"interpolate_run_limited",
	"interpolate_dataframe_nan_runs",
	"detrend_array",
	"detrend_dataframe",
	"normalize_array",
	"normalize_dataframe",
	"butterworth_filter_array",
	"butterworth_filter_dataframe",
	"sliding_windows",
	# Pose preprocessing
	"order_xyz_triplets",
	"df_to_pose_array",
	"pose_array_to_df",
	"compute_interpolation_limit",
	"preprocess_pose_dataframe",
	"align_keypoints_3d",
	"extract_keypoint_subset",
	"check_data_quality",
	# Geometry
	"center_points",
	"procrustes_align_sequence",
]
