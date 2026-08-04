"""Case 2 (MOSAIC) reproduction: config and entry points.

The pipeline lives in :mod:`.reproduce`; the dataset-specific mixed-effects models live
in :mod:`.stats`. Both are re-exported here. ``stats`` needs ``statsmodels``, an optional
(``repro``) extra, but imports it lazily at first fit, so importing this package does not
require it.
"""
from __future__ import annotations

from . import config
from .reproduce import (
    build_global_template,
    default_conditions_csv,
    load_condition_map,
    load_mosaic_file,
    parse_file,
    plot_alignment_comparison_crqa_arms,
    plot_alignment_comparison_crqa_invariant,
    plot_alignment_comparison_linear,
    plot_case2_crqa_figure,
    plot_case2_figure,
    preprocess_pose,
    process_dyad,
    resolve_rois,
    roi_velocity_signals,
    run_individual,
    run_reproduction,
    windowed_align,
)
from .stats import (
    METRICS_DYAD,
    METRICS_IND,
    ROIS,
    check_tukey_conservative,
    compare_alignment,
    containment_df,
    fit_dyadic,
    fit_individual,
    to_trial_dyad,
    to_trial_individual,
    tukey_pairwise,
)

__all__ = [
    "config",
    # --- pipeline (reproduce) ---
    "parse_file",
    "default_conditions_csv",
    "load_condition_map",
    "resolve_rois",
    "load_mosaic_file",
    "preprocess_pose",
    "build_global_template",
    "windowed_align",
    "roi_velocity_signals",
    "run_individual",
    "process_dyad",
    "run_reproduction",
    # --- statistics (stats) ---
    "ROIS",
    "METRICS_IND",
    "METRICS_DYAD",
    "to_trial_individual",
    "to_trial_dyad",
    "containment_df",
    "fit_individual",
    "fit_dyadic",
    "tukey_pairwise",
    "check_tukey_conservative",
    "compare_alignment",
    # --- figures ---
    "plot_case2_figure",
    "plot_case2_crqa_figure",
    "plot_alignment_comparison_linear",
    "plot_alignment_comparison_crqa_invariant",
    "plot_alignment_comparison_crqa_arms",
]
