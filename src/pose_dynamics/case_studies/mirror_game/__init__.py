"""Case 3 (Mirror Game) reproduction: config and entry points."""
from __future__ import annotations

from . import config
from .pca import (
    build_aligned_dataset,
    plot_principal_movements,
    plot_variance,
    run_pca_diagnostic,
)
from .reproduce import (
    load_and_resample,
    default_conditions_csv,
    load_condition_map,
    parse_file,
    plot_case3_figure,
    process_dyad,
    run_reproduction,
)

__all__ = [
    "config",
    "load_and_resample",
    "default_conditions_csv",
    "load_condition_map",
    "parse_file",
    "process_dyad",
    "run_reproduction",
    "plot_case3_figure",
    "build_aligned_dataset",
    "run_pca_diagnostic",
    "plot_variance",
    "plot_principal_movements",
]
