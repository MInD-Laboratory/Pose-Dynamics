from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def head_motion_series(
    df_transforms: pd.DataFrame, time_col: str
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    if df_transforms.empty:
        return out

    if "scale" in df_transforms.columns:
        out["head_scale"] = df_transforms["scale"].to_numpy(dtype=float)
    if "rotation_angle" in df_transforms.columns:
        out["head_rotation"] = df_transforms["rotation_angle"].to_numpy(dtype=float)

    tx = df_transforms[[c for c in df_transforms.columns if c == "translation_x"]]
    ty = df_transforms[[c for c in df_transforms.columns if c == "translation_y"]]

    if not tx.empty or not ty.empty:
        txv = (
            df_transforms["translation_x"].to_numpy(dtype=float)
            if "translation_x" in df_transforms.columns
            else np.zeros(len(df_transforms))
        )
        tyv = (
            df_transforms["translation_y"].to_numpy(dtype=float)
            if "translation_y" in df_transforms.columns
            else np.zeros(len(df_transforms))
        )
        out["head_tx"] = txv
        out["head_ty"] = tyv
        out["head_translation_mag"] = np.sqrt(txv**2 + tyv**2)

    if "head_translation_mag" in out or "head_rotation" in out or "head_scale" in out:
        components = []
        if "head_translation_mag" in out:
            components.append(out["head_translation_mag"])
        if "head_rotation" in out:
            components.append(out["head_rotation"])
        if "head_scale" in out:
            components.append(out["head_scale"] - 1.0)
        if components:
            stack = np.vstack(components)
            out["head_motion_mag"] = np.sqrt(np.nansum(stack**2, axis=0))

    return out
