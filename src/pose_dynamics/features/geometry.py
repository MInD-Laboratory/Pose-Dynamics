from __future__ import annotations

from itertools import combinations
from typing import List

import numpy as np
import pandas as pd


def pairwise_distance_features(
    df_win: pd.DataFrame, time_col: str, dims: List[str], keypoints: List[str]
) -> dict:
    out: dict[str, float] = {}
    if df_win.empty or len(keypoints) < 2:
        return out

    # Pivot to wide: index time, columns (dim, keypoint)
    pivot = df_win.pivot_table(index=time_col, columns="keypoint", values=dims)

    for kp1, kp2 in combinations(keypoints, 2):
        if kp1 not in pivot.columns.levels[1] or kp2 not in pivot.columns.levels[1]:
            continue
        v1 = np.stack([pivot[(d, kp1)].to_numpy() for d in dims], axis=1)
        v2 = np.stack([pivot[(d, kp2)].to_numpy() for d in dims], axis=1)
        mask = np.isnan(v1).any(axis=1) | np.isnan(v2).any(axis=1)
        v1 = v1[~mask]
        v2 = v2[~mask]
        if v1.shape[0] == 0:
            continue
        dist = np.linalg.norm(v1 - v2, axis=1)
        out[f"dist_{kp1}_{kp2}_mean"] = float(np.nanmean(dist))
        out[f"dist_{kp1}_{kp2}_std"] = float(np.nanstd(dist))

    return out
