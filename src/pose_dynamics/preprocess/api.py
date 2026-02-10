from __future__ import annotations

import inspect
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console

from pose_dynamics.preprocess.alignment import procrustes as procrustes_mod
from pose_dynamics.preprocess.confidence import apply_confidence_mask
from pose_dynamics.preprocess.filtering import apply_detrend_filter
from pose_dynamics.preprocess.missing import interpolate_missing
from pose_dynamics.preprocess.normalize import apply_normalization
from pose_dynamics.preprocess.schema import ConfigError, PreprocessConfig
from pose_dynamics.preprocess.selection import apply_selection
from pose_dynamics.preprocess.spatial import apply_spatial
from pose_dynamics.preprocess.timebase import apply_timebase, ensure_time_column
from pose_dynamics.preprocess.windowing import (
    build_windows,
    score_windows_missingness,
)

console = Console()


@dataclass(frozen=True)
class PreprocessOutputs:
    pose_clean_path: Path
    windows_path: Path
    qc_path: Path
    provenance_path: Path
    alignment_transforms_path: Path | None = None
    alignment_transforms_table_path: Path | None = None


@dataclass(frozen=True)
class AlignmentPlan:
    dims: List[str]
    template_by_trial: Dict[str, np.ndarray]
    keypoints_by_trial: Dict[str, List[str]]
    trial_means: Dict[str, np.ndarray]
    trial_used: Dict[str, List[str]]
    rotation: bool
    scaling: bool
    translation: bool
    reflection: bool
    framewise: bool


def _iter_pose_trials(pose_path: Path) -> Iterator[Tuple[str, pd.DataFrame]]:
    """Yield one concatenated DataFrame per trial across all row-groups."""
    parquet = pq.ParquetFile(pose_path)
    by_trial: dict[str, list[pd.DataFrame]] = defaultdict(list)

    for rg_idx in range(parquet.num_row_groups):
        table = parquet.read_row_group(rg_idx)
        if table.num_rows == 0:
            continue
        df = table.to_pandas()
        if "trial_id" not in df.columns:
            raise ConfigError("pose parquet is missing 'trial_id' column")
        for trial_id, df_trial in df.groupby("trial_id", sort=False):
            by_trial[str(trial_id)].append(df_trial)

    for trial_id, parts in by_trial.items():
        if not parts:
            continue
        yield trial_id, pd.concat(parts, ignore_index=True)


def _callback_accepts_advance(callback: Callable[..., None]) -> bool:
    try:
        sig = inspect.signature(callback)
    except (TypeError, ValueError):
        return False
    try:
        sig.bind_partial("label", 0)
        return True
    except TypeError:
        return False


def _normalize_stage_callback(
    callback: Callable[..., None] | None,
) -> Callable[[str, int], None] | None:
    if callback is None:
        return None
    accepts_advance = _callback_accepts_advance(callback)

    def _wrapped(label: str, advance: int = 0) -> None:
        if accepts_advance:
            callback(label, advance)
        else:
            callback(label)

    return _wrapped


class StreamingPreprocessRunner:
    def __init__(
        self,
        pose_path: Path,
        recording: dict,
        cfg: PreprocessConfig,
        pose_clean_path: Path,
        windows_path: Path,
        transforms_table_path: Path,
        stage_callback: Callable[[str, int], None] | None = None,
        alignment_progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self.pose_path = pose_path
        self.recording = recording
        self.cfg = cfg
        self.pose_clean_path = pose_clean_path
        self.windows_path = windows_path
        self.transforms_table_path = transforms_table_path
        self.stage_callback = stage_callback
        self.alignment_progress_callback = alignment_progress_callback
        self.alignment_progress_total = 0

        self.pose_clean_writer: pq.ParquetWriter | None = None
        self.windows_writer: pq.ParquetWriter | None = None
        self.transforms_table_writer: pq.ParquetWriter | None = None

        self.rows_clean = 0
        self.windows_count = 0
        self.windows_dropped = 0
        self.filter_meta: List[dict] = []
        self.transforms_meta: List[dict] = []
        self.transforms_rows_written = 0
        self.trials_processed = 0
        self.trials_with_rows: set[str] = set()

        self.alignment_plan: AlignmentPlan | None = None

    def _stage(self, label: str, advance: int = 0) -> None:
        if self.stage_callback is not None:
            self.stage_callback(label, advance)

    def _report_alignment_progress(self, advance: int = 0) -> None:
        if (
            self.alignment_progress_callback is None
            or self.alignment_progress_total <= 0
        ):
            return
        self.alignment_progress_callback(advance, self.alignment_progress_total)

    def run(self) -> dict:
        if self.cfg.alignment.enabled:
            self._stage("building alignment templates")
            console.log("Building alignment templates (streaming pass 1)...")
            self.alignment_plan = self._build_alignment_plan()
        console.log("Executing streaming preprocess (pass 2)...")
        self._stage("streaming preprocess trials")
        for trial_id, df_raw in _iter_pose_trials(self.pose_path):
            self.trials_processed += 1
            self._stage(f"trial {trial_id}: pre-alignment pipeline")
            df_stage = self._pre_alignment_stage(df_raw)
            if df_stage.empty:
                continue
            if self.cfg.alignment.enabled:
                self._stage(f"trial {trial_id}: alignment")
            df_aligned, transform_meta, transform_rows = self._apply_alignment(
                trial_id, df_stage
            )
            if transform_meta is not None:
                self.transforms_meta.append(transform_meta)
            if transform_rows is not None and not transform_rows.empty:
                self._write_transforms_table(transform_rows)

            self._stage(f"trial {trial_id}: post-alignment pipeline")
            df_clean, windows_scored = self._post_alignment_pipeline(df_aligned)
            self._stage(f"trial {trial_id}: writing outputs")
            self._write_pose_clean(trial_id, df_clean)
            self._write_windows(windows_scored)

        self._close_writers()
        self._stage("finalizing")

        return {
            "n_rows_clean": self.rows_clean,
            "windows_count": self.windows_count,
            "windows_dropped": self.windows_dropped,
            "filter_meta": self.filter_meta,
            "transforms_meta": self.transforms_meta,
            "transforms_rows_written": self.transforms_rows_written,
            "n_trials_clean": len(self.trials_with_rows),
        }

    def _pre_alignment_stage(self, df: pd.DataFrame) -> pd.DataFrame:
        df_selected = apply_selection(df, self.cfg)
        df_time = apply_timebase(df_selected, self.cfg, self.recording)
        df_conf = apply_confidence_mask(df_time, self.cfg)
        df_interp = interpolate_missing(df_conf, self.cfg)
        df_spatial = apply_spatial(df_interp, self.cfg)
        return df_spatial

    def _build_alignment_plan(self) -> AlignmentPlan:
        dims: List[str] | None = None
        trial_means: Dict[str, np.ndarray] = {}
        trial_used: Dict[str, List[str]] = {}

        trial_chunks: Dict[str, List[pd.DataFrame]] = defaultdict(list)
        for trial_id, df_raw_chunk in _iter_pose_trials(self.pose_path):
            trial_chunks[trial_id].append(df_raw_chunk)

        trial_items = list(trial_chunks.items())
        total_trials = len(trial_items)
        self.alignment_progress_total = total_trials
        if total_trials:
            self._report_alignment_progress(0)

        for idx, (trial_id, chunks) in enumerate(trial_items, start=1):
            if total_trials:
                self._stage(f"building alignment templates ({idx}/{total_trials})")
            else:
                self._stage("building alignment templates")
            if not chunks:
                self._report_alignment_progress(1)
                continue
            df_raw = pd.concat(chunks, ignore_index=True)
            df_stage = self._pre_alignment_stage(df_raw)
            if df_stage.empty:
                self._report_alignment_progress(1)
                continue
            dims_current = [c for c in ["x", "y", "z"] if c in df_stage.columns]
            if not dims_current:
                self._report_alignment_progress(1)
                continue
            if dims is None:
                dims = dims_current

            if self.cfg.alignment.keypoints == "all":
                keypoints = df_stage["keypoint"].dropna().unique().tolist()
            else:
                keypoints = list(self.cfg.alignment.keypoints)

            try:
                mean_pose, used = procrustes_mod._compute_mean_pose(
                    df_stage, keypoints, dims, self.cfg
                )
            except ConfigError as exc:
                console.log(
                    f"[yellow]Skipping trial {trial_id} for alignment template: {exc}"
                )
                self._report_alignment_progress(1)
                continue
            trial_means[trial_id] = mean_pose
            trial_used[trial_id] = used
            self._report_alignment_progress(1)

        if not trial_means:
            raise RuntimeError("alignment enabled but no trials produced valid data")

        rotation, scaling, translation = procrustes_mod._resolve_alignment_flags(
            self.cfg
        )

        if self.cfg.alignment.template_scope == "trial":
            template_by_trial = {tid: trial_means[tid] for tid in trial_means}
            kps_by_trial = {tid: trial_used[tid] for tid in trial_used}
        else:
            kp_sets = [set(kps) for kps in trial_used.values() if kps]
            if not kp_sets:
                raise ConfigError("no valid keypoints for global alignment template")
            kp_intersection = sorted(set.intersection(*kp_sets))
            if len(kp_intersection) < self.cfg.alignment.min_kps_for_fit:
                raise ConfigError(
                    "not enough shared keypoints for global alignment template"
                )
            template_rows = []
            for tid, mean_pose in trial_means.items():
                idx = [trial_used[tid].index(kp) for kp in kp_intersection]
                template_rows.append(mean_pose[idx, :])
            template = np.mean(np.stack(template_rows, axis=0), axis=0)
            template_by_trial = {tid: template for tid in trial_means}
            kps_by_trial = {tid: kp_intersection for tid in trial_means}

        return AlignmentPlan(
            dims=dims or ["x", "y"],
            template_by_trial=template_by_trial,
            keypoints_by_trial=kps_by_trial,
            trial_means=trial_means,
            trial_used=trial_used,
            rotation=rotation,
            scaling=scaling,
            translation=translation,
            reflection=self.cfg.alignment.reflection,
            framewise=bool(self.cfg.alignment.framewise),
        )

    def _apply_alignment(
        self, trial_id: str, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, dict | None, pd.DataFrame | None]:
        if not self.cfg.alignment.enabled:
            return df, None, None
        if self.alignment_plan is None:
            raise RuntimeError("alignment plan missing")
        plan = self.alignment_plan
        if trial_id not in plan.template_by_trial:
            raise RuntimeError(f"alignment template missing for trial {trial_id}")

        if not plan.framewise:
            mean_pose = plan.trial_means[trial_id]
            used = plan.trial_used[trial_id]
            template = plan.template_by_trial[trial_id]
            kps = plan.keypoints_by_trial[trial_id]

            if kps != used:
                idx_trial = [used.index(kp) for kp in kps]
                X = mean_pose[idx_trial, :]
            else:
                X = mean_pose
            Y = template

            R, s, t = procrustes_mod._procrustes_transform(
                X,
                Y,
                allow_reflection=self.cfg.alignment.reflection,
                rotation=plan.rotation,
                scaling=plan.scaling,
                translation=plan.translation,
            )
            df_aligned = procrustes_mod._apply_transform(df, plan.dims, R, s, t)

            d = len(plan.dims)
            T = np.eye(d + 1)
            T[:d, :d] = s * R
            T[:d, d] = t

            transform_meta = {
                "trial_id": trial_id,
                "dims": plan.dims,
                "keypoints_used": kps,
                "rotation": plan.rotation,
                "scaling": plan.scaling,
                "translate": plan.translation,
                "reflection": self.cfg.alignment.reflection,
                "transform": self.cfg.alignment.transform,
                "template_scope": self.cfg.alignment.template_scope,
                "framewise": False,
                "scale": float(s),
                "rotation_matrix": R.tolist(),
                "translation": t.tolist(),
                "transform_matrix": T.tolist(),
            }
            return df_aligned, transform_meta, None

        transform_rows: List[dict] = []
        aligned_parts: List[pd.DataFrame] = []
        time_col = "time" if "time" in df.columns else "frame"
        template = plan.template_by_trial[trial_id]
        kps = plan.keypoints_by_trial[trial_id]

        for tval, df_frame in df.groupby(time_col, sort=False):
            Xf, used = procrustes_mod._frame_pose(df_frame, kps, plan.dims)
            if len(used) < self.cfg.alignment.min_kps_for_fit:
                # Not enough keypoints to fit; keep original values instead of blanking the frame.
                row = {
                    "trial_id": trial_id,
                    time_col: tval,
                    "scale": float("nan"),
                    "rotation_angle": float("nan"),
                    **{f"translation_{d}": float("nan") for d in plan.dims},
                    "keypoints_used": used,
                }
                transform_rows.append(row)
                aligned_parts.append(df_frame)
                continue

            idx_template = [kps.index(kp) for kp in used]
            Y = template[idx_template, :]
            R, s, t_vec = procrustes_mod._procrustes_transform(
                Xf,
                Y,
                allow_reflection=self.cfg.alignment.reflection,
                rotation=plan.rotation,
                scaling=plan.scaling,
                translation=plan.translation,
            )
            df_aligned = procrustes_mod._apply_transform(
                df_frame, plan.dims, R, s, t_vec
            )
            aligned_parts.append(df_aligned)

            row = {
                "trial_id": trial_id,
                time_col: tval,
                "scale": float(s),
                "rotation_angle": procrustes_mod._rotation_angle_2d(R),
                **{
                    f"translation_{d}": float(t_vec[i]) for i, d in enumerate(plan.dims)
                },
                "keypoints_used": used,
            }
            transform_rows.append(row)

        transforms_df = pd.DataFrame(transform_rows)
        transforms_meta = {
            "trial_id": trial_id,
            "dims": plan.dims,
            "keypoints_used": kps,
            "rotation": plan.rotation,
            "scaling": plan.scaling,
            "translate": plan.translation,
            "reflection": self.cfg.alignment.reflection,
            "transform": self.cfg.alignment.transform,
            "template_scope": self.cfg.alignment.template_scope,
            "framewise": True,
        }
        df_aligned = (
            pd.concat(aligned_parts, ignore_index=True) if aligned_parts else df
        )
        return df_aligned, transforms_meta, transforms_df

    def _post_alignment_pipeline(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        windows = pd.DataFrame()
        df_clean = df

        if (
            self.cfg.normalization.enabled
            and self.cfg.normalization.scope == "windowed"
        ):
            df_for_windows = df_clean
            if self.cfg.windowing.enabled and self.cfg.windowing.units == "seconds":
                if "time" not in df_for_windows.columns:
                    df_for_windows = ensure_time_column(
                        df_for_windows, self.cfg, self.recording
                    )
            windows = build_windows(df_for_windows, self.cfg, self.recording)
            df_clean = apply_normalization(df_for_windows, self.cfg, windows=windows)
        else:
            df_clean = apply_normalization(df_clean, self.cfg, windows=None)

        df_clean, filter_meta = apply_detrend_filter(df_clean, self.cfg)
        self.filter_meta.extend(filter_meta)

        df_for_windowing = df_clean
        if self.cfg.windowing.enabled and self.cfg.windowing.units == "seconds":
            if "time" not in df_for_windowing.columns:
                df_for_windowing = ensure_time_column(
                    df_for_windowing, self.cfg, self.recording
                )

        if windows.empty:
            windows = build_windows(df_for_windowing, self.cfg, self.recording)
        windows_scored = score_windows_missingness(df_for_windowing, windows, self.cfg)
        return df_clean, windows_scored

    def _write_pose_clean(self, trial_id: str, df: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self.pose_clean_writer is None:
            self.pose_clean_writer = pq.ParquetWriter(
                str(self.pose_clean_path), table.schema, compression="snappy"
            )
        self.pose_clean_writer.write_table(table)
        self.rows_clean += len(df)
        if not df.empty:
            self.trials_with_rows.add(trial_id)

    def _write_windows(self, df: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self.windows_writer is None:
            self.windows_writer = pq.ParquetWriter(
                str(self.windows_path), table.schema, compression="snappy"
            )
        self.windows_writer.write_table(table)
        self.windows_count += len(df)
        if not df.empty and "dropped" in df.columns:
            self.windows_dropped += int(df["dropped"].sum())

    def _write_transforms_table(self, df: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self.transforms_table_writer is None:
            self.transforms_table_writer = pq.ParquetWriter(
                str(self.transforms_table_path),
                table.schema,
                compression="snappy",
            )
        self.transforms_table_writer.write_table(table)
        self.transforms_rows_written += len(df)

    def _close_writers(self) -> None:
        for writer in (
            self.pose_clean_writer,
            self.windows_writer,
            self.transforms_table_writer,
        ):
            if writer is not None:
                writer.close()


def run_preprocess(
    pose_path: str | Path,
    recording_path: str | Path,
    config: PreprocessConfig | str | Path,
    out_dir: str | Path,
    *,
    overwrite: bool = False,
    stage_callback: Callable[[str], None] | Callable[[str, int], None] | None = None,
    alignment_progress_callback: Callable[[int, int], None] | None = None,
) -> PreprocessOutputs:
    """
    Run preprocessing for canonical pose data.

    Inputs:
      - pose_path: canonical pose.parquet from ingest
      - recording_path: recording.json from ingest
      - config: PreprocessConfig OR path to preprocess.yml
      - out_dir: output directory

    Outputs (written to out_dir):
      - pose_clean.parquet
      - windows.parquet
      - qc_preprocess.json
      - provenance.json (resolved config + hashes + run metadata)
    """
    normalized_stage_callback = _normalize_stage_callback(stage_callback)

    def _report(label: str) -> None:
        if normalized_stage_callback is not None:
            normalized_stage_callback(label, 0)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    console.log(f"Preprocess outputs will be written to {out_dir}")

    if isinstance(config, PreprocessConfig):
        cfg = config
    else:
        console.log(f"Loading preprocess config: {config}")
        _report("loading preprocess config")
        cfg = PreprocessConfig.from_yaml(str(config))

    pose_clean_path = out_dir / "pose_clean.parquet"
    windows_path = out_dir / "windows.parquet"
    qc_path = out_dir / "qc_preprocess.json"
    provenance_path = out_dir / "provenance.json"
    alignment_transforms_path = out_dir / "alignment_transforms.json"
    alignment_transforms_table_path = out_dir / "alignment_transforms.parquet"

    if not overwrite:
        required = [pose_clean_path, windows_path, qc_path, provenance_path]
        if cfg.alignment.enabled:
            required.append(alignment_transforms_path)
            if cfg.alignment.framewise:
                required.append(alignment_transforms_table_path)
        for p in required:
            if p.exists():
                raise FileExistsError(f"Output already exists: {p}")
    pose_path = Path(pose_path)
    recording_path = Path(recording_path)

    with recording_path.open("r", encoding="utf-8") as f:
        recording = json.load(f)
    console.log(f"Loaded recording metadata: {recording_path}")
    _report("loaded recording metadata")

    runner = StreamingPreprocessRunner(
        pose_path=pose_path,
        recording=recording,
        cfg=cfg,
        pose_clean_path=pose_clean_path,
        windows_path=windows_path,
        transforms_table_path=alignment_transforms_table_path,
        stage_callback=normalized_stage_callback,
        alignment_progress_callback=alignment_progress_callback,
    )
    _report("starting preprocess runner")
    summary = runner.run()

    qc_payload = {
        "summary": {
            "n_windows": int(summary["windows_count"]),
            "n_dropped": int(summary["windows_dropped"]),
            "n_trials": int(summary["n_trials_clean"]),
            "filtering": summary["filter_meta"],
        },
        "n_rows_clean": int(summary["n_rows_clean"]),
    }
    qc_path.write_text(json.dumps(qc_payload, indent=2), encoding="utf-8")

    provenance_payload = {
        "pose_path": str(pose_path.resolve()),
        "recording_path": str(recording_path.resolve()),
        "config": cfg.to_dict(),
    }
    provenance_path.write_text(
        json.dumps(provenance_payload, indent=2), encoding="utf-8"
    )
    _report("writing provenance and qc")

    if cfg.alignment.enabled:
        console.log("Writing alignment transforms outputs")
        alignment_transforms_path.write_text(
            json.dumps(summary["transforms_meta"], indent=2), encoding="utf-8"
        )
        if cfg.alignment.framewise and summary["transforms_rows_written"] > 0:
            align_table_path = alignment_transforms_table_path
        else:
            align_table_path = None
        align_path = alignment_transforms_path
    else:
        align_path = None
        align_table_path = None

    console.log("Preprocess artifacts written successfully")
    _report("preprocess complete")
    return PreprocessOutputs(
        pose_clean_path=pose_clean_path,
        windows_path=windows_path,
        qc_path=qc_path,
        provenance_path=provenance_path,
        alignment_transforms_path=align_path,
        alignment_transforms_table_path=align_table_path,
    )
