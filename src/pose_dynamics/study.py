"""
The standard pipeline — describe a run in a config file, get features/metrics out.

This is the config-driven Tier 0/1 entry point (build plan §6). A
:class:`StudyConfig` — written as a YAML/JSON file and edited, not coded — describes
the data, the preprocessing, an optional **feature pipeline** (the composable
primitives), and what to compute. :func:`run_study` runs it over a folder of
canonical CSVs and writes:

- **feature time series** — one CSV per input file (what most users want: "get my
  features over a bunch of CSVs"); and/or
- a tidy **metrics table** — linear (magnitude) and recurrence (organization)
  summaries per file × feature × window; plus a per-trial data-quality report.

Run it from a notebook (``run_study``) or the command line
(``pose-dynamics run study.yaml``). It is single-person by design; interpersonal
cross-recurrence is dyadic and lives in the case-study layer.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

from .data.loader import load_pose_csv
from .features import FeaturePipeline
from .linear import summarise_signal
from .preprocessing import (
    assess_quality,
    butterworth_filter,
    combine_reports,
    interpolate_gaps,
    mask_low_confidence,
)
from .rqa import RqaParams, run_auto_rqa
from .windowing import make_windows

LINEAR_STATS = ("mean", "std", "rms", "max")
_RQA_METRICS = ("perc_recur", "perc_determ", "laminarity",
                "mean_line_length", "maxl_found", "entropy")


@dataclass
class StudyConfig:
    """A serializable, human-editable description of a standard-pipeline run."""

    data: str                         # folder or glob of canonical CSVs
    frame_rate: float                 # Hz (required; not inferred)
    tau: int                          # committed embedding delay
    m: int                            # committed embedding dimension

    # preprocessing
    conf_threshold: float = 0.30
    interp_cap: int | None = None     # None -> (m-1)*tau
    filter_cutoff: float = 10.0
    filter_order: int = 4

    # feature pipeline — a list of {primitive, params}. None -> per-keypoint speed.
    features: list[dict] | None = None
    default_signal: str = "speed"     # used only when `features` is None

    # what to compute
    compute_linear: bool = True
    compute_recurrence: bool = True

    # recurrence
    radius_mode: str = "fixed_rrec"
    target_rec: float = 5.0
    radius: float | None = None
    rescale: str = "mean"
    norm: str = "zscore"
    min_line: int = 2                 # l_min (minimum diagonal/vertical line length)
    theiler: int | None = None        # Theiler window for auto-RQA (None -> tau)

    # windowing (None -> whole trial as one window)
    window_s: float | None = None
    overlap: float = 0.5

    # data quality
    max_missing_frac: float = 0.30
    on_exceed: str = "flag"

    # output
    features_dir: str | None = None   # write per-file feature time series here
    output_csv: str | None = None     # write the windowed metrics table here

    def __post_init__(self) -> None:
        if self.frame_rate is None or self.frame_rate <= 0:
            raise ValueError("frame_rate (Hz) is required and must be positive.")
        if self.features is None and self.default_signal not in ("speed", "displacement"):
            raise ValueError("default_signal must be 'speed' or 'displacement'.")
        if self.radius_mode == "fixed_radius" and self.radius is None:
            raise ValueError("fixed_radius mode needs a radius.")

    # --- config file I/O ------------------------------------------------
    @classmethod
    def from_file(cls, path: str | Path) -> "StudyConfig":
        """Load a config from a ``.yaml``/``.yml`` or ``.json`` file."""
        path = Path(path)
        text = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            import yaml
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"Unknown config keys: {sorted(unknown)}. Valid keys: {sorted(known)}."
            )
        return cls(**data)

    def to_yaml(self, path: str | Path | None = None) -> str:
        import yaml
        text = yaml.safe_dump(self.to_dict(), sort_keys=False)
        if path is not None:
            Path(path).write_text(text)
        return text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # --- resolved values ------------------------------------------------
    @property
    def cap(self) -> int:
        """The interpolation cap: explicit, or the principled ``(m-1)·τ``."""
        return self.interp_cap if self.interp_cap is not None else (self.m - 1) * self.tau

    def rqa_params(self) -> RqaParams:
        kw = dict(eDim=self.m, tLag=self.tau, rescale=self.rescale,
                  norm=self.norm, min_line=self.min_line, theiler=self.theiler)
        if self.radius_mode == "fixed_radius":
            return RqaParams(radius_mode="fixed_radius", radius=self.radius, **kw)
        return RqaParams(radius_mode="fixed_rrec", target_rec=self.target_rec, **kw)

    def files(self) -> list[Path]:
        p = Path(self.data)
        if p.is_dir():
            return sorted(f for f in p.glob("*.csv") if not f.name.startswith("._"))
        return sorted(Path().glob(self.data))


# ----------------------------------------------------------------------
def _default_signals(coords: np.ndarray, fps: float, kind: str) -> tuple[list[str], np.ndarray]:
    """Per-keypoint 1-D movement signal (used when no feature pipeline is given)."""
    if kind == "displacement":
        d = np.diff(coords, axis=0, prepend=coords[:1])
        sig = np.linalg.norm(d, axis=-1)
    else:
        sig = np.linalg.norm(np.gradient(coords, axis=0) * fps, axis=-1)
    return sig  # (T, K)


def run_study(config: StudyConfig, progress: bool = True):
    """Run the standard pipeline over a dataset. Returns ``(results, quality)``."""
    import pandas as pd

    files = config.files()
    if not files:
        raise FileNotFoundError(f"No canonical CSVs found at {config.data!r}.")

    if config.features_dir:
        Path(config.features_dir).mkdir(parents=True, exist_ok=True)

    rp = config.rqa_params() if config.compute_recurrence else None
    rows: list[dict] = []
    reports = []

    for fi, f in enumerate(files):
        seq = load_pose_csv(f, frame_rate=config.frame_rate)
        seq = mask_low_confidence(seq, config.conf_threshold)
        seq = interpolate_gaps(seq, config.cap)
        seq = butterworth_filter(seq, config.filter_cutoff, config.filter_order)

        report = assess_quality(seq, max_missing_frac=config.max_missing_frac,
                                on_exceed=config.on_exceed)
        reports.append(report)
        if progress:
            print(f"[{fi + 1}/{len(files)}] {f.name}: {report.status}")
        if report.excluded:
            continue

        # --- extract feature signals (named) ---
        if config.features:
            fs = FeaturePipeline.from_config(config.features).run(seq).features
            if fs is None:
                raise ValueError(
                    f"The feature pipeline produced no signals for {f.name}. End it "
                    "with a primitive that produces SIGNALS (e.g. velocity_magnitude)."
                )
            names, values, fps = fs.names, fs.values, fs.frame_rate
            if config.features_dir:
                fs.to_dataframe().to_csv(Path(config.features_dir) / f"{f.stem}_features.csv", index=False)
        else:
            sig = _default_signals(seq.coords, seq.frame_rate, config.default_signal)
            names, values, fps = list(seq.keypoint_names), sig, seq.frame_rate
            if config.features_dir:
                pd.DataFrame(values, columns=[f"{n}_{config.default_signal}" for n in names]) \
                    .to_csv(Path(config.features_dir) / f"{f.stem}_features.csv", index=False)

        n_frames = values.shape[0]
        if config.window_s is None:
            windows = make_windows(n_frames, fps, n_frames / fps, 0.0)
        else:
            windows = make_windows(n_frames, fps, config.window_s, config.overlap)

        # --- per feature x window: linear + recurrence metrics ---
        if not (config.compute_linear or config.compute_recurrence):
            continue
        for j, name in enumerate(names):
            col = values[:, j]
            for w in windows:
                s = col[w.start:w.stop]
                if not np.all(np.isfinite(s)):
                    continue
                row: dict[str, Any] = {
                    "file": f.name, "feature": name, "window": w.index,
                    "t_start": round(w.t_start, 3), "t_end": round(w.t_stop, 3),
                }
                if config.compute_linear:
                    row.update(summarise_signal(s, stats=LINEAR_STATS))
                if rp is not None:
                    res = run_auto_rqa(s, rp)
                    row.update({k: res.metrics[k] for k in _RQA_METRICS})
                    row["radius_used"] = res.radius_used
                rows.append(row)

    results = pd.DataFrame(rows)
    quality = combine_reports(reports)
    if config.output_csv and len(results):
        results.to_csv(config.output_csv, index=False)
        if progress:
            print(f"wrote {len(results)} metric rows to {config.output_csv}")
    if config.features_dir and progress:
        print(f"wrote per-file feature time series to {config.features_dir}/")
    return results, quality
