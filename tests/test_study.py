"""Tests for the config-driven standard pipeline (StudyConfig + run_study + CLI)."""
import numpy as np
import pandas as pd
import pytest
from pose_dynamics import StudyConfig, run_study
from pose_dynamics.data import example_dataset


def _cfg(tmp_path, **kw):
    ds = example_dataset(2, out_dir=tmp_path / "ds")
    return StudyConfig(data=str(ds), frame_rate=60.0, tau=24, m=3, **kw)


def test_default_per_keypoint_run(tmp_path):
    results, quality = run_study(_cfg(tmp_path, window_s=30.0), progress=False)
    for c in ("mean", "rms", "max", "perc_recur", "perc_determ", "radius_used"):
        assert c in results.columns
    assert "feature" in results.columns
    assert len(quality) == 2


def test_feature_pipeline_run_and_names(tmp_path):
    cfg = _cfg(tmp_path, window_s=30.0, features=[
        {"primitive": "roi_centroid", "params": {"rois": {"left": [0, 1, 2], "right": [3, 4, 5]}}},
        {"primitive": "velocity_magnitude", "params": {"method": "diff"}},
    ])
    results, _ = run_study(cfg, progress=False)
    assert set(results["feature"]) == {"left_speed", "right_speed"}


def test_writes_feature_time_series(tmp_path):
    cfg = _cfg(tmp_path, features_dir=str(tmp_path / "feats"), compute_recurrence=False,
               compute_linear=False,
               features=[{"primitive": "velocity_magnitude", "params": {}}])
    run_study(cfg, progress=False)
    out = sorted((tmp_path / "feats").glob("*_features.csv"))
    assert len(out) == 2
    df = pd.read_csv(out[0])
    assert any("speed" in c for c in df.columns)


def test_config_yaml_roundtrip(tmp_path):
    cfg = _cfg(tmp_path, window_s=30.0, features=[{"primitive": "velocity_magnitude", "params": {}}])
    p = tmp_path / "study.yaml"
    cfg.to_yaml(p)
    loaded = StudyConfig.from_file(p)
    assert loaded.tau == 24 and loaded.m == 3
    assert loaded.features == cfg.features


def test_config_unknown_key_errors(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("data: ./d\nframe_rate: 60\ntau: 20\nm: 4\nnonsense: 1\n")
    with pytest.raises(ValueError, match="Unknown config keys"):
        StudyConfig.from_file(p)


def test_cap_defaults_to_principled_limit():
    assert StudyConfig(data=".", frame_rate=60.0, tau=20, m=4).cap == (4 - 1) * 20


def test_cli_new_config_and_run(tmp_path):
    from pose_dynamics.__main__ import main
    # new-config writes a template
    tmpl = tmp_path / "t.yaml"
    assert main(["new-config", str(tmpl)]) == 0
    assert tmpl.exists()
    # run a real config through the CLI
    ds = example_dataset(1, out_dir=tmp_path / "ds")
    cfg = tmp_path / "run.yaml"
    StudyConfig(data=str(ds), frame_rate=60.0, tau=24, m=3, window_s=30.0,
                output_csv=str(tmp_path / "m.csv")).to_yaml(cfg)
    assert main(["run", str(cfg), "--quiet"]) == 0
    assert (tmp_path / "m.csv").exists()
