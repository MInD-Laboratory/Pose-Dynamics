"""
Computational-cost benchmark for pose-dynamics.

Two parts:
  1. Microbenchmarks (synthetic): RQA and MdRQA scaling with signal length N and
     dimensionality, and the fixed-%REC (bisection) multiplier.
  2. End-to-end per-trial cost for Case 1 (MATB), Case 2 (MOSAIC) and Case 3
     (Mirror Game), if the data directories are given.

Usage:
    python benchmarks/benchmark.py                                  # microbenchmarks only
    python benchmarks/benchmark.py --matb DIR --mosaic DIR --mg DIR # + end-to-end
"""
from __future__ import annotations

import argparse
import glob
import time
from statistics import median

import numpy as np

from pose_dynamics.rqa import (
    RqaParams,
    run_auto_rqa,
    run_cross_rqa,
    run_multivariate_cross_rqa,
)

rng = np.random.default_rng(0)


def _t(fn, reps=3):
    ts = []
    for _ in range(reps):
        a = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - a)
    return median(ts)


def microbenchmarks():
    print("=== auto-RQA (fixed radius, m=4 tau=20) vs signal length N ===")
    for N in (500, 1000, 2000, 4000):
        x = np.sin(2 * np.pi * np.arange(N) / 40) + 0.1 * rng.standard_normal(N)
        p = RqaParams(eDim=4, tLag=20, radius_mode="fixed_radius", radius=0.2,
                      rescale="mean", norm="zscore", min_line=2)
        print(f"  N={N:5d}: {_t(lambda: run_auto_rqa(x, p)) * 1000:7.1f} ms")

    print("\n=== fixed-radius vs fixed-%REC bisection (cross-RQA, N=1000) ===")
    x = np.sin(2 * np.pi * np.arange(1000) / 40) + 0.1 * rng.standard_normal(1000)
    y = np.sin(2 * np.pi * np.arange(1000) / 40 + 0.3) + 0.1 * rng.standard_normal(1000)
    base = dict(eDim=4, tLag=20, rescale="mean", norm="zscore", min_line=2)
    tf = _t(lambda: run_cross_rqa(x, y, RqaParams(radius_mode="fixed_radius", radius=0.2, **base)))
    tr = _t(lambda: run_cross_rqa(x, y, RqaParams(radius_mode="fixed_rrec", target_rec=2.5, **base)))
    print(f"  fixed_radius : {tf * 1000:7.1f} ms")
    print(f"  fixed_rrec   : {tr * 1000:7.1f} ms  ({tr / tf:.1f}x  <- bisection cost)")

    print("\n=== MdRQA (multivariate cross, fixed radius) vs N and dims ===")
    print("       dims=2     dims=5     dims=10")
    for N in (500, 1000, 2000):
        row = []
        for d in (2, 5, 10):
            X = rng.standard_normal((N, d))
            Y = X + 0.1 * rng.standard_normal((N, d))
            pm = RqaParams(eDim=4, tLag=20, radius_mode="fixed_radius", radius=0.3,
                           rescale="mean", norm="zscore", min_line=2, multivariate=True)
            row.append(_t(lambda: run_multivariate_cross_rqa(X, Y, pm)) * 1000)
        print(f"  N={N:5d}: {row[0]:8.1f}  {row[1]:8.1f}  {row[2]:8.1f}   (ms)")


def case1(matb_dir):
    from pose_dynamics.case_studies.matb import load_matb_file, preprocess
    from pose_dynamics.case_studies.matb.reproduce import build_global_template, process_sequence

    f = sorted(glob.glob(f"{matb_dir}/*.csv"))[0]
    a = time.perf_counter(); seq = load_matb_file(f); t_load = time.perf_counter() - a
    a = time.perf_counter(); seq = preprocess(seq); t_prep = time.perf_counter() - a
    tmpl = build_global_template([seq])
    a = time.perf_counter(); rows = process_sequence(seq, tmpl); t_proc = time.perf_counter() - a
    total = t_load + t_prep + t_proc
    print("\n=== CASE 1 (MATB): one trial ===")
    print(f"  frames={seq.n_frames}, window-rows={len(rows)}")
    print(f"  load {t_load:.2f}s | preprocess {t_prep:.2f}s | features+RQA {t_proc:.2f}s")
    print(f"  per trial {total:.2f}s  ->  x216 ~ {total * 216 / 60:.1f} min")


def _mosaic_dyad(right_path, left_path):
    """Time one MOSAIC dyad-trial by stage; returns ``(load, preprocess, proc, info)``.

    The stages are inlined rather than calling ``process_dyad``, which preprocesses
    internally and would double-count that stage against the load/preprocess figures
    reported separately. Keep this in step with ``process_dyad`` if that changes.

    The global Procrustes template is a dataset-level fixed cost, not a per-trial one
    (``TEMPLATE_SAMPLE=None`` builds it from every file), so it is excluded from the
    per-trial total and reported separately -- the same treatment ``case1`` gives
    ``build_global_template``. Here it is built from this trial's own two files, which
    costs the same per window as the real dataset-wide template and keeps each trial
    independently timeable.
    """
    from pose_dynamics.case_studies.mosaic.reproduce import (
        _window_roi_speeds,
        build_global_template,
        cross_params,
        load_mosaic_file,
        preprocess_pose,
        windowed_align,
    )
    from pose_dynamics.data.pose_sequence import PoseSequence

    cp = cross_params()
    t_load, t_prep, t_proc = [], [], []
    template = None
    for _ in range(3):
        a = time.perf_counter()
        right, roi_map = load_mosaic_file(right_path)
        left, _ = load_mosaic_file(left_path)
        t_load.append(time.perf_counter() - a)
        raw_frames = right.n_frames

        a = time.perf_counter()
        right = preprocess_pose(right)
        left = preprocess_pose(left)
        t_prep.append(time.perf_counter() - a)

        n = min(right.n_frames, left.n_frames)
        right = PoseSequence(coords=right.coords[:n], keypoint_names=right.keypoint_names,
                             frame_rate=right.frame_rate)
        left = PoseSequence(coords=left.coords[:n], keypoint_names=left.keypoint_names,
                            frame_rate=left.frame_rate)
        if template is None:
            template = build_global_template([right, left])

        a = time.perf_counter()
        n_crqa = n_windows = 0
        for (w, aligned_r), (_, aligned_l) in zip(windowed_align(right, template),
                                                  windowed_align(left, template)):
            n_windows += 1
            feats_r = _window_roi_speeds(aligned_r, right.keypoint_names, roi_map, right.frame_rate)
            feats_l = _window_roi_speeds(aligned_l, left.keypoint_names, roi_map, left.frame_rate)
            for roi in roi_map:
                aw, bw = feats_r.get(f"{roi}_speed"), feats_l.get(f"{roi}_speed")
                if not (np.all(np.isfinite(aw)) and np.all(np.isfinite(bw))):
                    continue
                run_cross_rqa(aw, bw, cp)
                n_crqa += 1
        t_proc.append(time.perf_counter() - a)

    return (median(t_load), median(t_prep), median(t_proc),
            {"raw_frames": raw_frames, "frames": n, "windows": n_windows, "crqa": n_crqa})


def case2(mosaic_dir, n_trials=1, n_dyad_trials=272, n_files=550):
    """End-to-end MOSAIC cost, over the first ``n_trials`` dyad-trials.

    Trial length is bimodal in this dataset (~5 min and ~8 min recordings), so a
    single trial misrepresents it; the full-dataset estimate below is therefore
    built from a per-window fit rather than by multiplying one trial's total.
    """
    from pose_dynamics.case_studies.mosaic.reproduce import parse_file

    by_key = {}
    for p in sorted(glob.glob(f"{mosaic_dir}/S*_T*_*.csv")):
        if p.replace("\\", "/").rsplit("/", 1)[-1].startswith("._"):
            continue
        s, t, cam = parse_file(p)
        by_key.setdefault((s, t), {})[cam] = p
    keys = sorted(k for k, v in by_key.items() if {"left", "right"} <= set(v))[:n_trials]

    print("\n=== CASE 2 (MOSAIC): per dyad-trial ===")
    rows = []
    for (s, t) in keys:
        ld, pr, pc, info = _mosaic_dyad(by_key[(s, t)]["right"], by_key[(s, t)]["left"])
        total = ld + pr + pc
        rows.append((info["windows"], ld, pr, pc, total))
        print(f"  S{s:03d}_T{t}: {info['raw_frames']} frames @60Hz -> "
              f"{info['frames']} @30Hz, {info['windows']} windows, "
              f"{info['crqa']} CRQA runs (<=3 ROIs x windows)")
        print(f"    load x2 {ld:.2f}s | preprocess x2 {pr:.2f}s | "
              f"align+features+CRQA {pc:.2f}s | per dyad-trial {total:.2f}s")

    per_window = median(r[4] / r[0] for r in rows)
    per_file = median((r[1] + r[2]) / 2 for r in rows)
    print(f"  median {per_window:.2f}s per window; {sum(r[4] for r in rows) / len(rows):.2f}s "
          f"mean per dyad-trial over {len(rows)} trial(s)")
    print(f"  full dataset ({n_dyad_trials} dyad-trials, ~11.5 windows median) ~ "
          f"{per_window * 11.5 * n_dyad_trials / 60:.0f} min")
    print(f"  + global template pass: {per_file:.2f}s/file x{n_files} ~ "
          f"{per_file * n_files / 60:.0f} min (one-off, dataset-level)")


def case3(mg_dir):
    import re
    from pose_dynamics.case_studies.mirror_game import load_and_resample
    from pose_dynamics.case_studies.mirror_game.reproduce import _prep_subset, cross_params

    files = {re.search(r"P\d+_T\d+_P(\d)", x).group(1): x
             for x in glob.glob(f"{mg_dir}/P001_T1_P*_pose_3d.csv") if "/._" not in x}
    a = time.perf_counter(); p1 = load_and_resample(files["1"]); p2 = load_and_resample(files["2"]); t_load = time.perf_counter() - a
    a = time.perf_counter(); aa = _prep_subset(p1); bb = _prep_subset(p2); t_prep = time.perf_counter() - a
    n = min(aa.n_frames, bb.n_frames); cp = cross_params()
    a = time.perf_counter()
    for k in range(5):
        run_cross_rqa(np.linalg.norm(aa.coords[:n, k, :], axis=1),
                      np.linalg.norm(bb.coords[:n, k, :], axis=1), cp)
    t_crqa = time.perf_counter() - a
    total = t_load + t_prep + t_crqa
    print("\n=== CASE 3 (Mirror Game): one dyad-trial ===")
    print(f"  frames={n}")
    print(f"  load+resample x2 {t_load:.2f}s | center+filter x2 {t_prep:.2f}s | 5x CRQA (bisection) {t_crqa:.2f}s")
    print(f"  per dyad-trial {total:.2f}s  ->  x210 ~ {total * 210 / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--matb")
    ap.add_argument("--mosaic")
    ap.add_argument("--mosaic-trials", type=int, default=1,
                    help="dyad-trials to time for Case 2 (length is bimodal; >=4 recommended)")
    ap.add_argument("--mg")
    ap.add_argument("--skip-micro", action="store_true")
    args = ap.parse_args()
    if not args.skip_micro:
        microbenchmarks()
    if args.matb:
        case1(args.matb)
    if args.mosaic:
        case2(args.mosaic, n_trials=args.mosaic_trials)
    if args.mg:
        case3(args.mg)
