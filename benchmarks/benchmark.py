"""
Computational-cost benchmark for pose-dynamics.

Two parts:
  1. Microbenchmarks (synthetic): RQA and MdRQA scaling with signal length N and
     dimensionality, and the fixed-%REC (bisection) multiplier.
  2. End-to-end per-trial cost for Case 1 (MATB) and Case 3 (Mirror Game), if the
     data directories are given.

Usage:
    python benchmarks/benchmark.py                      # microbenchmarks only
    python benchmarks/benchmark.py --matb DIR --mg DIR  # + end-to-end
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
    ap.add_argument("--mg")
    args = ap.parse_args()
    microbenchmarks()
    if args.matb:
        case1(args.matb)
    if args.mg:
        case3(args.mg)
