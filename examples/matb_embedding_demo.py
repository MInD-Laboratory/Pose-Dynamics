"""
End-to-end driver: load MATB OpenPose data -> preprocess -> embedding evidence.

This exercises steps 1-3 of the pipeline on real Case 1 data and produces the
human-in-the-loop embedding presentation (AMI/FNN evidence + variability), so you
can judge whether it is good enough to commit (tau, m) from.

Run
---
    python examples/matb_embedding_demo.py \
        --data-dir "/Volumes/X9_Pro/.../experimental_pose" \
        --n-trials 6 --out-dir ./embedding_out

Then open the two PNGs in --out-dir. In a Jupyter notebook, call the same
functions and the figures display inline (that is the intended interface).
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import matplotlib

from openpose_to_canonical import convert_file

from pose_dynamics import load_pose_csv
from pose_dynamics.preprocessing import (
    butterworth_filter,
    interpolate_gaps,
    mask_low_confidence,
)
from pose_dynamics.embedding import (
    magnitude_channels,
    plot_embedding_evidence,
    plot_embedding_variability,
    pool_signals,
    select_embedding,
)

# Case 1 numeric settings (see numeric_inventory.md §1, §7).
CONF_THRESHOLD = 0.30
INTERP_CAP = 60          # frames (1 s @ 60 Hz)
FILTER_CUTOFF = 10.0     # Hz
FILTER_ORDER = 4
FRAME_RATE = 60.0


def build_sequences(data_dir: str, n_trials: int, work_dir: Path):
    work_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))[:n_trials]
    if not files:
        raise SystemExit(f"No CSVs found in {data_dir}")
    seqs = []
    for f in files:
        stem = Path(f).stem  # e.g. "402_L"
        canonical = convert_file(f, work_dir / f"{stem}_canonical.csv")
        participant, _, condition = stem.partition("_")
        seq = load_pose_csv(
            canonical, frame_rate=FRAME_RATE,
            meta={"participant": participant, "condition": condition},
        )
        seq = mask_low_confidence(seq, threshold=CONF_THRESHOLD)
        seq = interpolate_gaps(seq, max_gap=INTERP_CAP)
        seq = butterworth_filter(seq, cutoff_hz=FILTER_CUTOFF, order=FILTER_ORDER)
        print(f"  {stem}: {seq.n_frames} frames, {seq.missing_fraction():.2%} missing")
        seqs.append(seq)
    return seqs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--n-trials", type=int, default=6)
    p.add_argument("--subset", type=int, default=60, help="signals to sample for AMI/FNN")
    p.add_argument("--out-dir", default="./embedding_out")
    args = p.parse_args()

    matplotlib.use("Agg")  # save to file; remove this line in a notebook
    import matplotlib.pyplot as plt

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Preprocessing trials...")
    seqs = build_sequences(args.data_dir, args.n_trials, out / "canonical")

    print("\nComputing embedding evidence (per-keypoint speed, pooled across trials)...")
    signals = pool_signals(seqs, magnitude_channels)
    evidence = select_embedding(
        signals,
        tau_grid=(10, 25),
        m_grid=(3, 6),
        ami_max_lag=50,
        fnn_max_dim=8,
        subset=args.subset,
        seed=0,
    )

    print("\n" + "=" * 70)
    print(evidence.justification)
    print("=" * 70)

    ax_ami, _ = plot_embedding_evidence(evidence)
    ax_ami.figure.savefig(out / "embedding_evidence.png", dpi=110, bbox_inches="tight")
    plt.close(ax_ami.figure)

    ax_tau, _ = plot_embedding_variability(evidence, group_by="condition")
    ax_tau.figure.savefig(out / "embedding_variability.png", dpi=110, bbox_inches="tight")
    plt.close(ax_tau.figure)
    print(f"\nSaved plots to {out}/embedding_evidence.png and embedding_variability.png")

    # The human commits after inspecting the plots:
    params = evidence.commit(tau=evidence.proposed_tau, m=evidence.proposed_m,
                             notes="committed from demo proposal")
    print("\nCommitted EmbeddingParams:", params.to_dict())


if __name__ == "__main__":
    main()
