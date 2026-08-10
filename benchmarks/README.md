# Computational benchmarks

Reference timings for the `pose-dynamics` pipeline, reported to give a concrete
sense of what the analyses cost and where the limits are. Everything here is
reproducible with [`benchmark.py`](benchmark.py):

```bash
python benchmarks/benchmark.py                                  # microbenchmarks only
python benchmarks/benchmark.py --matb DIR                       # + Case 1
python benchmarks/benchmark.py --mosaic DIR --mosaic-trials 4   # + Case 2
python benchmarks/benchmark.py --mg DIR                         # + Case 3
```

## Machines

Each case study was timed on the machine holding its recordings. All runs are
single-threaded.

- **Machine A** — Apple M4 Pro, 24 GB RAM, macOS 26.5, Python 3.12.
  Cases 1 and 3, and every microbenchmark below.
- **Machine B** — AMD Ryzen AI Max+ 395 (16 cores), 64 GB RAM,
  Windows 11 Pro 26200, Python 3.13. Case 2.

A mid-range laptop should be read as roughly 2–3× slower than either; the scaling
behaviour is unchanged.

## What dominates cost

Recurrence is computed over all pairs of time points, so cost is **O(N²·d)** —
quadratic in the number of frames `N`, and only linear in the dimensionality `d`
of the signal. Recording duration, not feature count, is the binding
computational constraint.

## End-to-end, per trial

| Case | Machine | Signal | Frames / participant | Load | Preprocess | Features + RQA | Per trial | Full dataset |
|---|---|---|---|---|---|---|---|---|
| 1 — MATB (2D facial) | A | 1 participant, windowed | 28,835 | 0.25 s | 0.52 s | 10.66 s | **11.4 s** | ≈ 41 min (216 trials) |
| 2 — MOSAIC (2D body+face) | B | dyad, 3 ROIs, windowed | 11,700 | 1.12 s | 1.06 s | 0.79 s | **2.99 s** | ≈ 14 min (272 dyad-trials) + ≈ 10 min template |
| 3 — Mirror Game (3D full body) | A | dyad, 5 keypoints | 1,058 | 0.02 s | 0.01 s | 0.20 s | **0.23 s** | ≈ 50 s (210 dyad-trials) |

Cases 2 and 3 are dyadic: both partners are loaded and processed per trial, so
their Load and Preprocess figures cover two files, and both partners are trimmed
to a shared frame count so their window boundaries line up in time.

### Case 1 — MATB

Trials run ~16 minutes and every window is analysed separately, so the quadratic
term sets the cost: features + RQA is 94% of the per-trial total.

### Case 2 — MOSAIC

Windowing caps `N` at a fixed 1,800 frames per window while the raw files stay
large (60 Hz, 411 columns, 70–113 MB each), so data handling takes 73% of the
per-trial total and features + CRQA 27%. The global Procrustes template adds a
one-off pass over all 550 files.

### Case 3 — Mirror Game

Trials are ~35 s, so `N` stays small and the whole pipeline runs in a fraction of
a second; features + RQA is 87% of the per-trial total.

## Scaling with recording length and dimensionality

MdRQA (multivariate cross-recurrence, fixed radius), wall-clock seconds:

| N (frames) | ≈ min @ 30 Hz | d = 5 | d = 10 | d = 20 |
|---|---|---|---|---|
| 1,000 | 0.6 | 0.004 | 0.006 | 0.009 |
| 2,000 | 1.1 | 0.016 | 0.021 | 0.035 |
| 4,000 | 2.2 | 0.066 | 0.088 | 0.138 |
| 8,000 | 4.4 | 0.453 | 0.518 | 0.725 |
| 16,000 | 8.9 | 2.82 | 2.60 | 3.41 |

Dimensionality is cheap. At fixed `N = 2,000`, going from `d = 2` to `d = 100`
(a 400-dimensional embedded space at `m = 4`) costs 0.016 s → 0.207 s.

Memory is the real ceiling — the recurrence matrix is N×N:

| N | ≈ min @ 30 Hz | float64 matrix |
|---|---|---|
| 2,000 | 1 | 0.03 GB |
| 10,000 | 6 | 0.80 GB |
| 20,000 | 11 | 3.20 GB |
| 50,000 | 28 | 20.0 GB |

**Rule of thumb:** on a 16–32 GB machine, roughly 10–20 minutes of continuous
30 Hz recording per recurrence matrix. Beyond that, segment into epochs — no
amount of dimensionality reduction helps with the quadratic term.

## The other limit on MdRQA is not speed

Adding dimensions is computationally cheap but statistically costly: pairwise
distances concentrate around their mean, so the recurrence threshold starts
determining the result more than the dynamics do. Relative contrast
`(max − min) / mean` of pairwise distances, 2,000 standardised points:

| d | embedded dim (m = 4) | contrast | CV |
|---|---|---|---|
| 2 | 8 | 4.25 | 0.520 |
| 5 | 20 | 2.75 | 0.321 |
| 10 | 40 | 1.99 | 0.225 |
| 20 | 80 | 1.55 | 0.159 |
| 50 | 200 | 0.97 | 0.101 |
| 100 | 400 | 0.68 | 0.070 |

This is why the practical ceiling for MdRQA sits at roughly a dozen simultaneous
signals — an interpretive limit, not a runtime one. Prefer an anatomically
motivated keypoint subset (as Case 3 uses: 5 of 38 keypoints) over projecting
onto principal components, which distorts the Euclidean geometry that recurrence
thresholding depends on.

## Threshold mode

Fixed-%REC costs about **11×** fixed-radius, because the radius is found by
bisection (cross-RQA, `N = 1,000`: 3.9 ms fixed-radius vs 44.8 ms fixed-%REC).
Worth knowing when sweeping parameters over a whole dataset.
