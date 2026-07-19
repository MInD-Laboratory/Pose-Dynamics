# The canonical CSV format

pose-dynamics supports **one** input format. Converting your estimator's output to
it is the only barrier to entry, so this page is exact. Example converters (OpenPose,
ZED) are in `examples/` — copy and adapt one.

## The rule

**One CSV per person per trial. One row per frame. One group of columns per
keypoint.** Column names are an **axis letter followed by a 0-based keypoint
index**, with no other columns:

```
x0,y0,x1,y1, ...              # 2D, no confidence
x0,y0,z0,x1,y1,z1, ...        # 3D, no confidence
x0,y0,c0,x1,y1,c1, ...        # 2D with confidence
x0,y0,z0,c0,x1,y1,z1,c1, ...  # 3D with confidence
```

| Letter | Meaning | Required |
|--------|---------|----------|
| `x`, `y` | horizontal, vertical coordinate | **yes** |
| `z` | depth (3D only) | optional |
| `c` | per-keypoint confidence (0–1) | optional |

- **Dimensionality (2D vs 3D) is inferred from the header:** if every keypoint has
  a `z`, the file is 3D; otherwise 2D. You do not configure it.
- **Confidence is inferred from the header:** if every keypoint has a `c`, the file
  has confidence; otherwise it does not.
- Keypoint indices must start at `0` and be **contiguous** (`0, 1, …, K-1`).
- Column *order* within a file may vary (`y0,x0,x1,y1` is fine); the names are
  authoritative.
- Missing values are blank cells (become `NaN`); text is not allowed in coordinate
  or confidence columns.
- Units for `x`/`y`/`z` are whatever your estimator uses (pixels, metres). You
  normalize later in the pipeline.

## Frame rate is required — and configured, not inferred

The file carries no timestamps, so **you must supply the frame rate** when loading:

```python
from pose_dynamics import load_pose_csv
seq = load_pose_csv("trial01.csv", frame_rate=60.0)   # Hz — required
```

Frame rate drives the filter cutoff, the interpolation cap, and the dyad
shared-clock check. If your recording has a *variable* frame rate, resample it to a
uniform grid first (the Case 3 converter shows how, using timestamps).

## A worked example

A minimal 2-keypoint, 2D file with confidence, three frames — the second frame of
keypoint 1 is missing:

```csv
x0,y0,c0,x1,y1,c1
100.0,200.0,0.95,140.0,180.0,0.90
101.2,199.4,0.93,,,0.10
102.0,198.9,0.94,141.1,181.3,0.88
```

Loaded, this is a `PoseSequence` of shape `(3 frames, 2 keypoints, 2 dims)` with a
confidence array; the blank cells for keypoint 1 in frame 2 are marked missing in
the validity mask.

Try the bundled fixture to see a full-size example:

```python
from pose_dynamics.data import example_fixture
seq = load_pose_csv(example_fixture(), frame_rate=60.0)
```

## What a malformed file tells you

Loading validates the header and body and fails with a message you can act on:

| If your file… | You get (a `SchemaError`) | Fix |
|---------------|---------------------------|-----|
| has extra columns (timestamps, labels, `prob0`, …) | *"These column names are not in the expected format: […]. Every column must be an axis letter followed by a keypoint number, e.g. 'x0', 'y0', 'z0' (3D), or 'c0' (confidence)."* | Rename/remove non-coordinate columns; the canonical file holds pose coordinates only. |
| skips or misnumbers a keypoint | *"Keypoint numbers must start at 0 and be consecutive with no gaps (0 to K-1)…"* | Renumber keypoints contiguously from 0. |
| mixes 2D and 3D keypoints | *"…All keypoints in a file must share the same dimensionality — a file is either fully 2D or fully 3D."* | Give every keypoint the same axes. |
| has `c` on some keypoints but not others | *"Confidence columns must be present for either all keypoints or none…"* | Add the missing `c` columns, or remove all of them. |
| is missing an `x` or `y` | *"Keypoint 0 must have at least 'x0' and 'y0' columns (and optionally 'z0' for 3D)…"* | Every keypoint needs `x` and `y`. |
| repeats a column | *"Duplicate column 'x0': keypoint 0 already has an 'x' column…"* | Remove the duplicate. |
| has text in a coordinate cell | *"Column 'y0' … contains non-numeric values (first offending row indices: […]). … blanks are allowed for missing data but text is not."* | Replace text with a number or a blank. |
| has a header but no rows | *"File has a header but no data rows…"* | Provide at least one frame. |
| does not exist | *"File not found: … Check the path in your config…"* | Fix the path. |

Every error names the offending columns or rows, so a non-programmer can act on it
without reading the traceback.
