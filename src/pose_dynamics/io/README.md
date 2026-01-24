# IO / Ingest (Pose CSV)

**Purpose:**
Convert cleaned pose CSV files into the canonical long-form format used by the rest of the pipeline.

---

## Assumptions

* **One CSV = one trial/session**
* **One pose stream per CSV** (no multi-person tracking)
* Data is already cleaned upstream

---

## Supported input format

### Time axis (required)

Each CSV must have **one** of:

* `time` (float seconds), or
* `frame` (int) **and** `--fps` passed to the CLI

If both exist, `time` is used.

---

### Pose columns

Keypoints are encoded in **wide format** using column suffixes:

* Required per keypoint:

  * `x_<kp>`
  * `y_<kp>`
* Optional:

  * `z_<kp>`
  * `prob_<kp>` or `conf_<kp>`

Examples:

```
x_nose, y_nose, prob_nose
x_knee, y_knee
x_1, y_1
```

A keypoint is valid only if both `x_<kp>` and `y_<kp>` exist.

---

### Extra columns

Any column not matching the above rules is:

* ignored
* recorded in `recording.json`

---

## Output

Running ingest produces:

* `pose.parquet`
  Canonical long-form pose data (all trials concatenated)

* `recording.json`
  Run + per-trial metadata (timing mode, keypoints, ignored columns)

* `qc_ingest.json`
  Per-trial QC summary (missing data, ranges, flags)

---

## Canonical columns (internal)

Depending on input:

* `trial_id`
* `source_file`
* `time` **or** `frame`
* `keypoint`
* `x`, `y`
* optional: `z`
* optional: `conf`

---

## CLI

```bash
pose-dynamics ingest-csv \
  --in data/pose_csv \
  --out artifacts/ingest/run_001 \
  --fps 30
```

`--fps` is required only if `time` is not present.

---