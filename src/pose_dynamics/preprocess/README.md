```
src/pose_dynamics/preprocess/
  __init__.py
  schema.py                 # your strict YAML schema + validation (already exists)
  api.py                    # public entrypoint: run_preprocess(...)
  pipeline.py               # orchestrates stage order; calls the modules below

  selection.py              # apply SelectionConfig to df (drop keypoints/dims, enforce xyz rules)
  timebase.py               # ensure time; optional resample; dt stats helpers

  missing.py                # gap detection + short-gap interpolation policy (seconds/frames/embedding)
  confidence.py             # conf masking -> NaNs

  windowing.py              # build windows table + compute missingness per window + drop reasons
  qc.py                     # QC aggregation utilities; emits qc_preprocess.json
  provenance.py             # resolved config + hashes + run metadata writer

  alignment/
    __init__.py
    procrustes.py            # Procrustes (trial/global template; equal-weight trials for global)

  normalize.py              # zscore/minmax; global_trial or windowed (windowed needs windows)
  filtering.py              # detrend/lowpass, with guards (timebase regularity / resample)
```