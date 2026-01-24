# Case Study 4: Synthetic end‑to‑end pipeline

This case study generates synthetic pose CSVs and runs the full pipeline:
`ingest → preprocess → feature-extract → pca → rqa-params → rqa`.

## Run

From repo root:

```
python examples/case_study_4_synthetic/run.py
```

Outputs are written to:

- `examples/case_study_4_synthetic/data/`
- `artifacts/ingest/run_synth_001/`
- `artifacts/preprocess/run_synth_001/`
- `artifacts/features/run_synth_001/`
- `artifacts/pca/run_synth_001/`
- `artifacts/rqa_params/run_synth_001/`
- `artifacts/rqa/run_synth_001/`
