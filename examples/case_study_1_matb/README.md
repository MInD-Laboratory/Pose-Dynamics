# MATB Case Study

End-to-end workflow to reproduce the MATB facial dynamics analyses (ingest → preprocess → features → RQA) and generate figures/stats from the notebook.

## Quickstart
- From this folder run: `python run.py ingest preprocess features rqa --force-ingest --force-preprocess --force-features --force-rqa`
- Outputs land under `artifacts/`: ingest (`artifacts/ingest/matb`), preprocess (`artifacts/preprocess/matb`), features (`artifacts/features/matb`), RQA (`artifacts/rqa/matb`).
- RQA config: see `configs/rqa.yaml` (minmax + rescale, expanded RQA/CRQA signals).

## Notebook (tutorial.ipynb)
- Requires you run the steps in main.py before running.
- Linear stats cell runs `build_table_with_emmeans` on the wide feature table (drops degenerate metrics, safe plotting). RQA stats cell does the same on `rqa_stats.csv` outputs.
- AMI/FNN helpers and plotting live in `notebook_utils` as well.

## Stats prerequisites
- Mixed models require R + `rpy2`. Install R (e.g., `brew install r`) then `pip install rpy2` in your venv. If `rpy2` is missing, stats cells will raise.
- Degenerate metrics (all-NA, zero variance, single value) are auto-dropped to avoid PD errors; plots skip NaN/Inf means.
