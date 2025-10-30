# pose-dynamics

Minimal, flat repo for pose analysis with a clear **General Method**: Pose acquisition → Preprocessing → Dimensionality Reduction → Dynamics (RQA).

This repo is intentionally simple:
- **`rqa/`**: put the xkiwilabs RQA code here (or install `rqa_analysis`).
- **`preprocessing/`**: gap handling, normalization, detrend, IO helpers.
- **`features/`**: kinematics and PCA.
- **`examples/`**: runnable scripts + notebooks that follow the General Method.
- **`data/`**: your CSVs go here (git-ignored by default; keep small examples).

## Install (conda or venv is fine)
```bash
pip install -r requirements.txt
```

You have *two* ways to get RQA working:
1) **Vendored**: clone xkiwilabs into `./rqa/` so imports like `from rqa.autoRQA import autoRQA` work.
```bash
git clone https://github.com/xkiwilabs/Recurrence-Quantification-Analysis.git rqa
```
2) **Installed**: `pip install rqa_analysis` and our helper will fall back to `from rqa_analysis import autoRQA`.

## Quick run (2D example)
```bash
python examples/run_2d_example.py
```

## Quick run (3D example)
```bash
python examples/run_3d_example.py
```

## Structure
```
pose-dynamics/
├─ rqa/                      
├─ preprocessing/
│  ├─ io_2d_openpose.py
│  ├─ io_3d_csv.py
│  ├─ gaps.py
│  ├─ normalization.py
│  └─ detrend.py
├─ features/
│  ├─ kinematics.py
│  └─ pca_simple.py
├─ examples/
│  ├─ _rqa_helpers.py
│  ├─ run_2d_example.py
│  ├─ run_3d_example.py
│  ├─ 01_face_2d_general_method.ipynb
│  └─ 02_body_3d_general_method.ipynb
├─ data/
│  └─ .gitkeep
├─ requirements.txt
└─ .gitignore
```

## Notes
- The AMI-based `tau` estimator in `gaps.py` is robust to NaNs (uses longest clean run).
- Interpolation policy follows your paper: max gap `L_max = min(0.25 T, 2 τ)`.
- Notebooks mirror the **General Method** and can be expanded into case studies.
