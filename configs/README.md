# Study configs

Copy one of these, edit the values, and run it:

```bash
pose-dynamics run configs/full_analysis.yaml     # or from a notebook: StudyConfig.from_file(...)
pose-dynamics new-config my_study.yaml           # write a fresh template to edit
pose-dynamics inspect trial.csv                  # plot keypoints with index numbers (to fill ROI lists)
```

- **`per_keypoint.yaml`** — simplest; analyses each keypoint's speed. Works on any file.
- **`features_only.yaml`** — extract a feature pipeline's outputs to `features/`, no analysis.
- **`full_analysis.yaml`** — features + linear + recurrence metrics per window.

Every field is documented in [`docs/configuration.md`](../docs/configuration.md);
the feature primitives in [`docs/feature_steps.md`](../docs/feature_steps.md).
