# Feature steps

A **feature pipeline** turns a pose sequence into the signals you analyse. It is a
**list of steps**, applied in order — you declare it in a config (or in Python), and
each step either transforms the pose or produces feature signals. This is how you
get apertures, ROI movement, kinematics, or aligned coordinates without writing code.

In a config, each step is one list item:

```yaml
features:
  - step: coordinate_normalization
    params: {width: 720, height: 720}
  - step: roi_centroid
    params: {rois: {arms: [2, 3, 4, 5, 6, 7]}}
  - step: velocity_magnitude
    params: {method: diff}
```

> **Which index is which body part?** Steps that take keypoint indices (ROIs,
> distances, offsets) need your skeleton's numbering. Run
> `pose-dynamics inspect trial.csv` (or `seq.plot_keypoints()` in a notebook) to plot
> the keypoints labelled with their index numbers, then read the indices off the plot.

## Two kinds of data flow through the pipeline

- **pose** — the keypoint coordinates (a `PoseSequence`).
- **signals** — named 1-D feature time series (a `FeatureSet`).

Each step declares what it needs and what it makes. A pipeline starts with a pose; a
step that needs *signals* (like `zscore`) only works after a step that *produces*
them has run. Getting the order wrong is caught when the config is loaded, with a
clear message — not halfway through a long run.

## The steps

| Step | in → out | What it does |
|------|----------|--------------|
| `coordinate_normalization` | pose → pose | Rescale to a resolution-independent range (`unit` [0,1] or `centered` [−1,1]). |
| `center` | pose → pose | Centre each frame on a keypoint, `"centroid"`, or a set (e.g. Torso). |
| `canonicalise` | pose → pose | 3-D body-frame rotation from pelvis/shoulders/neck. |
| `procrustes` | pose → **geometry and/or parameters** | Align to a template. See the dual output below. |
| `select_keypoints` | pose → pose | Keep a subset of keypoints. |
| `roi_centroid` | pose → pose | Reduce named regions (ROIs) to their centre points. |
| `distance_feature` | pose → signals | Aperture / distance between two keypoint groups. |
| `offset_feature` | pose → signals | Offset of point(s) from a centre, averaged (e.g. pupil). |
| `coordinate_magnitude` | pose → signals | Each keypoint's distance from the origin. |
| `velocity_magnitude` | pose → signals | Each keypoint's speed (movement intensity). |
| `keypoint_coordinates` | pose → signals | Flatten keypoints to multi-dim signals (for multivariate RQA). |
| `zscore` | signals → signals | Standardise each feature. |
| `kinematic_derivatives` | signals → signals | Expand each feature into position / velocity / acceleration. |

List them at runtime with `available_steps()`.

## The Procrustes step's dual output

`procrustes` aligns each frame to a template and can emit, via `emit`:

- `emit: geometry` — an aligned pose that later steps consume (the default; Procrustes
  as preprocessing);
- `emit: parameters` — the per-frame transform (translation, rotation, scale) as
  features;
- `emit: both` — both at once. This is what lets a single alignment step serve as
  both "remove head motion" and "measure head motion."

`scale` chooses the family: `none` (rigid), `uniform` (one scalar), or `anisotropic`
(per-axis, 2-D).

## Advanced: consuming both streams (the Case 1 example)

Case 1 aligns the face and uses **both** outputs — the transform parameters become
head-movement features, the aligned coordinates become expression features with head
motion removed:

```python
[
  {"step": "coordinate_normalization",
   "params": {"width": 2560, "height": 1440, "mode": "unit"}},
  {"step": "procrustes",                       # emit BOTH streams
   "params": {"template": template, "landmarks": [29, 30, 36, 45],
              "scale": "anisotropic", "emit": "both", "prefix": "head"}},
  {"step": "offset_feature",                   # feature on the ALIGNED pose
   "params": {"name_out": "pupil_metric",
              "point": [68, 69], "center": [[36,37,38,39,40,41], [42,43,44,45,46,47]]}},
]
```

The result carries `head_motion_mag` (from the parameters) and `pupil_metric_mag` (a
feature on the aligned geometry) side by side.

## Writing your own step

Implement the small interface, register it by name, then reference it in any config:

```python
import numpy as np
from pose_dynamics.features import FeatureStep, register
from pose_dynamics.features.types import StreamType, PipelineContext

@register
class RangeOfMotion(FeatureStep):
    name = "range_of_motion"
    consumes = frozenset({StreamType.POSE})
    produces = frozenset({StreamType.SIGNALS})

    def __init__(self, keypoint: int):
        self.keypoint = int(keypoint)          # validate params here

    def params(self):                          # recorded in provenance
        return {"keypoint": self.keypoint}

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        coords = ctx.pose.coords[:, self.keypoint, :]
        rom = np.linalg.norm(coords - np.nanmean(coords, axis=0), axis=1)
        from pose_dynamics.features.primitives import _emit_features
        return _emit_features(ctx, [f"kp{self.keypoint}_rom"], rom, self.name, self.params())
```

`{"step": "range_of_motion", "params": {"keypoint": 4}}` now works everywhere,
participates in validation via its declared types, and records itself in the
provenance log.
