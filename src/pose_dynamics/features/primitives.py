"""
The feature primitive library.

Each primitive is a small, typed transform registered under a name and referenced
from config. The three case studies are expressible purely as different
compositions of these primitives, with no case-specific branching (build plan §4).

Stream types (see :mod:`.types`):
- POSE  — a PoseSequence (geometry)
- SIGNALS — a FeatureSet (derived 1-D signals)

The alignment primitive (:class:`Procrustes`) is the key abstraction: it may emit
a geometry stream, a parameter stream, or both.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..data.pose_sequence import PoseSequence
from .base import Primitive, register
from .geometry import (
    body_frame_rotation,
    procrustes_anisotropic,
    procrustes_uniform,
)
from .types import FeatureSet, PipelineContext, StreamType

POSE = StreamType.POSE
SIGNALS = StreamType.SIGNALS


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _emit_features(
    ctx: PipelineContext,
    names: list[str],
    values: np.ndarray,
    stage: str,
    params: dict[str, Any],
) -> PipelineContext:
    """Append signal columns to the context's FeatureSet (creating it if needed)."""
    values = np.asarray(values, float)
    if values.ndim == 1:
        values = values[:, None]
    if ctx.features is None:
        fs = FeatureSet(values, list(names), ctx.pose.frame_rate, meta=dict(ctx.pose.meta))
        fs = fs.replaced(list(names), values, stage, params)
    else:
        fs = ctx.features.added(list(names), values, stage, params)
    return PipelineContext(pose=ctx.pose, features=fs)


def _gradient(x: np.ndarray, fps: float, method: str) -> np.ndarray:
    """First derivative keeping length T. 'gradient' = central diff; 'diff' = backward."""
    if method == "diff":
        d = np.diff(x, axis=0, prepend=x[:1])
        return d * fps
    return np.gradient(x, axis=0) * fps


# ----------------------------------------------------------------------
# Geometry primitives (POSE -> POSE)
# ----------------------------------------------------------------------
@register
class CoordinateNormalization(Primitive):
    """Rescale coordinates to a resolution-independent range.

    ``mode="unit"`` (default): divide axis 0 by ``width``, axis 1 by ``height``
    (and axis 2 by ``depth`` if 3-D), mapping to [0, 1] (Cases 1 & 2, decided).
    ``mode="centered"``: translate the frame centre to the origin first, then
    rescale so edges map to [-1, 1].
    """

    name = "coordinate_normalization"
    consumes = frozenset({POSE})
    produces = frozenset({POSE})

    def __init__(self, width: float, height: float, depth: float | None = None, mode: str = "unit"):
        if mode not in ("unit", "centered"):
            raise ValueError("mode must be 'unit' or 'centered'.")
        self.width, self.height, self.depth, self.mode = width, height, depth, mode

    def params(self) -> dict[str, Any]:
        return {"width": self.width, "height": self.height, "depth": self.depth, "mode": self.mode}

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        pose = ctx.pose
        scale = [self.width, self.height] + ([self.depth] if pose.dims == 3 else [])
        scale = np.asarray(scale[: pose.dims], float)
        coords = pose.coords.copy()
        if self.mode == "centered":
            coords = coords - (scale / 2.0)
            coords = coords / (scale / 2.0)
        else:
            coords = coords / scale
        return PipelineContext(
            pose=pose.with_stage("coordinate_normalization", self.params(), coords=coords),
            features=ctx.features,
        )


@register
class Center(Primitive):
    """Centre each frame on a reference: a keypoint index, ``"centroid"``, or a
    set of keypoints averaged (e.g. Torso = mean of shoulders/hips)."""

    name = "center"
    consumes = frozenset({POSE})
    produces = frozenset({POSE})

    def __init__(self, reference: Any = "centroid"):
        self.reference = reference

    def params(self) -> dict[str, Any]:
        return {"reference": self.reference}

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        pose = ctx.pose
        coords = pose.coords
        if self.reference == "centroid":
            ref = np.nanmean(coords, axis=1, keepdims=True)         # (T,1,d)
        elif isinstance(self.reference, (list, tuple)):
            ref = np.nanmean(coords[:, list(self.reference), :], axis=1, keepdims=True)
        else:
            ref = coords[:, int(self.reference), :][:, None, :]     # single keypoint
        return PipelineContext(
            pose=pose.with_stage("center", self.params(), coords=coords - ref),
            features=ctx.features,
        )


@register
class Canonicalise(Primitive):
    """3-D body-frame canonicalisation (Case 3): express the pose in a frame built
    from pelvis, shoulders and neck. One rigid rotation per frame."""

    name = "canonicalise"
    consumes = frozenset({POSE})
    produces = frozenset({POSE})

    def __init__(self, pelvis: int, left_shoulder: int, right_shoulder: int, neck: int):
        self.pelvis, self.left_shoulder = pelvis, left_shoulder
        self.right_shoulder, self.neck = right_shoulder, neck

    def params(self) -> dict[str, Any]:
        return {"pelvis": self.pelvis, "left_shoulder": self.left_shoulder,
                "right_shoulder": self.right_shoulder, "neck": self.neck}

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        pose = ctx.pose
        if pose.dims != 3:
            raise ValueError("canonicalise requires 3-D poses.")
        out = np.empty_like(pose.coords)
        for t in range(pose.n_frames):
            P = pose.coords[t] - pose.coords[t, self.pelvis]
            R = body_frame_rotation(pose.coords[t], self.pelvis, self.left_shoulder,
                                    self.right_shoulder, self.neck)
            out[t] = P @ R
        return PipelineContext(
            pose=pose.with_stage("canonicalise", self.params(), coords=out),
            features=ctx.features,
        )


@register
class Procrustes(Primitive):
    """Procrustes alignment with a dual-output contract.

    Fits each frame to a template (over the ``landmarks`` subset if given) and,
    per ``emit``:

    - ``"geometry"``  -> replaces the pose with the aligned pose;
    - ``"parameters"`` -> emits the per-frame transform parameters as features
      (``tx, ty[, tz], rotation, scale[/scale_x, scale_y], motion_mag``);
    - ``"both"``      -> both simultaneously (the Case 1 worked example).

    ``scale`` selects the transform family: ``"none"`` (rigid), ``"uniform"``
    (single scalar), or ``"anisotropic"`` (per-axis, 2-D; Case 1).
    """

    name = "procrustes"
    consumes = frozenset({POSE})

    def __init__(
        self,
        template: Any = "self_mean",
        landmarks: list[int] | None = None,
        scale: str = "none",
        emit: str = "geometry",
        prefix: str = "head",
    ):
        if scale not in ("none", "uniform", "anisotropic"):
            raise ValueError("scale must be 'none', 'uniform', or 'anisotropic'.")
        if emit not in ("geometry", "parameters", "both"):
            raise ValueError("emit must be 'geometry', 'parameters', or 'both'.")
        self.template = template
        self.landmarks = landmarks
        self.scale = scale
        self.emit = emit
        self.prefix = prefix
        # instance-level produced streams, driven by `emit`
        produced = set()
        if emit in ("geometry", "both"):
            produced.add(POSE)
        if emit in ("parameters", "both"):
            produced.add(SIGNALS)
        self.produces = frozenset(produced)

    def params(self) -> dict[str, Any]:
        tmpl = self.template if isinstance(self.template, str) else "array"
        return {"template": tmpl, "landmarks": self.landmarks, "scale": self.scale,
                "emit": self.emit, "prefix": self.prefix}

    def _fit(self, X, Y):
        if self.scale == "anisotropic":
            return procrustes_anisotropic(X, Y)
        return procrustes_uniform(X, Y, allow_scale=(self.scale == "uniform"))

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        pose = ctx.pose
        coords = pose.coords
        d = pose.dims

        if isinstance(self.template, str) and self.template == "self_mean":
            template = np.nanmean(coords, axis=0)          # (K, d)
        else:
            template = np.asarray(self.template, float)
        fit_idx = np.asarray(self.landmarks, int) if self.landmarks is not None else np.arange(pose.n_keypoints)
        Y = template[fit_idx]

        aligned = np.empty_like(coords)
        tparams = np.full((pose.n_frames, d), np.nan)
        rot = np.full(pose.n_frames, np.nan)
        scales = np.full((pose.n_frames, d), np.nan)
        for t in range(pose.n_frames):
            # Fit on whatever stable landmarks are present this frame (>= 3), like
            # the prototype's availability mask; skip alignment if too few.
            avail = np.all(np.isfinite(coords[t, fit_idx]), axis=1)
            if avail.sum() < 3:
                aligned[t] = coords[t]  # leave unaligned; params stay NaN
                continue
            use = fit_idx[avail]
            try:
                tp = self._fit(coords[t, use], template[use])
                aligned[t] = tp.apply(coords[t])
                tparams[t] = tp.t
                rot[t] = tp.rotation_angle
                scales[t] = tp.scale
            except np.linalg.LinAlgError:
                aligned[t] = coords[t]  # degenerate fit; leave unaligned

        new_pose = pose
        new_features = ctx.features
        if POSE in self.produces:
            new_pose = pose.with_stage("procrustes", self.params(), coords=aligned)

        out = PipelineContext(pose=new_pose, features=new_features)
        if SIGNALS in self.produces:
            names = [f"{self.prefix}_t{ax}" for ax in "xyz"[:d]]
            cols = [tparams[:, i] for i in range(d)]
            names.append(f"{self.prefix}_rotation")
            cols.append(rot)
            if self.scale == "anisotropic":
                names += [f"{self.prefix}_scale_x", f"{self.prefix}_scale_y"]
                cols += [scales[:, 0], scales[:, 1]]
            else:
                names.append(f"{self.prefix}_scale")
                cols.append(scales[:, 0])
            # combined head-motion magnitude: sqrt(sum t^2 + sum (scale-1)^2)
            motion = np.sqrt((tparams ** 2).sum(axis=1) + ((scales - 1.0) ** 2).sum(axis=1))
            names.append(f"{self.prefix}_motion_mag")
            cols.append(motion)
            out = _emit_features(out, names, np.column_stack(cols), "procrustes_params", self.params())
        return out


# ----------------------------------------------------------------------
# Keypoint selection / ROI reduction (POSE -> POSE)
# ----------------------------------------------------------------------
@register
class SelectKeypoints(Primitive):
    """Keep only the given keypoint indices (e.g. Case 3's five-keypoint subset)."""

    name = "select_keypoints"
    consumes = frozenset({POSE})
    produces = frozenset({POSE})

    def __init__(self, indices: list[int], names: list[str] | None = None):
        self.indices = list(indices)
        self.names = names

    def params(self) -> dict[str, Any]:
        return {"indices": self.indices, "names": self.names}

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        pose = ctx.pose
        idx = np.asarray(self.indices, int)
        names = self.names or [pose.keypoint_names[i] for i in idx]
        new = PoseSequence(
            coords=pose.coords[:, idx, :],
            keypoint_names=list(names),
            frame_rate=pose.frame_rate,
            confidence=None if pose.confidence is None else pose.confidence[:, idx],
            mask=pose.mask[:, idx],
            source_file=pose.source_file,
            provenance=pose.provenance.appended("select_keypoints", self.params()),
            meta=dict(pose.meta),
        )
        return PipelineContext(pose=new, features=ctx.features)


@register
class RoiCentroid(Primitive):
    """Replace the pose with one centroid keypoint per named ROI (Case 2).

    ``rois`` maps ROI name -> keypoint indices; the output pose has one keypoint
    (the mean x/y[/z]) per ROI.
    """

    name = "roi_centroid"
    consumes = frozenset({POSE})
    produces = frozenset({POSE})

    def __init__(self, rois: dict[str, list[int]]):
        if not rois:
            raise ValueError("roi_centroid needs at least one ROI.")
        self.rois = {k: list(v) for k, v in rois.items()}

    def params(self) -> dict[str, Any]:
        return {"rois": self.rois}

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        pose = ctx.pose
        names = list(self.rois)
        cent = np.stack(
            [np.nanmean(pose.coords[:, np.asarray(idx, int), :], axis=1) for idx in self.rois.values()],
            axis=1,
        )  # (T, n_roi, d)
        mask = np.stack(
            [np.all(pose.mask[:, np.asarray(idx, int)], axis=1) for idx in self.rois.values()],
            axis=1,
        )
        new = PoseSequence(
            coords=cent, keypoint_names=names, frame_rate=pose.frame_rate,
            mask=mask, source_file=pose.source_file,
            provenance=pose.provenance.appended("roi_centroid", self.params()),
            meta=dict(pose.meta),
        )
        return PipelineContext(pose=new, features=ctx.features)


# ----------------------------------------------------------------------
# Feature primitives (POSE -> SIGNALS)
# ----------------------------------------------------------------------
@register
class DistanceFeature(Primitive):
    """Aperture / inter-landmark distance between two keypoint groups.

    Emits one signal: the distance between the mean of ``group_a`` and the mean of
    ``group_b``. ``metric="euclidean"`` (full norm; e.g. Case 1 mouth/blink) or
    ``"vertical"`` (|Δy| only)."""

    name = "distance_feature"
    consumes = frozenset({POSE})
    produces = frozenset({SIGNALS})

    def __init__(self, name_out: str, group_a: list[int], group_b: list[int], metric: str = "euclidean"):
        if metric not in ("euclidean", "vertical"):
            raise ValueError("metric must be 'euclidean' or 'vertical'.")
        self.name_out, self.group_a, self.group_b, self.metric = name_out, list(group_a), list(group_b), metric

    def params(self) -> dict[str, Any]:
        return {"name_out": self.name_out, "group_a": self.group_a, "group_b": self.group_b, "metric": self.metric}

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        coords = ctx.pose.coords
        a = np.nanmean(coords[:, self.group_a, :], axis=1)   # (T, d)
        b = np.nanmean(coords[:, self.group_b, :], axis=1)
        if self.metric == "vertical":
            dist = np.abs(a[:, 1] - b[:, 1])
        else:
            dist = np.linalg.norm(a - b, axis=1)
        return _emit_features(ctx, [self.name_out], dist, "distance_feature", self.params())


@register
class OffsetFeature(Primitive):
    """Offset of point(s) from centre(s), averaged across pairs.

    For each ``(point, center)`` pair, offset = ``point - mean(center)``. Per-axis
    offsets and magnitude are averaged across pairs (Case 1 pupil displacement:
    left pupil vs left eye ring, right pupil vs right eye ring, averaged): emits
    ``{name}_dx, {name}_dy[, _dz], {name}_mag`` where the magnitude is the mean of
    the per-pair magnitudes.

    ``point`` may be a single index (with ``center`` a list of indices) or a list
    of indices (with ``center`` a list of index-lists, one per point)."""

    name = "offset_feature"
    consumes = frozenset({POSE})
    produces = frozenset({SIGNALS})

    def __init__(self, name_out: str, point, center):
        self.name_out = name_out
        if isinstance(point, (list, tuple)):
            self.points = [int(p) for p in point]
            self.centers = [list(c) for c in center]
            if len(self.points) != len(self.centers):
                raise ValueError("point and center lists must have equal length.")
        else:
            self.points = [int(point)]
            self.centers = [list(center)]

    def params(self) -> dict[str, Any]:
        return {"name_out": self.name_out, "point": self.points, "center": self.centers}

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        coords = ctx.pose.coords
        d = ctx.pose.dims
        offsets = []   # per-pair (T, d)
        mags = []      # per-pair (T,)
        for p, cen in zip(self.points, self.centers):
            c = np.nanmean(coords[:, cen, :], axis=1)
            off = coords[:, p, :] - c
            offsets.append(off)
            mags.append(np.linalg.norm(off, axis=1))
        offset = np.mean(offsets, axis=0)               # (T, d): mean per-axis offset
        mag = np.mean(mags, axis=0)                      # (T,): mean of per-pair magnitudes
        names = [f"{self.name_out}_d{ax}" for ax in "xyz"[:d]] + [f"{self.name_out}_mag"]
        cols = [offset[:, i] for i in range(d)] + [mag]
        return _emit_features(ctx, names, np.column_stack(cols), "offset_feature", self.params())


@register
class CoordinateMagnitude(Primitive):
    """Per-keypoint scalar magnitude of position ``||coords[t,k,:]||`` (Case 3
    per-keypoint CRQA signal). One signal per keypoint."""

    name = "coordinate_magnitude"
    consumes = frozenset({POSE})
    produces = frozenset({SIGNALS})

    def __init__(self, suffix: str = "mag"):
        self.suffix = suffix

    def params(self) -> dict[str, Any]:
        return {"suffix": self.suffix}

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        pose = ctx.pose
        mag = np.linalg.norm(pose.coords, axis=2)             # (T, K)
        names = [f"{n}_{self.suffix}" for n in pose.keypoint_names]
        return _emit_features(ctx, names, mag, "coordinate_magnitude", self.params())


@register
class VelocityMagnitude(Primitive):
    """Per-keypoint speed ``||d/dt coords||`` (Case 2 ROI movement intensity).

    One signal per keypoint, length preserved. ``method="gradient"`` (central) or
    ``"diff"`` (backward, first frame 0, matching the prototype's ``df.diff()``)."""

    name = "velocity_magnitude"
    consumes = frozenset({POSE})
    produces = frozenset({SIGNALS})

    def __init__(self, method: str = "gradient", suffix: str = "speed"):
        if method not in ("gradient", "diff"):
            raise ValueError("method must be 'gradient' or 'diff'.")
        self.method, self.suffix = method, suffix

    def params(self) -> dict[str, Any]:
        return {"method": self.method, "suffix": self.suffix}

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        pose = ctx.pose
        vel = _gradient(pose.coords, pose.frame_rate, self.method)   # (T, K, d)
        speed = np.linalg.norm(vel, axis=2)                          # (T, K)
        names = [f"{n}_{self.suffix}" for n in pose.keypoint_names]
        return _emit_features(ctx, names, speed, "velocity_magnitude", self.params())


@register
class KeypointCoordinates(Primitive):
    """Flatten selected keypoints' coordinates into multi-dimensional signals.

    Emits one column per keypoint-axis (for multivariate cross-RQA, Case 3)."""

    name = "keypoint_coordinates"
    consumes = frozenset({POSE})
    produces = frozenset({SIGNALS})

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        pose = ctx.pose
        T = pose.n_frames
        flat = pose.coords.reshape(T, -1)
        names = [f"{n}_{ax}" for n in pose.keypoint_names for ax in "xyz"[: pose.dims]]
        return _emit_features(ctx, names, flat, "keypoint_coordinates", {})


# ----------------------------------------------------------------------
# Signal primitives (SIGNALS -> SIGNALS)
# ----------------------------------------------------------------------
@register
class ZScore(Primitive):
    """Z-score each feature signal: ``(x - mean) / std``."""

    name = "zscore"
    consumes = frozenset({SIGNALS})
    produces = frozenset({SIGNALS})

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def params(self) -> dict[str, Any]:
        return {"eps": self.eps}

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        fs = ctx.features
        x = fs.values
        z = (x - np.nanmean(x, axis=0)) / (np.nanstd(x, axis=0) + self.eps)
        return PipelineContext(pose=ctx.pose, features=fs.replaced(fs.names, z, "zscore", self.params()))


@register
class KinematicDerivatives(Primitive):
    """Expand each feature into position/velocity/acceleration (finite difference).

    Case 1 expands all features by kinematic derivatives before windowing. Keeps
    length T (same-length derivatives) so features stay stackable. ``orders``
    selects which of position(0)/velocity(1)/acceleration(2) to keep."""

    name = "kinematic_derivatives"
    consumes = frozenset({SIGNALS})
    produces = frozenset({SIGNALS})

    def __init__(self, orders: tuple[int, ...] = (0, 1, 2), method: str = "gradient"):
        if method not in ("gradient", "diff"):
            raise ValueError("method must be 'gradient' or 'diff'.")
        if not set(orders) <= {0, 1, 2}:
            raise ValueError("orders must be a subset of {0, 1, 2}.")
        self.orders, self.method = tuple(orders), method

    def params(self) -> dict[str, Any]:
        return {"orders": list(self.orders), "method": self.method}

    def apply(self, ctx: PipelineContext) -> PipelineContext:
        fs = ctx.features
        fps = fs.frame_rate
        pos = fs.values
        vel = _gradient(pos, fps, self.method)
        acc = _gradient(vel, fps, self.method)
        by_order = {0: (pos, ""), 1: (vel, "_vel"), 2: (acc, "_accel")}

        out_names: list[str] = []
        out_cols: list[np.ndarray] = []
        for order in self.orders:
            arr, sfx = by_order[order]
            out_names += [f"{n}{sfx}" for n in fs.names]
            out_cols.append(arr)
        values = np.hstack(out_cols)
        return PipelineContext(
            pose=ctx.pose,
            features=fs.replaced(out_names, values, "kinematic_derivatives", self.params()),
        )
