"""
pose_dynamics.preprocess.schema

Strict schema + validation for preprocess.yml.

Design goals:
- Declarative config (YAML) drives preprocessing.
- Unknown keys are rejected (prevents silent typos).
- Defaults are conservative: no alignment, no normalization, no filtering, no resampling.
- Options map directly to methods-paper steps:
  - confidence masking
  - short-gap interpolation with limit policies
  - windowing + window-drop policy
  - Procrustes alignment (trial/global templates)
  - normalization
  - detrend/filtering
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


# -------------------------
# Helpers: strict dict parsing
# -------------------------


class ConfigError(ValueError):
    """Raised when preprocess.yml is invalid or inconsistent."""


def _expect_dict(x: Any, ctx: str) -> Dict[str, Any]:
    if x is None:
        return {}
    if not isinstance(x, dict):
        raise ConfigError(f"{ctx} must be a mapping/dict, got {type(x).__name__}.")
    return x


def _reject_unknown_keys(d: Dict[str, Any], allowed: Sequence[str], ctx: str) -> None:
    unknown = sorted(set(d.keys()) - set(allowed))
    if unknown:
        raise ConfigError(
            f"{ctx} contains unknown keys: {unknown}. Allowed keys: {sorted(allowed)}"
        )


def _get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default)


# -------------------------
# Sub-configs
# -------------------------


@dataclass(frozen=True)
class SelectionConfig:
    keypoints: Union[Literal["all"], List[str]] = "all"
    exclude_keypoints: List[str] = field(default_factory=list)
    dims: Literal["xy", "xyz"] = "xy"
    require_xyz: bool = False
    keep_unselected: bool = False

    @staticmethod
    def from_dict(x: Any, ctx: str = "selection") -> "SelectionConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(
            d,
            [
                "keypoints",
                "exclude_keypoints",
                "dims",
                "require_xyz",
                "keep_unselected",
            ],
            ctx,
        )

        keypoints = _get(d, "keypoints", "all")
        if keypoints != "all" and not (
            isinstance(keypoints, list) and all(isinstance(k, str) for k in keypoints)
        ):
            raise ConfigError(f"{ctx}.keypoints must be 'all' or a list[str].")

        exclude = _get(d, "exclude_keypoints", [])
        if not (isinstance(exclude, list) and all(isinstance(k, str) for k in exclude)):
            raise ConfigError(f"{ctx}.exclude_keypoints must be a list[str].")

        dims = _get(d, "dims", "xy")
        if dims not in ("xy", "xyz"):
            raise ConfigError(f"{ctx}.dims must be 'xy' or 'xyz'.")

        return SelectionConfig(
            keypoints=keypoints,
            exclude_keypoints=exclude,
            dims=dims,
            require_xyz=bool(_get(d, "require_xyz", False)),
            keep_unselected=bool(_get(d, "keep_unselected", False)),
        )


@dataclass(frozen=True)
class ResampleConfig:
    enabled: bool = False
    target_hz: float = 30.0
    method: Literal["linear"] = "linear"
    jitter_tol: float = 0.05  # relative dt jitter threshold used for warnings/QC

    @staticmethod
    def from_dict(x: Any, ctx: str = "timebase.resample") -> "ResampleConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(d, ["enabled", "target_hz", "method", "jitter_tol"], ctx)
        return ResampleConfig(
            enabled=bool(_get(d, "enabled", False)),
            target_hz=float(_get(d, "target_hz", 30.0)),
            method=_get(d, "method", "linear"),
            jitter_tol=float(_get(d, "jitter_tol", 0.05)),
        )


@dataclass(frozen=True)
class TimebaseConfig:
    enforce_time: bool = True
    resample: ResampleConfig = field(default_factory=ResampleConfig)

    @staticmethod
    def from_dict(x: Any, ctx: str = "timebase") -> "TimebaseConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(d, ["enforce_time", "resample"], ctx)
        return TimebaseConfig(
            enforce_time=bool(_get(d, "enforce_time", True)),
            resample=ResampleConfig.from_dict(
                _get(d, "resample", {}), ctx=f"{ctx}.resample"
            ),
        )


@dataclass(frozen=True)
class ConfidenceConfig:
    enabled: bool = False
    conf_min: Optional[float] = None  # if enabled and conf_min is None -> error

    @staticmethod
    def from_dict(x: Any, ctx: str = "confidence") -> "ConfidenceConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(d, ["enabled", "conf_min"], ctx)
        enabled = bool(_get(d, "enabled", False))
        conf_min = _get(d, "conf_min", None)
        conf_min_f = None if conf_min is None else float(conf_min)
        if enabled and conf_min_f is None:
            raise ConfigError(f"{ctx}: enabled=true requires conf_min.")
        return ConfidenceConfig(enabled=enabled, conf_min=conf_min_f)


@dataclass(frozen=True)
class EmbeddingLimitConfig:
    m: int = 4
    tau: int = 15
    units: Literal["frames", "seconds"] = "frames"

    @staticmethod
    def from_dict(x: Any, ctx: str) -> "EmbeddingLimitConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(d, ["m", "tau", "units"], ctx)
        return EmbeddingLimitConfig(
            m=int(_get(d, "m", 4)),
            tau=int(_get(d, "tau", 15)),
            units=_get(d, "units", "frames"),
        )


@dataclass(frozen=True)
class InterpLimitConfig:
    """
    Interpolation limit policy.

    type:
      - seconds: max_gap_s required
      - frames: max_gap_frames required
      - embedding: embedding required (m,tau,units)
    """

    type: Literal["seconds", "frames", "embedding"] = "seconds"
    max_gap_s: Optional[float] = 0.25
    max_gap_frames: Optional[int] = None
    embedding: Optional[EmbeddingLimitConfig] = None

    @staticmethod
    def from_dict(
        x: Any, ctx: str = "missing.interpolation.limit"
    ) -> "InterpLimitConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(
            d, ["type", "max_gap_s", "max_gap_frames", "embedding"], ctx
        )

        limit_type = _get(d, "type", "seconds")
        max_gap_s = _get(d, "max_gap_s", 0.25)
        max_gap_frames = _get(d, "max_gap_frames", None)

        emb = None
        if "embedding" in d and d["embedding"] is not None:
            emb = EmbeddingLimitConfig.from_dict(d["embedding"], ctx=f"{ctx}.embedding")

        cfg = InterpLimitConfig(
            type=limit_type,
            max_gap_s=None if max_gap_s is None else float(max_gap_s),
            max_gap_frames=None if max_gap_frames is None else int(max_gap_frames),
            embedding=emb,
        )

        if cfg.type == "seconds" and cfg.max_gap_s is None:
            raise ConfigError(f"{ctx}: type='seconds' requires max_gap_s.")
        if cfg.type == "frames" and cfg.max_gap_frames is None:
            raise ConfigError(f"{ctx}: type='frames' requires max_gap_frames.")
        if cfg.type == "embedding" and cfg.embedding is None:
            raise ConfigError(
                f"{ctx}: type='embedding' requires embedding: {{m,tau,units}}."
            )
        return cfg


@dataclass(frozen=True)
class InterpolationConfig:
    enabled: bool = True
    method: Literal["linear"] = "linear"
    limit: InterpLimitConfig = field(default_factory=InterpLimitConfig)

    @staticmethod
    def from_dict(x: Any, ctx: str = "missing.interpolation") -> "InterpolationConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(d, ["enabled", "method", "limit"], ctx)
        return InterpolationConfig(
            enabled=bool(_get(d, "enabled", True)),
            method=_get(d, "method", "linear"),
            limit=InterpLimitConfig.from_dict(_get(d, "limit", {}), ctx=f"{ctx}.limit"),
        )


@dataclass(frozen=True)
class MissingConfig:
    interpolation: InterpolationConfig = field(default_factory=InterpolationConfig)

    @staticmethod
    def from_dict(x: Any, ctx: str = "missing") -> "MissingConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(d, ["interpolation"], ctx)
        return MissingConfig(
            interpolation=InterpolationConfig.from_dict(
                _get(d, "interpolation", {}), ctx=f"{ctx}.interpolation"
            )
        )


# -------------------------
# NEW: Windowing + drop policy
# -------------------------


@dataclass(frozen=True)
class TrimEdgesConfig:
    start: float = 0.0
    end: float = 0.0

    @staticmethod
    def from_dict(x: Any, ctx: str = "windowing.trim_edges_s") -> "TrimEdgesConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(d, ["start", "end"], ctx)
        return TrimEdgesConfig(
            start=float(_get(d, "start", 0.0)),
            end=float(_get(d, "end", 0.0)),
        )


@dataclass(frozen=True)
class WindowDropConfig:
    enabled: bool = True

    # any_dim_nan -> sample missing if ANY dim (x,y,z) is NaN (conservative)
    # all_dims_nan -> missing only if ALL dims are NaN
    missing_rule: Literal["any_dim_nan", "all_dims_nan"] = "any_dim_nan"

    # Drop thresholds: window is dropped if EITHER is exceeded (if max_nans is not None).
    max_missing_frac: float = 0.4
    max_nans: Optional[int] = None

    # How missingness aggregates over keypoints:
    # aggregate -> compute missingness over all selected keypoints jointly
    # per_keypoint -> compute per keypoint, then apply per_keypoint_policy
    scope: Literal["aggregate", "per_keypoint"] = "aggregate"
    per_keypoint_policy: Literal["any", "all"] = (
        "any"  # only used if scope=per_keypoint
    )

    @staticmethod
    def from_dict(x: Any, ctx: str = "windowing.drop") -> "WindowDropConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(
            d,
            [
                "enabled",
                "missing_rule",
                "max_missing_frac",
                "max_nans",
                "scope",
                "per_keypoint_policy",
            ],
            ctx,
        )
        max_nans = _get(d, "max_nans", None)
        max_nans_i = None if max_nans is None else int(max_nans)
        return WindowDropConfig(
            enabled=bool(_get(d, "enabled", True)),
            missing_rule=_get(d, "missing_rule", "any_dim_nan"),
            max_missing_frac=float(_get(d, "max_missing_frac", 0.4)),
            max_nans=max_nans_i,
            scope=_get(d, "scope", "aggregate"),
            per_keypoint_policy=_get(d, "per_keypoint_policy", "any"),
        )


@dataclass(frozen=True)
class WindowingConfig:
    enabled: bool = True

    # Define windows in seconds (preferred) or frames.
    units: Literal["seconds", "frames"] = "seconds"

    # Seconds-based definition
    length_s: float = 60.0
    step_s: float = 30.0

    # Frames-based definition
    length_frames: int = 1800
    step_frames: int = 900

    # Edge behavior
    include_partial: bool = False

    # Optional trimming before windowing (seconds; applied only if units=seconds or time is available)
    trim_edges_s: TrimEdgesConfig = field(default_factory=TrimEdgesConfig)

    # QC definition: which keypoints/dims determine missingness/drop logic (does not change signals)
    qc_keypoints: Union[Literal["all"], List[str]] = "all"
    qc_dims: Literal["xy", "xyz"] = "xy"

    drop: WindowDropConfig = field(default_factory=WindowDropConfig)

    @staticmethod
    def from_dict(x: Any, ctx: str = "windowing") -> "WindowingConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(
            d,
            [
                "enabled",
                "units",
                "length_s",
                "step_s",
                "length_frames",
                "step_frames",
                "include_partial",
                "trim_edges_s",
                "qc_keypoints",
                "qc_dims",
                "drop",
            ],
            ctx,
        )

        qc_keypoints = _get(d, "qc_keypoints", "all")
        if qc_keypoints != "all" and not (
            isinstance(qc_keypoints, list)
            and all(isinstance(k, str) for k in qc_keypoints)
        ):
            raise ConfigError(f"{ctx}.qc_keypoints must be 'all' or a list[str].")

        return WindowingConfig(
            enabled=bool(_get(d, "enabled", True)),
            units=_get(d, "units", "seconds"),
            length_s=float(_get(d, "length_s", 60.0)),
            step_s=float(_get(d, "step_s", 30.0)),
            length_frames=int(_get(d, "length_frames", 1800)),
            step_frames=int(_get(d, "step_frames", 900)),
            include_partial=bool(_get(d, "include_partial", False)),
            trim_edges_s=TrimEdgesConfig.from_dict(
                _get(d, "trim_edges_s", {}), ctx=f"{ctx}.trim_edges_s"
            ),
            qc_keypoints=qc_keypoints,
            qc_dims=_get(d, "qc_dims", "xy"),
            drop=WindowDropConfig.from_dict(_get(d, "drop", {}), ctx=f"{ctx}.drop"),
        )


@dataclass(frozen=True)
class AlignmentConfig:
    enabled: bool = False
    method: Literal["procrustes"] = "procrustes"

    template_scope: Literal["trial", "global"] = "global"
    transform: Literal["rigid", "similarity"] = "similarity"
    rotation: Optional[bool] = None
    scaling: Optional[bool] = None
    translation: Optional[bool] = None
    keypoints: Union[Literal["all"], List[str]] = "all"
    reflection: bool = False
    template_agg: Literal["mean"] = "mean"  # median later if needed

    min_valid_frac_per_kp: float = 0.6
    min_kps_for_fit: int = 5

    @staticmethod
    def from_dict(x: Any, ctx: str = "alignment") -> "AlignmentConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(
            d,
            [
                "enabled",
                "method",
                "template_scope",
                "transform",
                "rotation",
                "scaling",
                "translation",
                "keypoints",
                "reflection",
                "template_agg",
                "min_valid_frac_per_kp",
                "min_kps_for_fit",
            ],
            ctx,
        )

        keypoints = _get(d, "keypoints", "all")
        if keypoints != "all" and not (
            isinstance(keypoints, list) and all(isinstance(k, str) for k in keypoints)
        ):
            raise ConfigError(f"{ctx}.keypoints must be 'all' or a list[str].")

        return AlignmentConfig(
            enabled=bool(_get(d, "enabled", False)),
            method=_get(d, "method", "procrustes"),
            template_scope=_get(d, "template_scope", "global"),
            transform=_get(d, "transform", "similarity"),
            rotation=_get(d, "rotation", None),
            scaling=_get(d, "scaling", None),
            translation=_get(d, "translation", None),
            keypoints=keypoints,
            reflection=bool(_get(d, "reflection", False)),
            template_agg=_get(d, "template_agg", "mean"),
            min_valid_frac_per_kp=float(_get(d, "min_valid_frac_per_kp", 0.6)),
            min_kps_for_fit=int(_get(d, "min_kps_for_fit", 5)),
        )


@dataclass(frozen=True)
class NormalizationConfig:
    enabled: bool = False
    method: Literal["none", "zscore", "minmax"] = "none"
    scope: Literal["global_trial", "windowed"] = "global_trial"

    @staticmethod
    def from_dict(x: Any, ctx: str = "normalization") -> "NormalizationConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(d, ["enabled", "method", "scope"], ctx)
        enabled = bool(_get(d, "enabled", False))
        method = _get(d, "method", "none")
        if enabled and method == "none":
            raise ConfigError(f"{ctx}: enabled=true requires method != 'none'.")
        return NormalizationConfig(
            enabled=enabled,
            method=method,
            scope=_get(d, "scope", "global_trial"),
        )


@dataclass(frozen=True)
class LowpassConfig:
    enabled: bool = False
    cutoff_hz: float = 6.0
    order: int = 4

    @staticmethod
    def from_dict(x: Any, ctx: str = "detrend_filter.lowpass") -> "LowpassConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(d, ["enabled", "cutoff_hz", "order"], ctx)
        return LowpassConfig(
            enabled=bool(_get(d, "enabled", False)),
            cutoff_hz=float(_get(d, "cutoff_hz", 6.0)),
            order=int(_get(d, "order", 4)),
        )


@dataclass(frozen=True)
class DetrendFilterConfig:
    detrend: Literal["none", "linear", "highpass"] = "none"
    lowpass: LowpassConfig = field(default_factory=LowpassConfig)

    @staticmethod
    def from_dict(x: Any, ctx: str = "detrend_filter") -> "DetrendFilterConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(d, ["detrend", "lowpass"], ctx)
        return DetrendFilterConfig(
            detrend=_get(d, "detrend", "none"),
            lowpass=LowpassConfig.from_dict(
                _get(d, "lowpass", {}), ctx=f"{ctx}.lowpass"
            ),
        )


@dataclass(frozen=True)
class SpatialCenteringConfig:
    method: Literal["none", "mean_keypoints", "anchor_keypoint"] = "none"
    anchor_keypoint: Optional[str] = None

    @staticmethod
    def from_dict(x: Any, ctx: str = "spatial.centering") -> "SpatialCenteringConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(d, ["method", "anchor_keypoint"], ctx)
        method = _get(d, "method", "none")
        anchor = _get(d, "anchor_keypoint", None)
        if method == "anchor_keypoint" and not anchor:
            raise ConfigError(
                f"{ctx}.anchor_keypoint required when method='anchor_keypoint'."
            )
        return SpatialCenteringConfig(method=method, anchor_keypoint=anchor)


@dataclass(frozen=True)
class SpatialScaleConfig:
    method: Literal["none", "unit_range"] = "none"

    @staticmethod
    def from_dict(x: Any, ctx: str = "spatial.scale") -> "SpatialScaleConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(d, ["method"], ctx)
        return SpatialScaleConfig(method=_get(d, "method", "none"))


@dataclass(frozen=True)
class SpatialConfig:
    centering: SpatialCenteringConfig = field(default_factory=SpatialCenteringConfig)
    scale: SpatialScaleConfig = field(default_factory=SpatialScaleConfig)

    @staticmethod
    def from_dict(x: Any, ctx: str = "spatial") -> "SpatialConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(d, ["centering", "scale"], ctx)
        return SpatialConfig(
            centering=SpatialCenteringConfig.from_dict(
                _get(d, "centering", {}), ctx=f"{ctx}.centering"
            ),
            scale=SpatialScaleConfig.from_dict(
                _get(d, "scale", {}), ctx=f"{ctx}.scale"
            ),
        )


# -------------------------
# Top-level config
# -------------------------


@dataclass(frozen=True)
class PreprocessConfig:
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    timebase: TimebaseConfig = field(default_factory=TimebaseConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    missing: MissingConfig = field(default_factory=MissingConfig)
    windowing: WindowingConfig = field(default_factory=WindowingConfig)  # NEW
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    detrend_filter: DetrendFilterConfig = field(default_factory=DetrendFilterConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)

    @staticmethod
    def from_dict(x: Any, ctx: str = "preprocess") -> "PreprocessConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(
            d,
            [
                "selection",
                "timebase",
                "confidence",
                "missing",
                "windowing",
                "alignment",
                "normalization",
                "detrend_filter",
                "spatial",
            ],
            ctx,
        )
        cfg = PreprocessConfig(
            selection=SelectionConfig.from_dict(
                _get(d, "selection", {}), ctx=f"{ctx}.selection"
            ),
            timebase=TimebaseConfig.from_dict(
                _get(d, "timebase", {}), ctx=f"{ctx}.timebase"
            ),
            confidence=ConfidenceConfig.from_dict(
                _get(d, "confidence", {}), ctx=f"{ctx}.confidence"
            ),
            missing=MissingConfig.from_dict(
                _get(d, "missing", {}), ctx=f"{ctx}.missing"
            ),
            windowing=WindowingConfig.from_dict(
                _get(d, "windowing", {}), ctx=f"{ctx}.windowing"
            ),
            alignment=AlignmentConfig.from_dict(
                _get(d, "alignment", {}), ctx=f"{ctx}.alignment"
            ),
            normalization=NormalizationConfig.from_dict(
                _get(d, "normalization", {}), ctx=f"{ctx}.normalization"
            ),
            detrend_filter=DetrendFilterConfig.from_dict(
                _get(d, "detrend_filter", {}), ctx=f"{ctx}.detrend_filter"
            ),
            spatial=SpatialConfig.from_dict(
                _get(d, "spatial", {}), ctx=f"{ctx}.spatial"
            ),
        )
        _validate_cross_constraints(cfg)
        return cfg

    @staticmethod
    def from_yaml(path: str) -> "PreprocessConfig":
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to load preprocess.yml (pip install pyyaml)."
            )
        with open(path, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f) or {}
        # allow either top-level mapping or nested under 'preprocess'
        if "preprocess" in obj and isinstance(obj["preprocess"], dict):
            obj = obj["preprocess"]
        return PreprocessConfig.from_dict(obj)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _validate_cross_constraints(cfg: PreprocessConfig) -> None:
    """
    Cross-field validation that depends on multiple sections.
    Keep this strict and explicit.
    """

    # Window QC dims should not exceed selection dims (otherwise QC is based on dims you removed)
    if cfg.windowing.enabled:
        if cfg.windowing.qc_dims == "xyz" and cfg.selection.dims != "xyz":
            raise ConfigError("windowing.qc_dims='xyz' requires selection.dims='xyz'.")

    # --- Windowing constraints ---
    if cfg.windowing.enabled:
        if cfg.windowing.units == "seconds":
            if cfg.windowing.length_s <= 0:
                raise ConfigError("windowing.length_s must be > 0.")
            if cfg.windowing.step_s <= 0:
                raise ConfigError("windowing.step_s must be > 0.")
            if cfg.windowing.step_s > cfg.windowing.length_s:
                raise ConfigError(
                    "windowing.step_s cannot be greater than windowing.length_s."
                )
            if (
                cfg.windowing.trim_edges_s.start < 0
                or cfg.windowing.trim_edges_s.end < 0
            ):
                raise ConfigError("windowing.trim_edges_s.start/end must be >= 0.")
        elif cfg.windowing.units == "frames":
            if cfg.windowing.length_frames <= 0:
                raise ConfigError("windowing.length_frames must be > 0.")
            if cfg.windowing.step_frames <= 0:
                raise ConfigError("windowing.step_frames must be > 0.")
            if cfg.windowing.step_frames > cfg.windowing.length_frames:
                raise ConfigError(
                    "windowing.step_frames cannot be greater than windowing.length_frames."
                )
            # trim_edges_s is seconds-based; we allow it to exist but should be zero in frames mode.
            if (
                cfg.windowing.trim_edges_s.start != 0.0
                or cfg.windowing.trim_edges_s.end != 0.0
            ):
                raise ConfigError(
                    "windowing.trim_edges_s must be 0 when windowing.units='frames'."
                )

        # Drop policy sanity
        if cfg.windowing.drop.enabled:
            if not (0.0 <= cfg.windowing.drop.max_missing_frac <= 1.0):
                raise ConfigError("windowing.drop.max_missing_frac must be in [0,1].")
            if (
                cfg.windowing.drop.max_nans is not None
                and cfg.windowing.drop.max_nans < 0
            ):
                raise ConfigError("windowing.drop.max_nans must be >= 0 or null.")
            if (
                cfg.windowing.drop.scope == "aggregate"
                and cfg.windowing.drop.per_keypoint_policy not in ("any", "all")
            ):
                # not used, but keep tight
                raise ConfigError(
                    "windowing.drop.per_keypoint_policy must be 'any' or 'all'."
                )

    # --- Normalization constraints ---
    if cfg.normalization.enabled and cfg.normalization.scope == "windowed":
        # Windowed normalization requires windows to exist.
        if not cfg.windowing.enabled:
            raise ConfigError(
                "normalization.scope='windowed' requires windowing.enabled=true."
            )

    # --- Filtering constraints ---
    # Lowpass generally assumes regular sampling; resampling might be needed.
    # We don't hard-require resample here because some datasets are already regular.
    # Runtime should still check dt jitter and error if not acceptable.
    if cfg.detrend_filter.lowpass.enabled and not cfg.timebase.enforce_time:
        raise ConfigError("lowpass filtering requires timebase.enforce_time=true.")

    # --- Alignment constraints ---
    if cfg.alignment.enabled and cfg.alignment.method != "procrustes":
        raise ConfigError("alignment.method currently supports only 'procrustes'.")
