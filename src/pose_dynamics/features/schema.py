"""Feature extraction config schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Sequence, Union

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


class ConfigError(ValueError):
    """Raised when features.yml is invalid or inconsistent."""


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


def _parse_keypoint_ref(value: Any, ctx: str) -> Union[str, List[str], None]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    raise ConfigError(f"{ctx} must be a string keypoint name or list[str].")


@dataclass(frozen=True)
class KinematicsConfig:
    enabled: bool = True
    metrics: List[Literal["speed", "accel"]] = field(
        default_factory=lambda: ["speed", "accel"]
    )

    @staticmethod
    def from_dict(x: Any, ctx: str = "kinematics") -> "KinematicsConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(d, ["enabled", "metrics"], ctx)
        metrics = _get(d, "metrics", ["speed", "accel"])
        if not (
            isinstance(metrics, list) and all(m in ("speed", "accel") for m in metrics)
        ):
            raise ConfigError(f"{ctx}.metrics must be list of ['speed','accel'].")
        return KinematicsConfig(enabled=bool(_get(d, "enabled", True)), metrics=metrics)


@dataclass(frozen=True)
class GeometryConfig:
    enabled: bool = False
    pairwise_distances: bool = True

    @staticmethod
    def from_dict(x: Any, ctx: str = "geometry") -> "GeometryConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(d, ["enabled", "pairwise_distances"], ctx)
        return GeometryConfig(
            enabled=bool(_get(d, "enabled", False)),
            pairwise_distances=bool(_get(d, "pairwise_distances", True)),
        )


@dataclass(frozen=True)
class BlinkConfig:
    enabled: bool = True
    left_upper: Union[str, List[str], None] = None
    left_lower: Union[str, List[str], None] = None
    right_upper: Union[str, List[str], None] = None
    right_lower: Union[str, List[str], None] = None

    @staticmethod
    def from_dict(x: Any, ctx: str = "blink") -> "BlinkConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(
            d,
            ["enabled", "left_upper", "left_lower", "right_upper", "right_lower"],
            ctx,
        )
        return BlinkConfig(
            enabled=bool(_get(d, "enabled", True)),
            left_upper=_parse_keypoint_ref(_get(d, "left_upper"), f"{ctx}.left_upper"),
            left_lower=_parse_keypoint_ref(_get(d, "left_lower"), f"{ctx}.left_lower"),
            right_upper=_parse_keypoint_ref(
                _get(d, "right_upper"), f"{ctx}.right_upper"
            ),
            right_lower=_parse_keypoint_ref(
                _get(d, "right_lower"), f"{ctx}.right_lower"
            ),
        )


@dataclass(frozen=True)
class MouthConfig:
    enabled: bool = True
    upper: str | None = None
    lower: str | None = None

    @staticmethod
    def from_dict(x: Any, ctx: str = "mouth") -> "MouthConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(d, ["enabled", "upper", "lower"], ctx)
        return MouthConfig(
            enabled=bool(_get(d, "enabled", True)),
            upper=_get(d, "upper"),
            lower=_get(d, "lower"),
        )


@dataclass(frozen=True)
class PupilConfig:
    enabled: bool = True
    left_pupil: str | None = None
    right_pupil: str | None = None
    left_eye_contour: List[str] = field(default_factory=list)
    right_eye_contour: List[str] = field(default_factory=list)

    @staticmethod
    def from_dict(x: Any, ctx: str = "pupil") -> "PupilConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(
            d,
            [
                "enabled",
                "left_pupil",
                "right_pupil",
                "left_eye_contour",
                "right_eye_contour",
            ],
            ctx,
        )
        left_eye = _get(d, "left_eye_contour", [])
        right_eye = _get(d, "right_eye_contour", [])
        if not (
            isinstance(left_eye, list) and all(isinstance(k, str) for k in left_eye)
        ):
            raise ConfigError(f"{ctx}.left_eye_contour must be list[str].")
        if not (
            isinstance(right_eye, list) and all(isinstance(k, str) for k in right_eye)
        ):
            raise ConfigError(f"{ctx}.right_eye_contour must be list[str].")
        return PupilConfig(
            enabled=bool(_get(d, "enabled", True)),
            left_pupil=_get(d, "left_pupil"),
            right_pupil=_get(d, "right_pupil"),
            left_eye_contour=left_eye,
            right_eye_contour=right_eye,
        )


@dataclass(frozen=True)
class FacialConfig:
    enabled: bool = False
    blink: BlinkConfig = field(default_factory=BlinkConfig)
    mouth: MouthConfig = field(default_factory=MouthConfig)
    pupil: PupilConfig = field(default_factory=PupilConfig)
    center_face: List[str] = field(default_factory=list)
    scale_by_interocular: bool = False
    stats: List[str] = field(
        default_factory=lambda: [
            "mean",
            "std",
            "min",
            "max",
            "median",
            "iqr",
            "rms",
            "skew",
            "kurtosis",
        ]
    )
    derivatives: List[Literal["velocity", "acceleration"]] = field(
        default_factory=lambda: ["velocity", "acceleration"]
    )

    @staticmethod
    def from_dict(x: Any, ctx: str = "facial") -> "FacialConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(
            d,
            [
                "enabled",
                "blink",
                "mouth",
                "pupil",
                "center_face",
                "scale_by_interocular",
                "stats",
                "derivatives",
            ],
            ctx,
        )
        stats = _get(
            d,
            "stats",
            [
                "mean",
                "std",
                "min",
                "max",
                "median",
                "iqr",
                "rms",
                "skew",
                "kurtosis",
            ],
        )
        if not (isinstance(stats, list) and all(isinstance(s, str) for s in stats)):
            raise ConfigError(f"{ctx}.stats must be list[str].")
        derivatives = _get(d, "derivatives", ["velocity", "acceleration"])
        if not (
            isinstance(derivatives, list)
            and all(dv in ("velocity", "acceleration") for dv in derivatives)
        ):
            raise ConfigError(
                f"{ctx}.derivatives must be list of ['velocity','acceleration']."
            )
        return FacialConfig(
            enabled=bool(_get(d, "enabled", False)),
            blink=BlinkConfig.from_dict(_get(d, "blink", {}), ctx=f"{ctx}.blink"),
            mouth=MouthConfig.from_dict(_get(d, "mouth", {}), ctx=f"{ctx}.mouth"),
            pupil=PupilConfig.from_dict(_get(d, "pupil", {}), ctx=f"{ctx}.pupil"),
            center_face=_get(d, "center_face", []),
            scale_by_interocular=bool(_get(d, "scale_by_interocular", False)),
            stats=stats,
            derivatives=derivatives,
        )


@dataclass(frozen=True)
class HeadMotionConfig:
    enabled: bool = False
    stats: List[str] = field(
        default_factory=lambda: [
            "mean",
            "std",
            "min",
            "max",
            "median",
            "iqr",
            "rms",
            "skew",
            "kurtosis",
        ]
    )
    derivatives: List[Literal["velocity", "acceleration"]] = field(
        default_factory=lambda: ["velocity", "acceleration"]
    )

    @staticmethod
    def from_dict(x: Any, ctx: str = "head_motion") -> "HeadMotionConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(d, ["enabled", "stats", "derivatives"], ctx)
        stats = _get(
            d,
            "stats",
            [
                "mean",
                "std",
                "min",
                "max",
                "median",
                "iqr",
                "rms",
                "skew",
                "kurtosis",
            ],
        )
        if not (isinstance(stats, list) and all(isinstance(s, str) for s in stats)):
            raise ConfigError(f"{ctx}.stats must be list[str].")
        derivatives = _get(d, "derivatives", ["velocity", "acceleration"])
        if not (
            isinstance(derivatives, list)
            and all(dv in ("velocity", "acceleration") for dv in derivatives)
        ):
            raise ConfigError(
                f"{ctx}.derivatives must be list of ['velocity','acceleration']."
            )
        return HeadMotionConfig(
            enabled=bool(_get(d, "enabled", False)),
            stats=stats,
            derivatives=derivatives,
        )

@dataclass(frozen=True)
class ROIRegion:
    """Configuration for a single Region of Interest."""
    name: str
    keypoints: List[str] = field(default_factory=list)

    @staticmethod
    def from_dict(x: Any, ctx: str = "roi.region") -> "ROIRegion":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(d, ["name", "keypoints"], ctx)
        name = _get(d, "name")
        if not isinstance(name, str):
            raise ConfigError(f"{ctx}.name must be a string.")
        keypoints = _get(d, "keypoints", [])
        if not (isinstance(keypoints, list) and all(isinstance(k, str) for k in keypoints)):
            raise ConfigError(f"{ctx}.keypoints must be list[str].")
        return ROIRegion(name=name, keypoints=keypoints)


@dataclass(frozen=True)
class ROIConfig:
    """Configuration for ROI (Region of Interest) feature extraction."""
    enabled: bool = False
    derivatives: List[Literal["velocity", "acceleration"]] = field(
        default_factory=lambda: ["velocity"]
    )
    stats: List[str] = field(
        default_factory=lambda: ["mean", "rms"]
    )
    regions: List[ROIRegion] = field(default_factory=list)

    @staticmethod
    def from_dict(x: Any, ctx: str = "roi") -> "ROIConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(d, ["enabled", "derivatives", "stats", "regions"], ctx)
        
        derivatives = _get(d, "derivatives", ["velocity"])
        if not (
            isinstance(derivatives, list)
            and all(dv in ("velocity", "acceleration") for dv in derivatives)
        ):
            raise ConfigError(
                f"{ctx}.derivatives must be list of ['velocity','acceleration']."
            )
        
        stats = _get(d, "stats", ["mean", "rms"])
        if not (isinstance(stats, list) and all(isinstance(s, str) for s in stats)):
            raise ConfigError(f"{ctx}.stats must be list[str].")
        
        regions_raw = _get(d, "regions", [])
        if not isinstance(regions_raw, list):
            raise ConfigError(f"{ctx}.regions must be a list.")
        
        regions = [
            ROIRegion.from_dict(r, ctx=f"{ctx}.regions[{i}]")
            for i, r in enumerate(regions_raw)
        ]
        
        return ROIConfig(
            enabled=bool(_get(d, "enabled", False)),
            derivatives=derivatives,
            stats=stats,
            regions=regions,
        )

@dataclass(frozen=True)
class FeaturesConfig:
    keypoints: Union[Literal["all"], List[str]] = "all"
    kinematics: KinematicsConfig = field(default_factory=KinematicsConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    facial: FacialConfig = field(default_factory=FacialConfig)
    head_motion: HeadMotionConfig = field(default_factory=HeadMotionConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)  # NEW

    @staticmethod
    def from_dict(x: Any, ctx: str = "features") -> "FeaturesConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(
            d,
            ["keypoints", "kinematics", "geometry", "facial", "head_motion", "roi"],  # Added "roi"
            ctx,
        )
        keypoints = _get(d, "keypoints", "all")
        if keypoints != "all" and not (
            isinstance(keypoints, list) and all(isinstance(k, str) for k in keypoints)
        ):
            raise ConfigError(f"{ctx}.keypoints must be 'all' or a list[str].")
        return FeaturesConfig(
            keypoints=keypoints,
            kinematics=KinematicsConfig.from_dict(
                _get(d, "kinematics", {}), ctx=f"{ctx}.kinematics"
            ),
            geometry=GeometryConfig.from_dict(
                _get(d, "geometry", {}), ctx=f"{ctx}.geometry"
            ),
            facial=FacialConfig.from_dict(_get(d, "facial", {}), ctx=f"{ctx}.facial"),
            head_motion=HeadMotionConfig.from_dict(
                _get(d, "head_motion", {}), ctx=f"{ctx}.head_motion"
            ),
            roi=ROIConfig.from_dict(_get(d, "roi", {}), ctx=f"{ctx}.roi"),  # NEW
        )

    @staticmethod
    def from_yaml(path: str) -> "FeaturesConfig":
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to load features.yml (pip install pyyaml)."
            )
        with open(path, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f) or {}
        if "features" in obj and isinstance(obj["features"], dict):
            obj = obj["features"]
        return FeaturesConfig.from_dict(obj)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
