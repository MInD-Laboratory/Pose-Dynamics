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
class FeaturesConfig:
    keypoints: Union[Literal["all"], List[str]] = "all"
    kinematics: KinematicsConfig = field(default_factory=KinematicsConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)

    @staticmethod
    def from_dict(x: Any, ctx: str = "features") -> "FeaturesConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(d, ["keypoints", "kinematics", "geometry"], ctx)
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
