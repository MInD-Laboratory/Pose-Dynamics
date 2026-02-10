"""RQA execution config schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Sequence, Union

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


class ConfigError(ValueError):
    """Raised when rqa.yml is invalid or inconsistent."""


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
class EpsilonConfig:
    mode: Literal["absolute", "percentile", "rr_target", "mean_scaled"] = "percentile"
    value: float = 10.0

    @staticmethod
    def from_dict(x: Any, ctx: str = "epsilon") -> "EpsilonConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(d, ["mode", "value"], ctx)
        return EpsilonConfig(
            mode=_get(d, "mode", "percentile"),
            value=float(_get(d, "value", 10.0)),
        )


@dataclass(frozen=True)
class PlotConfig:
    enabled: bool = True
    max_plots: int = 5

    @staticmethod
    def from_dict(x: Any, ctx: str = "plots") -> "PlotConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(d, ["enabled", "max_plots"], ctx)
        return PlotConfig(
            enabled=bool(_get(d, "enabled", True)),
            max_plots=int(_get(d, "max_plots", 5)),
        )


@dataclass(frozen=True)
class RQAConfig:
    keypoints: Union[Literal["all"], List[str]] = "all"
    signal: Literal["coords", "magnitude"] = "coords"
    m: int = 4
    tau: int = 10
    l_min: int = 2
    v_min: int = 2
    theiler: int = 0
    epsilon: EpsilonConfig = EpsilonConfig()
    analysis: Literal["rqa", "crqa"] = "rqa"
    plots: PlotConfig = PlotConfig()

    @staticmethod
    def from_dict(x: Any, ctx: str = "rqa") -> "RQAConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(
            d,
            [
                "keypoints",
                "signal",
                "m",
                "tau",
                "l_min",
                "v_min",
                "epsilon",
                "analysis",
                "plots",
                "theiler",
            ],
            ctx,
        )
        keypoints = _get(d, "keypoints", "all")
        if keypoints != "all" and not (
            isinstance(keypoints, list) and all(isinstance(k, str) for k in keypoints)
        ):
            raise ConfigError(f"{ctx}.keypoints must be 'all' or list[str].")
        return RQAConfig(
            keypoints=keypoints,
            signal=_get(d, "signal", "coords"),
            m=int(_get(d, "m", 4)),
            tau=int(_get(d, "tau", 10)),
            l_min=int(_get(d, "l_min", 2)),
            v_min=int(_get(d, "v_min", 2)),
            theiler=int(_get(d, "theiler", 0)),
            epsilon=EpsilonConfig.from_dict(
                _get(d, "epsilon", {}), ctx=f"{ctx}.epsilon"
            ),
            analysis=_get(d, "analysis", "rqa"),
            plots=PlotConfig.from_dict(_get(d, "plots", {}), ctx=f"{ctx}.plots"),
        )

    @staticmethod
    def from_yaml(path: str) -> "RQAConfig":
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to load rqa.yml (pip install pyyaml)."
            )
        with open(path, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f) or {}
        if "rqa" in obj and isinstance(obj["rqa"], dict):
            obj = obj["rqa"]
        return RQAConfig.from_dict(obj)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
