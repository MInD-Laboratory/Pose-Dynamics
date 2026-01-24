"""RQA parameter estimation config schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


class ConfigError(ValueError):
    """Raised when rqa params config is invalid."""


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
class AMIConfig:
    max_lag: int = 60
    bins: int = 32

    @staticmethod
    def from_dict(x: Any, ctx: str = "ami") -> "AMIConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(d, ["max_lag", "bins"], ctx)
        return AMIConfig(
            max_lag=int(_get(d, "max_lag", 60)),
            bins=int(_get(d, "bins", 32)),
        )


@dataclass(frozen=True)
class FNNConfig:
    max_dim: int = 10
    tau: int = 10
    rtol: float = 10.0
    atol: float = 2.0

    @staticmethod
    def from_dict(x: Any, ctx: str = "fnn") -> "FNNConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(d, ["max_dim", "tau", "rtol", "atol"], ctx)
        return FNNConfig(
            max_dim=int(_get(d, "max_dim", 10)),
            tau=int(_get(d, "tau", 10)),
            rtol=float(_get(d, "rtol", 10.0)),
            atol=float(_get(d, "atol", 2.0)),
        )


@dataclass(frozen=True)
class EpsilonConfig:
    m: int = 4
    tau: int = 10
    percentiles: List[int] = (1, 2, 5, 10, 15, 20, 25, 30)

    @staticmethod
    def from_dict(x: Any, ctx: str = "epsilon") -> "EpsilonConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(d, ["m", "tau", "percentiles"], ctx)
        percs = _get(d, "percentiles", [1, 2, 5, 10, 15, 20, 25, 30])
        if not (isinstance(percs, list) and all(isinstance(p, int) for p in percs)):
            raise ConfigError(f"{ctx}.percentiles must be list[int].")
        return EpsilonConfig(
            m=int(_get(d, "m", 4)),
            tau=int(_get(d, "tau", 10)),
            percentiles=percs,
        )


@dataclass(frozen=True)
class RQAParamsConfig:
    keypoints: Union[Literal["all"], List[str]] = "all"
    n_keypoints: Optional[int] = 5
    n_windows: int = 5
    signal: Literal["magnitude"] = "magnitude"
    ami: AMIConfig = AMIConfig()
    fnn: FNNConfig = FNNConfig()
    epsilon: EpsilonConfig = EpsilonConfig()

    @staticmethod
    def from_dict(x: Any, ctx: str = "rqa_params") -> "RQAParamsConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(
            d,
            [
                "keypoints",
                "n_keypoints",
                "n_windows",
                "signal",
                "ami",
                "fnn",
                "epsilon",
            ],
            ctx,
        )
        keypoints = _get(d, "keypoints", "all")
        if keypoints != "all" and not (
            isinstance(keypoints, list) and all(isinstance(k, str) for k in keypoints)
        ):
            raise ConfigError(f"{ctx}.keypoints must be 'all' or list[str].")
        return RQAParamsConfig(
            keypoints=keypoints,
            n_keypoints=None
            if _get(d, "n_keypoints", 5) is None
            else int(_get(d, "n_keypoints", 5)),
            n_windows=int(_get(d, "n_windows", 5)),
            signal=_get(d, "signal", "magnitude"),
            ami=AMIConfig.from_dict(_get(d, "ami", {}), ctx=f"{ctx}.ami"),
            fnn=FNNConfig.from_dict(_get(d, "fnn", {}), ctx=f"{ctx}.fnn"),
            epsilon=EpsilonConfig.from_dict(
                _get(d, "epsilon", {}), ctx=f"{ctx}.epsilon"
            ),
        )

    @staticmethod
    def from_yaml(path: str) -> "RQAParamsConfig":
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to load rqa-params.yml (pip install pyyaml)."
            )
        with open(path, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f) or {}
        if "rqa_params" in obj and isinstance(obj["rqa_params"], dict):
            obj = obj["rqa_params"]
        return RQAParamsConfig.from_dict(obj)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
