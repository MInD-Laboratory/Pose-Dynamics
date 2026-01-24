"""PCA config schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional, Sequence

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


class ConfigError(ValueError):
    """Raised when pca.yml is invalid or inconsistent."""


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
class PCAConfig:
    include_pose_summary: bool = True
    include_features: bool = True
    standardize: bool = True
    n_components: Optional[int] = None
    variance_threshold: Optional[float] = None
    scope: Literal["global", "per_trial"] = "global"

    @staticmethod
    def from_dict(x: Any, ctx: str = "pca") -> "PCAConfig":
        d = _expect_dict(x, ctx)
        _reject_unknown_keys(
            d,
            [
                "include_pose_summary",
                "include_features",
                "standardize",
                "n_components",
                "variance_threshold",
                "scope",
            ],
            ctx,
        )
        n_comp = _get(d, "n_components", None)
        n_comp_i = None if n_comp is None else int(n_comp)
        var_thr = _get(d, "variance_threshold", None)
        var_thr_f = None if var_thr is None else float(var_thr)
        if var_thr_f is not None and not (0.0 < var_thr_f <= 1.0):
            raise ConfigError(f"{ctx}.variance_threshold must be in (0, 1].")
        return PCAConfig(
            include_pose_summary=bool(_get(d, "include_pose_summary", True)),
            include_features=bool(_get(d, "include_features", True)),
            standardize=bool(_get(d, "standardize", True)),
            n_components=n_comp_i,
            variance_threshold=var_thr_f,
            scope=_get(d, "scope", "global"),
        )

    @staticmethod
    def from_yaml(path: str) -> "PCAConfig":
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to load pca.yml (pip install pyyaml)."
            )
        with open(path, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f) or {}
        if "pca" in obj and isinstance(obj["pca"], dict):
            obj = obj["pca"]
        return PCAConfig.from_dict(obj)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
