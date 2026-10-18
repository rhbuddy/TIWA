"""Policy registry used by the CLI and experiment runner."""

from __future__ import annotations

from typing import Any

from fa_rlwa.forecast.base import Forecaster
from fa_rlwa.forecast.ewma import EWMAForecaster
from fa_rlwa.policies.base import ClientDrivenPolicy, Policy
from fa_rlwa.policies.random_policy import RandomPolicy
from fa_rlwa.policies.rl_policy import RLPolicy
from fa_rlwa.policies.tiwa import CorrectedTiwaPolicy, TiwaForecastPolicy, TiwaOriginalPolicy


def make_policy(
    name: str,
    *,
    forecaster: Forecaster | None = None,
    model_path: str | None = None,
    k_max: int = 5,
    use_forecast: bool = True,
    settings: dict[str, Any] | None = None,
) -> Policy:
    """Create a named policy; extensions can be registered without runner changes."""

    options = dict(settings or {})
    normalized = name.lower().replace("-", "_")
    if normalized in {"client_driven", "client", "rss"}:
        return ClientDrivenPolicy()
    if normalized in {"random", "random_policy"}:
        return RandomPolicy(
            **{key: value for key, value in options.items() if key in {"probability", "dwell_s"}}
        )
    if normalized in {"tiwa_original", "tiwa"}:
        return TiwaOriginalPolicy(**options)
    if normalized in {"tiwa_fixed", "corrected_tiwa", "tiwa_corrected"}:
        return CorrectedTiwaPolicy(**options)
    if normalized in {"tiwa_forecast", "tiwa_plus_forecast"}:
        base_name = str(options.pop("base_policy", "tiwa_fixed"))
        base: Policy = (
            TiwaOriginalPolicy() if base_name == "tiwa_original" else CorrectedTiwaPolicy()
        )
        return TiwaForecastPolicy(
            forecaster or EWMAForecaster(), base, float(options.get("beta", 0.5))
        )
    if normalized in {"rl_forecast", "rl_noforecast", "rl"}:
        if model_path is None:
            raise ValueError("A trained checkpoint is required for the rl policy")
        enabled = normalized != "rl_noforecast" and use_forecast
        return RLPolicy(model_path, k_max=k_max, use_forecast=enabled, forecaster=forecaster)
    raise ValueError(
        f"Unknown policy {name!r}; choose client_driven, random, tiwa_original, "
        "tiwa_fixed, tiwa_forecast, rl_forecast, or rl_noforecast"
    )
