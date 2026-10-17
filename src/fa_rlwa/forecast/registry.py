"""Load a persisted forecaster from its model metadata."""

from __future__ import annotations

import json
from pathlib import Path

from fa_rlwa.forecast.base import Forecaster


def load_forecaster(path: str | Path) -> Forecaster:
    """Resolve a saved model directory to the registered forecaster class."""

    directory = Path(path)
    metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    name = metadata.get("model")
    if name == "ewma":
        from fa_rlwa.forecast.ewma import EWMAForecaster

        return EWMAForecaster.load(directory)
    if name == "moving_average":
        from fa_rlwa.forecast.ewma import MovingAverageForecaster

        return MovingAverageForecaster.load(directory)
    if name == "arima":
        from fa_rlwa.forecast.arima import ARIMAForecaster

        return ARIMAForecaster.load(directory)
    if name == "random_forest":
        from fa_rlwa.forecast.trees import RandomForestForecaster

        return RandomForestForecaster.load(directory)
    if name == "xgboost":
        from fa_rlwa.forecast.trees import XGBoostForecaster

        return XGBoostForecaster.load(directory)
    if name == "lstm":
        from fa_rlwa.forecast.lstm import LSTMForecaster

        return LSTMForecaster.load(directory)
    raise ValueError(f"Unknown forecaster type {name!r} in {directory / 'model.json'}")
