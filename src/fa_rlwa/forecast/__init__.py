"""Forecasting interfaces and reference models."""

from fa_rlwa.forecast.base import Forecaster
from fa_rlwa.forecast.ewma import EWMAForecaster, MovingAverageForecaster

__all__ = ["Forecaster", "EWMAForecaster", "MovingAverageForecaster"]
