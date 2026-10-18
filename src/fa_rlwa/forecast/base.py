"""Forecast-model interface and common input/output conventions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

# Per-AP feature order: RSSI dBm, loss fraction, throughput Mbps, occupancy, clients, demand Mbps.
FEATURE_COUNT = 6
TARGET_FEATURES = (0, 1, 2, 3)
DEFAULT_HORIZONS = (5, 10, 30)


class Forecaster(ABC):
    """Base class for AP-indicator forecasting models."""

    horizons: tuple[int, ...] = DEFAULT_HORIZONS

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> Forecaster:
        """Fit on windows ``x[N, 30, 6]`` and targets ``y[N, 3, 4]``."""

    @abstractmethod
    def predict(self, history: np.ndarray) -> np.ndarray:
        """Forecast each AP from ``history[AP, time, 6]`` to ``[AP, horizon, 4]``."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save model weights and metadata beneath ``path``."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> Forecaster:
        """Load a saved forecaster directory."""


def validate_history(history: np.ndarray) -> tuple[np.ndarray, bool]:
    """Normalize one or many history windows to ``[AP, time, 6]``."""

    values = np.asarray(history, dtype=float)
    single = values.ndim == 2
    if single:
        values = values[np.newaxis, ...]
    if values.ndim != 3 or values.shape[-1] != FEATURE_COUNT or values.shape[1] < 1:
        raise ValueError("history must have shape [time, 6] or [AP, time, 6]")
    if not np.all(np.isfinite(values)):
        raise ValueError("history must contain only finite values")
    return values, single


def validate_training_data(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate training arrays using the SRS dataset shapes."""

    features = np.asarray(x, dtype=np.float32)
    targets = np.asarray(y, dtype=np.float32)
    if features.ndim != 3 or features.shape[1:] != (30, FEATURE_COUNT):
        raise ValueError("training x must have shape [samples, 30, 6]")
    if targets.shape != (features.shape[0], len(DEFAULT_HORIZONS), len(TARGET_FEATURES)):
        raise ValueError("training y must have shape [samples, 3, 4]")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
        raise ValueError("training data must be finite")
    return features, targets
