"""EWMA and moving-average forecast baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fa_rlwa.forecast.base import DEFAULT_HORIZONS, TARGET_FEATURES, Forecaster, validate_history


class EWMAForecaster(Forecaster):
    """Exponentially weighted level forecast repeated over each requested horizon."""

    def __init__(self, alpha: float = 0.3, horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("EWMA alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.horizons = tuple(horizons)

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> EWMAForecaster:
        """Select alpha on optional validation arrays; fitting is otherwise parameter-free."""

        if y is None or len(x) == 0:
            return self
        candidates = np.arange(0.1, 1.0, 0.1)
        best_alpha, best_error = self.alpha, float("inf")
        for alpha in candidates:
            trial = EWMAForecaster(float(alpha), self.horizons)
            predictions = trial.predict(np.asarray(x))
            if predictions.shape[1] != y.shape[1]:
                predictions = predictions[:, : y.shape[1], :]
            error = float(np.mean(np.abs(predictions - np.asarray(y))))
            if error < best_error:
                best_alpha, best_error = float(alpha), error
        self.alpha = best_alpha
        return self

    def predict(self, history: np.ndarray) -> np.ndarray:
        """Return the EWMA level for RSSI, loss, throughput, and occupancy."""

        values, single = validate_history(history)
        result = np.empty((values.shape[0], len(self.horizons), len(TARGET_FEATURES)))
        for ap_id, series in enumerate(values):
            level = series[0, list(TARGET_FEATURES)].astype(float)
            for row in series[1:]:
                level = self.alpha * row[list(TARGET_FEATURES)] + (1.0 - self.alpha) * level
            result[ap_id, :, :] = level
        return result[0] if single else result

    def save(self, path: str | Path) -> None:
        """Save alpha, horizons, and model type as JSON metadata."""

        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.json").write_text(
            json.dumps({"model": "ewma", "alpha": self.alpha, "horizons": self.horizons}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> EWMAForecaster:
        """Load an EWMA model directory."""

        metadata = json.loads((Path(path) / "model.json").read_text(encoding="utf-8"))
        return cls(float(metadata["alpha"]), tuple(metadata.get("horizons", DEFAULT_HORIZONS)))


class MovingAverageForecaster(Forecaster):
    """Ten-report moving-average persistence baseline used by TiWA."""

    def __init__(self, window: int = 10, horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> None:
        if window < 1:
            raise ValueError("moving-average window must be positive")
        self.window = int(window)
        self.horizons = tuple(horizons)

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> MovingAverageForecaster:
        """The moving-average baseline has no learned parameters."""

        return self

    def predict(self, history: np.ndarray) -> np.ndarray:
        """Repeat the latest-window mean for every configured horizon."""

        values, single = validate_history(history)
        level = np.mean(values[:, -self.window :, :][:, :, list(TARGET_FEATURES)], axis=1)
        result = np.repeat(level[:, np.newaxis, :], len(self.horizons), axis=1)
        return result[0] if single else result

    def save(self, path: str | Path) -> None:
        """Save window size and horizons as JSON metadata."""

        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.json").write_text(
            json.dumps(
                {"model": "moving_average", "window": self.window, "horizons": self.horizons},
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> MovingAverageForecaster:
        """Load a moving-average model directory."""

        metadata = json.loads((Path(path) / "model.json").read_text(encoding="utf-8"))
        return cls(int(metadata["window"]), tuple(metadata.get("horizons", DEFAULT_HORIZONS)))
