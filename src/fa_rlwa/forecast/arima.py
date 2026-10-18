"""Per-AP ARIMA forecaster; statsmodels is loaded only when selected."""

from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from fa_rlwa.forecast.base import (
    DEFAULT_HORIZONS,
    Forecaster,
    validate_history,
    validate_training_data,
)


class ARIMAForecaster(Forecaster):
    """ARIMA forecast with cached models and periodic asynchronous refits.

    Model fitting is scheduled off the control path. Until the first fit completes,
    predictions fall back to persistence; subsequent predictions use the latest cached
    fit while a refresh runs in the background every ``refit_interval_s`` calls.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 0, 0),
        horizons: tuple[int, ...] = DEFAULT_HORIZONS,
        auto_order: bool = True,
        refit_interval_s: int = 60,
        history_window_s: int = 300,
    ) -> None:
        if refit_interval_s < 1 or history_window_s < 1:
            raise ValueError("ARIMA refit interval and history window must be positive")
        self.order = tuple(int(value) for value in order)
        self.horizons = tuple(horizons)
        self.auto_order = bool(auto_order)
        self.refit_interval_s = int(refit_interval_s)
        self.history_window_s = int(history_window_s)
        self._cached_models: list[list[Any | None]] | None = None
        self._fit_future: Future[list[list[Any | None]]] | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fa-rlwa-arima")
        self._prediction_calls = 0
        self._last_refit_requested = -self.refit_interval_s

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> ARIMAForecaster:
        """Select a shared ARIMA order by mean AIC on training windows."""

        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError as exc:
            raise ImportError(
                "Install statsmodels with pip install -r requirements-ml.txt"
            ) from exc
        if not self.auto_order or y is None:
            return self
        features, _ = validate_training_data(x, y)
        sample = features[: min(len(features), 16), :, :4]
        candidates = [(p, d, q) for p in range(4) for d in range(2) for q in range(4)]
        best_order, best_aic = self.order, float("inf")
        for order in candidates:
            scores = []
            for window in sample:
                for target in range(4):
                    try:
                        fitted = ARIMA(window[:, target], order=order).fit()
                        if np.isfinite(fitted.aic):
                            scores.append(float(fitted.aic))
                    except (ValueError, np.linalg.LinAlgError):
                        continue
            if scores and float(np.mean(scores)) < best_aic:
                best_order, best_aic = order, float(np.mean(scores))
        self.order = best_order
        return self

    def _fit_models(self, histories: np.ndarray) -> list[list[Any | None]]:
        """Fit one univariate ARIMA model for each AP and target indicator."""

        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError as exc:
            raise ImportError(
                "Install statsmodels with pip install -r requirements-ml.txt"
            ) from exc
        fitted_models: list[list[Any | None]] = []
        for window in histories:
            target_models: list[Any | None] = []
            for target in range(4):
                series = window[-self.history_window_s :, target]
                try:
                    target_models.append(ARIMA(series, order=self.order).fit())
                except (ValueError, np.linalg.LinAlgError):
                    target_models.append(None)
            fitted_models.append(target_models)
        return fitted_models

    def _update_cache(self) -> None:
        """Install a completed asynchronous refit, keeping the old fit if it failed."""

        if self._fit_future is None or not self._fit_future.done():
            return
        try:
            self._cached_models = self._fit_future.result()
        except Exception:
            # Persistence remains available as a safe fallback if a series cannot be fitted.
            pass
        self._fit_future = None

    def predict(self, history: np.ndarray) -> np.ndarray:
        """Predict from cached fits; schedule ARIMA refits outside the control path."""

        values, single = validate_history(history)
        self._prediction_calls += 1
        self._update_cache()
        refit_due = self._prediction_calls - self._last_refit_requested >= self.refit_interval_s
        if self._fit_future is None and (self._cached_models is None or refit_due):
            history_copy = values.copy()
            self._fit_future = self._executor.submit(self._fit_models, history_copy)
            self._last_refit_requested = self._prediction_calls

        output = np.empty((len(values), len(self.horizons), 4), dtype=float)
        maximum_horizon = max(self.horizons)
        for ap_id, window in enumerate(values):
            for target in range(4):
                prediction: np.ndarray | None = None
                if self._cached_models is not None and ap_id < len(self._cached_models):
                    fitted = self._cached_models[ap_id][target]
                    if fitted is not None:
                        try:
                            prediction = np.asarray(
                                fitted.forecast(steps=maximum_horizon), dtype=float
                            )
                        except (ValueError, np.linalg.LinAlgError):
                            prediction = None
                if prediction is None or not np.all(np.isfinite(prediction)):
                    prediction = np.repeat(window[-1, target], maximum_horizon)
                for index, horizon in enumerate(self.horizons):
                    output[ap_id, index, target] = prediction[horizon - 1]
        return output[0] if single else output

    def close(self) -> None:
        """Stop scheduling refits and release the background worker."""

        self._executor.shutdown(wait=False, cancel_futures=True)

    def save(self, path: str | Path) -> None:
        """Save ARIMA order and horizons as model metadata."""

        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.json").write_text(
            json.dumps(
                {
                    "model": "arima",
                    "order": self.order,
                    "horizons": self.horizons,
                    "auto_order": self.auto_order,
                    "refit_interval_s": self.refit_interval_s,
                    "history_window_s": self.history_window_s,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> ARIMAForecaster:
        """Load ARIMA hyperparameters from a model directory."""

        metadata = json.loads((Path(path) / "model.json").read_text(encoding="utf-8"))
        return cls(
            tuple(metadata["order"]),
            tuple(metadata.get("horizons", DEFAULT_HORIZONS)),
            bool(metadata.get("auto_order", False)),
            int(metadata.get("refit_interval_s", 60)),
            int(metadata.get("history_window_s", 300)),
        )
