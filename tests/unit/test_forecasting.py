"""Pure-numpy forecast baselines and input contract tests."""

import numpy as np
import pytest

from fa_rlwa.forecast.arima import ARIMAForecaster
from fa_rlwa.forecast.ewma import EWMAForecaster, MovingAverageForecaster


def test_forecaster_shapes_and_ewma_persistence() -> None:
    history = np.zeros((2, 30, 6), dtype=float)
    history[0, :, :4] = [(-65.0, 0.1, 7.0, 0.2)]
    history[1, :, :4] = [(-70.0, 0.2, 4.0, 0.4)]
    prediction = EWMAForecaster(alpha=1.0).predict(history)
    assert prediction.shape == (2, 3, 4)
    np.testing.assert_allclose(prediction[0], np.tile([-65.0, 0.1, 7.0, 0.2], (3, 1)))
    assert EWMAForecaster().predict(history[0]).shape == (3, 4)


def test_moving_average_uses_latest_ten_reports() -> None:
    history = np.zeros((12, 6), dtype=float)
    history[:, :4] = np.arange(1, 13, dtype=float)[:, None]
    prediction = MovingAverageForecaster(window=10).predict(history)
    assert prediction.shape == (3, 4)
    np.testing.assert_allclose(prediction, np.full((3, 4), 7.5))


def test_forecast_history_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="history must have shape"):
        EWMAForecaster().predict(np.zeros((3, 5)))


def test_moving_average_model_round_trip(tmp_path) -> None:
    original = MovingAverageForecaster(window=10)
    original.save(tmp_path)
    loaded = MovingAverageForecaster.load(tmp_path)
    assert loaded.window == 10
    np.testing.assert_array_equal(loaded.horizons, (5, 10, 30))


def test_arima_returns_fast_persistence_then_uses_background_cache(monkeypatch) -> None:
    class StubFit:
        def __init__(self, offset: float) -> None:
            self.offset = offset

        def forecast(self, steps: int) -> np.ndarray:
            return np.arange(steps, dtype=float) + self.offset

    history = np.zeros((2, 30, 6), dtype=float)
    history[:, :, :4] = 7.0
    model = ARIMAForecaster(auto_order=False)
    calls = []

    def fake_fit(histories):
        calls.append(histories.shape)
        return [[StubFit(10.0 + ap_id + target) for target in range(4)] for ap_id in range(2)]

    monkeypatch.setattr(model, "_fit_models", fake_fit)
    first = model.predict(history)
    assert first.shape == (2, 3, 4)
    np.testing.assert_allclose(first, 7.0)  # Cold-start fallback is persistence.
    assert model._fit_future is not None
    model._fit_future.result(timeout=2)
    second = model.predict(history)
    assert second[0, 0, 0] == 14.0  # First model, target RSSI, +5 s.
    assert second[1, 1, 3] == 23.0  # Second AP, occupancy target, +10 s.
    assert calls == [(2, 30, 6)]
    model.close()
