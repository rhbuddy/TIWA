"""Small optional-extra smoke tests; skipped when a model library is not installed."""

import numpy as np
import pytest


def _sample_arrays(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(8, 30, 6)).astype(np.float32)
    y = np.repeat(x[:, -1:, :4], 3, axis=1)
    y += rng.normal(scale=0.05, size=y.shape).astype(np.float32)
    return x, y


def test_xgboost_validation_early_stopping_and_round_trip(tmp_path) -> None:
    pytest.importorskip("xgboost")
    from fa_rlwa.forecast.trees import XGBoostForecaster

    x, y = _sample_arrays()
    model = XGBoostForecaster(
        n_estimators=8, max_depth=2, learning_rate=0.1, early_stopping_rounds=2
    )
    model.fit(x[:6], y[:6], validation_data=(x[6:], y[6:]))
    prediction = model.predict(x[6:])
    assert prediction.shape == (2, 3, 4)
    model.save(tmp_path)
    loaded = XGBoostForecaster.load(tmp_path)
    assert loaded.early_stopping_rounds == 2
    assert loaded.predict(x[6:]).shape == (2, 3, 4)


def test_lstm_external_validation_and_checkpoint_round_trip(tmp_path) -> None:
    pytest.importorskip("torch")
    from fa_rlwa.forecast.lstm import LSTMForecaster

    x, y = _sample_arrays(seed=4)
    model = LSTMForecaster(
        hidden_size=8,
        layers=1,
        dropout=0.0,
        epochs=3,
        batch_size=4,
        patience=2,
        seed=2,
    )
    model.fit(x[:6], y[:6], validation_data=(x[6:], y[6:]))
    prediction = model.predict(x[6:])
    assert prediction.shape == (2, 3, 4)
    model.save(tmp_path)
    loaded = LSTMForecaster.load(tmp_path)
    assert loaded.patience == 2
    np.testing.assert_allclose(loaded.predict(x[6:]), prediction, atol=1e-5)
