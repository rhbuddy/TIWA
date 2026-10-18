"""Dataset tagging and comparative forecast evaluation tests."""

import csv

import numpy as np

from fa_rlwa.forecast.dataset import _window_samples
from fa_rlwa.forecast.evaluate import evaluate_models
from fa_rlwa.forecast.ewma import EWMAForecaster, MovingAverageForecaster


def test_dataset_marks_windows_around_large_client_count_events() -> None:
    series = np.zeros((150, 1, 6), dtype=float)
    series[:, :, 0] = -60.0
    series[40:, 0, 4] = 5.0
    _, _, metadata = _window_samples(
        series,
        warmup_s=0,
        run_id="flash_s0",
        scenario="flash",
        policy="random",
        seed=0,
        sample_every_s=1,
    )
    labels = {row["t"]: row["event_window"] for row in metadata}
    assert labels[35]
    assert not labels[110]


def test_comparative_evaluation_writes_paired_event_metrics(tmp_path) -> None:
    rng = np.random.default_rng(17)
    x = rng.normal(size=(6, 30, 6)).astype(np.float32)
    y = np.zeros((6, 3, 4), dtype=np.float32)
    y[:, :, 0] = -60.0
    y[:, :, 2] = 5.0
    y[:, :, 3] = 0.2
    y[:, :, 1] = np.linspace(0.05, 0.3, 6)[:, None]
    dataset = tmp_path / "test_unseen.npz"
    np.savez_compressed(
        dataset,
        X=x,
        y=y,
        run_id=np.asarray(["run0", "run0", "run1", "run1", "run2", "run2"]),
        event_window=np.asarray([True, True, False, False, True, True]),
    )
    ewma_dir = tmp_path / "ewma"
    ma_dir = tmp_path / "moving_average"
    EWMAForecaster(alpha=0.3).save(ewma_dir)
    MovingAverageForecaster(window=10).save(ma_dir)

    metrics, comparisons = evaluate_models(
        [ewma_dir, ma_dir], [dataset], tmp_path / "forecast_metrics.csv"
    )

    assert len(metrics) == 2 * 2 * 3 * 4
    assert comparisons
    assert all(row["n_pairs"] >= 2 for row in comparisons)
    assert all("p_holm" in row for row in comparisons)
    with (tmp_path / "forecast_metrics.csv").open(newline="") as stream:
        saved_metrics = list(csv.DictReader(stream))
    with (tmp_path / "forecast_metrics_paired_tests.csv").open(newline="") as stream:
        saved_tests = list(csv.DictReader(stream))
    assert len(saved_metrics) == len(metrics)
    assert len(saved_tests) == len(comparisons)
    assert {row["scope"] for row in saved_metrics} == {"all", "event_window"}
