"""End-to-end CLI pipeline smoke test."""

import csv
from pathlib import Path

from fa_rlwa.experiments.runner import run_experiment


def test_runner_writes_srs_compatible_run_artifacts_and_streams_progress(tmp_path: Path) -> None:
    progress = []
    result = run_experiment(
        "configs/scenarios/sc1_steady.yaml",
        "client_driven",
        7,
        tmp_path / "run",
        duration_s=5,
        make_plots=False,
        progress_callback=progress.append,
    )
    run_dir = tmp_path / "run"
    for name in (
        "ap_indicators.csv",
        "sta_log.csv",
        "handover_log.csv",
        "channel_change_log.csv",
        "invalid_actions.csv",
        "run_events.csv",
        "metrics_summary.csv",
        "config_resolved.yaml",
    ):
        assert (run_dir / name).is_file()
    with (run_dir / "metrics_summary.csv").open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["run_id"] == "sc1_steady_client_driven_s7"
    assert float(row["decision_ms_mean"]) >= 0
    assert result["seed"] == 7
    assert [snapshot.elapsed_s for snapshot in progress] == [1, 2, 3, 4, 5]
    assert progress[-1].ap_indicators
    assert progress[-1].stations
    assert progress[-1].metrics["active_stations"] == len(progress[-1].stations)
    with (run_dir / "run_events.csv").open(encoding="utf-8", newline="") as stream:
        event_types = {event["event_type"] for event in csv.DictReader(stream)}
    assert {"run_started", "run_completed"}.issubset(event_types)
