"""End-to-end CLI pipeline smoke test."""

import csv
from pathlib import Path

from fa_rlwa.experiments.runner import run_experiment


def test_runner_writes_srs_compatible_run_artifacts(tmp_path: Path) -> None:
    result = run_experiment(
        "configs/scenarios/sc1_steady.yaml",
        "client_driven",
        7,
        tmp_path / "run",
        duration_s=5,
        make_plots=False,
    )
    run_dir = tmp_path / "run"
    for name in (
        "ap_indicators.csv",
        "sta_log.csv",
        "handover_log.csv",
        "metrics_summary.csv",
        "config_resolved.yaml",
    ):
        assert (run_dir / name).is_file()
    with (run_dir / "metrics_summary.csv").open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["run_id"] == "sc1_steady_client_driven_s7"
    assert float(row["decision_ms_mean"]) >= 0
    assert result["seed"] == 7
