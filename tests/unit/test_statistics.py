"""Experiment-analysis utility tests."""

import csv
from pathlib import Path

import numpy as np
import pytest

from fa_rlwa.controller.handover import count_ping_pongs
from fa_rlwa.experiments.report import build_report
from fa_rlwa.experiments.stats import bootstrap_mean_ci, holm_adjust, paired_wilcoxon


def test_bootstrap_interval_is_deterministic_and_contains_mean() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    first = bootstrap_mean_ci(values, resamples=2_000, seed=11)
    second = bootstrap_mean_ci(values, resamples=2_000, seed=11)
    assert first == second
    assert first[0] <= np.mean(values) <= first[1]


def test_holm_correction_preserves_original_order() -> None:
    np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_paired_wilcoxon_and_ping_pong_definition() -> None:
    assert paired_wilcoxon([3, 4, 5], [1, 2, 3]) == pytest.approx(0.25)
    events = [
        {"t": 1, "sta_id": 2, "from_ap": 0, "to_ap": 1},
        {"t": 8, "sta_id": 2, "from_ap": 1, "to_ap": 0},
        {"t": 20, "sta_id": 3, "from_ap": 0, "to_ap": 1},
        {"t": 35, "sta_id": 3, "from_ap": 1, "to_ap": 0},
    ]
    assert count_ping_pongs(events, window_s=10) == 1
    with pytest.raises(ValueError, match="window_s"):
        count_ping_pongs(events, window_s=-1)


def test_report_includes_paired_bootstrap_difference_interval(tmp_path: Path) -> None:
    source = tmp_path / "runs"
    for policy, values in (("client_driven", [1.0, 2.0, 3.0]), ("tiwa_original", [4.0, 5.0, 6.0])):
        for seed, throughput in enumerate(values):
            run_dir = source / f"run-{policy}-{seed}"
            run_dir.mkdir(parents=True)
            with (run_dir / "metrics_summary.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["scenario", "policy", "seed", "agg_tput_mbps"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "scenario": "fixture",
                        "policy": policy,
                        "seed": seed,
                        "agg_tput_mbps": throughput,
                    }
                )

    _, paired_path = build_report(source, tmp_path / "report")
    with paired_path.open(encoding="utf-8", newline="") as stream:
        comparison = next(csv.DictReader(stream))
    assert comparison["metric"] == "agg_tput_mbps"
    assert float(comparison["mean_difference_b_minus_a"]) == pytest.approx(3.0)
    assert float(comparison["difference_ci95_low"]) <= 3.0
    assert float(comparison["difference_ci95_high"]) >= 3.0
