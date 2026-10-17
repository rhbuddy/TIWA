"""Aggregate run CSVs into confidence intervals and paired significance tests."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fa_rlwa.experiments.stats import bootstrap_mean_ci, holm_adjust, paired_wilcoxon

METRICS = (
    "agg_tput_mbps",
    "jfi",
    "retx_per_frame",
    "occ_mean",
    "occ_std",
    "occ_channel_std",
    "phy_mean_mbps",
    "handovers",
    "pingpongs",
    "decision_ms_mean",
    "decision_ms_p95",
)


def _read_summary(path: Path) -> list[dict[str, str]]:
    """Load one metrics_summary.csv file."""

    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def build_report(results_dir: str | Path, output_dir: str | Path) -> tuple[Path, Path]:
    """Aggregate all completed run directories under ``results_dir``."""

    source, target = Path(results_dir), Path(output_dir)
    summaries = [row for path in source.rglob("metrics_summary.csv") for row in _read_summary(path)]
    if not summaries:
        raise FileNotFoundError(f"No metrics_summary.csv files found under {source}")
    target.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in summaries:
        grouped[(row["scenario"], row["policy"])].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (scenario, policy), rows in sorted(grouped.items()):
        for metric in METRICS:
            if metric not in rows[0]:
                continue
            values = np.asarray([float(row[metric]) for row in rows], dtype=float)
            low, high = bootstrap_mean_ci(values, seed=0)
            summary_rows.append(
                {
                    "scenario": scenario,
                    "policy": policy,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    summary_path = target / "aggregate_metrics.csv"
    _write_csv(summary_path, summary_rows)

    raw_comparisons: list[dict[str, Any]] = []
    for scenario in sorted({row["scenario"] for row in summaries}):
        scenario_rows = [row for row in summaries if row["scenario"] == scenario]
        policies = sorted({row["policy"] for row in scenario_rows})
        for metric in METRICS:
            for left_index, left in enumerate(policies):
                for right in policies[left_index + 1 :]:
                    left_rows = {
                        int(row["seed"]): float(row[metric])
                        for row in scenario_rows
                        if row["policy"] == left and metric in row
                    }
                    right_rows = {
                        int(row["seed"]): float(row[metric])
                        for row in scenario_rows
                        if row["policy"] == right and metric in row
                    }
                    seeds = sorted(set(left_rows) & set(right_rows))
                    if len(seeds) < 2:
                        continue
                    values_a = np.asarray([left_rows[seed] for seed in seeds], dtype=float)
                    values_b = np.asarray([right_rows[seed] for seed in seeds], dtype=float)
                    paired_differences = values_b - values_a
                    delta_low, delta_high = bootstrap_mean_ci(paired_differences, seed=0)
                    p_value = paired_wilcoxon(values_a, values_b)
                    raw_comparisons.append(
                        {
                            "scenario": scenario,
                            "metric": metric,
                            "policy_a": left,
                            "policy_b": right,
                            "n_pairs": len(seeds),
                            "mean_a": float(np.mean(values_a)),
                            "mean_b": float(np.mean(values_b)),
                            "mean_difference_b_minus_a": float(np.mean(paired_differences)),
                            "difference_ci95_low": delta_low,
                            "difference_ci95_high": delta_high,
                            "relative_improvement_pct": _relative_improvement(
                                left_rows, right_rows, seeds
                            ),
                            "p_value": p_value,
                        }
                    )
    if raw_comparisons:
        adjusted = holm_adjust([float(row["p_value"]) for row in raw_comparisons])
        for comparison, p_adjusted in zip(raw_comparisons, adjusted, strict=True):
            comparison["p_holm"] = float(p_adjusted)
    tests_path = target / "paired_tests.csv"
    _write_csv(tests_path, raw_comparisons)
    _plot_aggregate(summary_rows, target / "figures")
    return summary_path, tests_path


def _relative_improvement(a: dict[int, float], b: dict[int, float], seeds: list[int]) -> float:
    """Return relative mean improvement of B over A, in percent."""

    mean_a = float(np.mean([a[seed] for seed in seeds]))
    mean_b = float(np.mean([b[seed] for seed in seeds]))
    return (mean_b - mean_a) / abs(mean_a) * 100.0 if mean_a else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a flexible table, preserving first-row field order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_aggregate(rows: list[dict[str, Any]], figure_dir: Path) -> None:
    """Create aggregate throughput and fairness bars with bootstrap error bars."""

    figure_dir.mkdir(parents=True, exist_ok=True)
    for metric, title, ylabel in (
        ("agg_tput_mbps", "Aggregate throughput by policy", "Throughput (Mbps)"),
        ("jfi", "Fairness by policy", "Jain's fairness index"),
    ):
        items = [row for row in rows if row["metric"] == metric]
        if not items:
            continue
        scenarios = sorted({str(row["scenario"]) for row in items})
        policies = sorted({str(row["policy"]) for row in items})
        x = np.arange(len(scenarios))
        width = 0.8 / max(1, len(policies))
        fig, axis = plt.subplots(figsize=(max(7, len(scenarios) * 1.5), 4.5))
        for index, policy in enumerate(policies):
            subset = {str(row["scenario"]): row for row in items if row["policy"] == policy}
            means = [
                float(subset[scenario]["mean"]) if scenario in subset else 0.0
                for scenario in scenarios
            ]
            lows = [
                float(subset[scenario]["ci95_low"]) if scenario in subset else 0.0
                for scenario in scenarios
            ]
            highs = [
                float(subset[scenario]["ci95_high"]) if scenario in subset else 0.0
                for scenario in scenarios
            ]
            errors = np.asarray(
                [
                    [mean - low for mean, low in zip(means, lows, strict=False)],
                    [h - m for m, h in zip(means, highs, strict=False)],
                ]
            )
            axis.bar(
                x + (index - (len(policies) - 1) / 2) * width,
                means,
                width,
                yerr=errors,
                capsize=3,
                label=policy,
            )
        axis.set(title=title, xlabel="Scenario", ylabel=ylabel, xticks=x, xticklabels=scenarios)
        axis.legend(title="Policy")
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{metric}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    """CLI entry point for aggregating completed experiments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="parent directory containing run outputs")
    parser.add_argument("--out", required=True, help="report output directory")
    args = parser.parse_args()
    for path in build_report(args.results, args.out):
        print(path)


if __name__ == "__main__":
    main()
