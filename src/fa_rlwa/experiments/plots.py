"""Reproducible labelled plots for one experiment run."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fa_rlwa.monitor.indicators import jain_fairness


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV table as dictionaries."""

    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def plot_run(run_dir: str | Path) -> list[Path]:
    """Write throughput, per-AP occupancy, and fairness time-series plots."""

    root = Path(run_dir)
    ap_rows = _read_csv(root / "ap_indicators.csv")
    sta_rows = _read_csv(root / "sta_log.csv")
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    by_time_tput: dict[int, float] = defaultdict(float)
    by_time_ap: dict[tuple[int, int], float] = {}
    by_time_sta: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in ap_rows:
        t, ap_id = int(row["t"]), int(row["ap_id"])
        by_time_tput[t] += float(row["tput_mbps"])
        by_time_ap[(t, ap_id)] = float(row["occ_frac"])
    for row in sta_rows:
        demand = float(row["offered_mbps"])
        throughput = float(row["tput_mbps"])
        by_time_sta[int(row["t"])].append((throughput, demand))
    times = sorted(by_time_tput)
    outputs: list[Path] = []

    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(times, [by_time_tput[t] for t in times], color="#1570ef", linewidth=1.5)
    axis.set(title="Aggregate throughput over time", xlabel="Time (s)", ylabel="Throughput (Mbps)")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "throughput_over_time.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    outputs.append(path)

    ap_ids = sorted({ap_id for _, ap_id in by_time_ap})
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for ap_id in ap_ids:
        axis.plot(
            times,
            [by_time_ap.get((t, ap_id), 0.0) * 100.0 for t in times],
            label=f"AP {ap_id}",
            linewidth=1.3,
        )
    axis.set(title="Per-AP channel occupancy", xlabel="Time (s)", ylabel="Occupancy (%)")
    axis.set_ylim(0, 100)
    axis.legend(title="Access point", ncol=min(5, max(1, len(ap_ids))))
    axis.grid(alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "per_ap_load.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    outputs.append(path)

    fairness = []
    for t in times:
        samples = by_time_sta.get(t, [])
        satisfaction = [
            throughput / demand if demand > 0 else 0.0 for throughput, demand in samples
        ]
        fairness.append(jain_fairness(satisfaction))
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(times, fairness, color="#12b76a", linewidth=1.5)
    axis.set(title="Jain's fairness over time", xlabel="Time (s)", ylabel="Jain's fairness index")
    axis.set_ylim(0, 1.05)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "fairness_over_time.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    outputs.append(path)
    return outputs


def main() -> None:
    """CLI entry point for plotting an existing run directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run output directory")
    args = parser.parse_args()
    for path in plot_run(args.run):
        print(path)


if __name__ == "__main__":
    main()
