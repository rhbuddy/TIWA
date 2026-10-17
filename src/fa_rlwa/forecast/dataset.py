"""Generate scenario-separated forecaster datasets from repeatable simulator runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fa_rlwa.config import load_scenario
from fa_rlwa.controller.controller import SDNController
from fa_rlwa.policies.factory import make_policy
from fa_rlwa.sim.network import WifiNetwork


def _seed_values(value: Any) -> list[int]:
    """Parse an integer list or half-open ``[start, stop)`` seed range."""

    if isinstance(value, dict):
        return list(range(int(value["start"]), int(value["stop"])))
    if isinstance(value, list):
        return [int(seed) for seed in value]
    raise ValueError("seed configuration must be a list or {start, stop} mapping")


def _feature_row(state) -> np.ndarray:
    """Build one AP-by-feature matrix in the forecaster's documented feature order."""

    report = state.reports[-1]
    return np.stack(
        [
            report.rssi_dbm,
            report.loss_frac,
            report.tput_mbps,
            report.occ_frac,
            report.n_clients.astype(float),
            report.demand_mbps,
        ],
        axis=1,
    )


def _run_series(scenario_path: str, policy_name: str, seed: int) -> tuple[np.ndarray, int, str]:
    """Simulate one complete run and return features indexed by time and AP."""

    scenario = load_scenario(scenario_path)
    network = WifiNetwork(scenario, seed=seed)
    policy = make_policy(policy_name)
    controller = SDNController(network, policy, seed)
    state = controller.reset(seed)
    series = [_feature_row(state)]
    for _ in range(scenario.duration_s):
        state = controller.step()
        series.append(_feature_row(state))
    return np.stack(series, axis=0), scenario.warmup_s, scenario.name


def _window_samples(
    series: np.ndarray,
    warmup_s: int,
    run_id: str,
    scenario: str,
    policy: str,
    seed: int,
    sample_every_s: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    """Extract 30-step features and 5/10/30-second targets, one sample per AP."""

    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    first_t = max(29, warmup_s + 29)
    last_t = len(series) - 1 - 30
    event_times_by_ap: list[np.ndarray] = []
    for ap_id in range(series.shape[1]):
        counts = series[:, ap_id, 4]
        events = [
            t
            for t in range(10, len(counts))
            if float(np.max(counts[t - 10 : t + 1]) - np.min(counts[t - 10 : t + 1])) >= 5.0
        ]
        event_times_by_ap.append(np.asarray(events, dtype=int))
    for t in range(first_t, last_t + 1):
        if (t - warmup_s) % sample_every_s:
            continue
        for ap_id in range(series.shape[1]):
            x_rows.append(series[t - 29 : t + 1, ap_id, :])
            y_rows.append(
                np.stack([series[t + horizon, ap_id, :4] for horizon in (5, 10, 30)], axis=0)
            )
            event_times = event_times_by_ap[ap_id]
            in_event_window = bool(np.any(np.abs(event_times - t) <= 30))
            metadata.append(
                {
                    "run_id": run_id,
                    "scenario": scenario,
                    "policy": policy,
                    "seed": seed,
                    "t": t,
                    "ap_id": ap_id,
                    "event_window": in_event_window,
                }
            )
    return x_rows, y_rows, metadata


def generate(config_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Generate train/val/in-distribution/unseen ``.npz`` windows per YAML protocol."""

    with Path(config_path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    scenario_sets = cfg.get("scenarios", {})
    policy_names = cfg.get("policies", ["client_driven", "tiwa_original", "random"])
    seeds_cfg = cfg.get("seeds", {})
    sample_every = int(cfg.get("sample_every_s", 5))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for split in ("train", "val", "test_id", "test_unseen"):
        paths = scenario_sets.get(split, [])
        if not paths:
            continue
        seeds = _seed_values(seeds_cfg[split])
        x_all: list[np.ndarray] = []
        y_all: list[np.ndarray] = []
        metadata_all: list[dict[str, Any]] = []
        for scenario_path in paths:
            for policy_name in policy_names:
                for seed in seeds:
                    series, warmup, scenario_name = _run_series(scenario_path, policy_name, seed)
                    run_id = f"{scenario_name}_{policy_name}_s{seed}"
                    x_rows, y_rows, rows = _window_samples(
                        series, warmup, run_id, scenario_name, policy_name, seed, sample_every
                    )
                    x_all.extend(x_rows)
                    y_all.extend(y_rows)
                    metadata_all.extend(rows)
        if not x_all:
            raise ValueError(f"No forecast windows generated for split {split!r}")
        output_path = output / f"{split}.npz"
        np.savez_compressed(
            output_path,
            X=np.asarray(x_all, dtype=np.float32),
            y=np.asarray(y_all, dtype=np.float32),
            run_id=np.asarray([row["run_id"] for row in metadata_all], dtype="U128"),
            scenario=np.asarray([row["scenario"] for row in metadata_all], dtype="U64"),
            policy=np.asarray([row["policy"] for row in metadata_all], dtype="U32"),
            seed=np.asarray([row["seed"] for row in metadata_all], dtype=np.int32),
            t=np.asarray([row["t"] for row in metadata_all], dtype=np.int32),
            ap_id=np.asarray([row["ap_id"] for row in metadata_all], dtype=np.int16),
            event_window=np.asarray(
                [row["event_window"] for row in metadata_all], dtype=bool
            ),
        )
        result[split] = output_path
    return result


def main() -> None:
    """CLI entry point for forecaster dataset generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="data/forecast/v1")
    args = parser.parse_args()
    for split, path in generate(args.config, args.out).items():
        print(f"{split}: {path}")


if __name__ == "__main__":
    main()
