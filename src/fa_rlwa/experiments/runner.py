"""Run one deterministic scenario/policy experiment and write traceable CSV outputs."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import logging
import os
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fa_rlwa.config import load_scenario, scenario_to_mapping
from fa_rlwa.controller.controller import SDNController
from fa_rlwa.experiments.metrics import channel_occupancy_std, summarize_run
from fa_rlwa.experiments.plots import plot_run
from fa_rlwa.policies.factory import make_policy
from fa_rlwa.sim.network import WifiNetwork
from fa_rlwa.types import Action, ChannelChange, Handover
from fa_rlwa.validation.wifi_scan import check_wifi_scan

LOGGER = logging.getLogger(__name__)

AP_FIELDS = [
    "run_id",
    "t",
    "ap_id",
    "channel",
    "n_clients",
    "rssi_dbm",
    "loss_frac",
    "tput_mbps",
    "occ_frac",
    "airtime_own_frac",
    "demand_mbps",
]
STA_BASE_FIELDS = [
    "run_id",
    "t",
    "sta_id",
    "ap_id",
    "x_m",
    "y_m",
    "phy_mbps",
    "offered_mbps",
    "tput_mbps",
    "p_fail",
    "retx_per_frame",
    "traffic_type",
]
HANDOVER_FIELDS = ["run_id", "t", "sta_id", "from_ap", "to_ap", "policy", "trigger"]
CHANNEL_CHANGE_FIELDS = [
    "run_id",
    "t",
    "ap_id",
    "from_channel",
    "to_channel",
    "policy",
    "trigger",
]
RUN_EVENT_FIELDS = ["run_id", "t", "event_type", "entity", "details"]


@dataclass(frozen=True)
class ExperimentProgress:
    """Latest controller/simulator snapshot exposed to live UI monitors."""

    elapsed_s: int
    duration_s: int
    metrics: dict[str, float | int]
    ap_indicators: list[dict[str, Any]]
    stations: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    events: list[dict[str, Any]]


ProgressCallback = Callable[[ExperimentProgress], None]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write CSV with stable column ordering, including an empty header-only table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _event_row(run_id: str, t: int, event_type: str, entity: str, details: Any) -> dict[str, Any]:
    """Create a structured, JSON-safe event row for UI display and CSV export."""

    return {
        "run_id": run_id,
        "t": t,
        "event_type": event_type,
        "entity": entity,
        "details": json.dumps(details, sort_keys=True, default=str),
    }


def _action_record(action: Action) -> dict[str, Any]:
    """Normalize a policy request for live display and the event trail."""

    if isinstance(action, Handover):
        return {
            "action": "handover",
            "entity": f"STA {action.sta_id}",
            "target": f"AP {action.target_ap}",
            "reason": action.reason,
        }
    if isinstance(action, ChannelChange):
        return {
            "action": "channel_change",
            "entity": f"AP {action.ap_id}",
            "target_channel": action.channel,
            "reason": action.reason,
        }
    return {"action": type(action).__name__, "details": repr(action)}


def _git_commit() -> str:
    """Return the checked-out Git revision, or 'unknown' outside a Git checkout."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _versions() -> dict[str, str]:
    """Capture versions of installed runtime libraries for run traceability."""

    packages = ("fa-rlwa", "numpy", "PyYAML", "gymnasium", "matplotlib")
    result = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _load_forecaster(path: str | None, model_type: str) -> Any:
    """Load a saved forecaster or create an EWMA/moving-average baseline."""

    if path:
        metadata_path = Path(path) / "model.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No model.json found under forecast model directory {path}")
        import json

        model_name = json.loads(metadata_path.read_text(encoding="utf-8")).get("model", model_type)
        if model_name == "ewma":
            from fa_rlwa.forecast.ewma import EWMAForecaster

            return EWMAForecaster.load(path)
        if model_name == "moving_average":
            from fa_rlwa.forecast.ewma import MovingAverageForecaster

            return MovingAverageForecaster.load(path)
        if model_name == "arima":
            from fa_rlwa.forecast.arima import ARIMAForecaster

            return ARIMAForecaster.load(path)
        if model_name == "random_forest":
            from fa_rlwa.forecast.trees import RandomForestForecaster

            return RandomForestForecaster.load(path)
        if model_name == "xgboost":
            from fa_rlwa.forecast.trees import XGBoostForecaster

            return XGBoostForecaster.load(path)
        if model_name == "lstm":
            from fa_rlwa.forecast.lstm import LSTMForecaster

            return LSTMForecaster.load(path)
        raise ValueError(f"Unknown forecaster type in {metadata_path}: {model_name}")
    if model_type == "moving_average":
        from fa_rlwa.forecast.ewma import MovingAverageForecaster

        return MovingAverageForecaster()
    if model_type == "ewma":
        from fa_rlwa.forecast.ewma import EWMAForecaster

        return EWMAForecaster()
    raise ValueError(f"A saved model directory is required for forecaster {model_type!r}")


def run_experiment(
    scenario_path: str | Path,
    policy_name: str,
    seed: int,
    output_dir: str | Path,
    *,
    duration_s: int | None = None,
    model_path: str | None = None,
    forecast_model_path: str | None = None,
    forecast_model: str = "ewma",
    policy_config: str | Path | None = None,
    k_max: int | None = None,
    make_plots: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Execute a run, optionally streaming each control-cycle snapshot to a UI."""

    scenario = load_scenario(scenario_path)
    duration = int(duration_s or scenario.duration_s)
    if duration < 1:
        raise ValueError("duration_s must be positive")
    settings: dict[str, Any] = {}
    if policy_config:
        with Path(policy_config).open(encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream) or {}
        settings = {key: value for key, value in loaded.items() if key != "policy"}
    forecast = None
    if policy_name in {"tiwa_forecast", "rl_forecast"}:
        forecast = _load_forecaster(forecast_model_path, forecast_model)
    policy = make_policy(
        policy_name,
        forecaster=forecast,
        model_path=model_path,
        k_max=int(k_max or max(5, len(scenario.aps))),
        use_forecast=policy_name != "rl_noforecast",
        settings=settings,
    )
    run_id = f"{scenario.name}_{policy_name}_s{seed}"
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    controller = SDNController(WifiNetwork(scenario, seed), policy, seed)
    state = controller.reset(seed)
    ap_rows: list[dict[str, Any]] = []
    sta_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    report_rows = []
    max_ap = len(scenario.aps)
    sta_fields = STA_BASE_FIELDS + [f"rssi_dbm_ap{i}" for i in range(max_ap)]
    run_event_rows = [
        _event_row(
            run_id,
            0,
            "run_started",
            "controller",
            {
                "scenario": scenario.name,
                "policy": policy_name,
                "seed": seed,
                "duration_s": duration,
                "warmup_s": scenario.warmup_s,
            },
        )
    ]

    for _ in range(duration):
        previous_ap_by_sta = {sta.sta_id: sta.ap_id for sta in state.stas}
        previous_sta_ids = set(previous_ap_by_sta)
        handover_start = len(controller.network.handover_events)
        channel_change_start = len(controller.network.channel_events)
        invalid_action_start = len(controller.network.invalid_action_events)

        state = controller.step()
        report = state.reports[-1]
        network = controller.network
        step_metrics = dict(network.last_step_metrics)
        step_metrics["occ_std"] = channel_occupancy_std(report, state.aps)
        if state.t >= scenario.warmup_s:
            step_rows.append(step_metrics)
            report_rows.append(report)

        ap_tick_rows: list[dict[str, Any]] = []
        for ap_id, ap in enumerate(state.aps):
            row = {
                "run_id": run_id,
                "t": state.t,
                "ap_id": ap_id,
                "channel": ap.channel,
                "n_clients": int(report.n_clients[ap_id]),
                "rssi_dbm": float(report.rssi_dbm[ap_id]),
                "loss_frac": float(report.loss_frac[ap_id]),
                "tput_mbps": float(report.tput_mbps[ap_id]),
                "occ_frac": float(report.occ_frac[ap_id]),
                "airtime_own_frac": float(report.airtime_own_frac[ap_id]),
                "demand_mbps": float(report.demand_mbps[ap_id]),
            }
            ap_rows.append(row)
            ap_tick_rows.append(row)

        sta_tick_rows: list[dict[str, Any]] = []
        for row_index, sta in enumerate(state.stas):
            item: dict[str, Any] = {
                "run_id": run_id,
                "t": state.t,
                "sta_id": sta.sta_id,
                "ap_id": sta.ap_id,
                "x_m": sta.pos[0],
                "y_m": sta.pos[1],
                "phy_mbps": float(state.sta_phy_mbps[row_index]),
                "offered_mbps": float(state.sta_demand_mbps[row_index]),
                "tput_mbps": float(state.sta_tput_mbps[row_index]),
                "p_fail": float(state.sta_loss_frac[row_index]),
                "retx_per_frame": float(state.sta_retx_per_frame[row_index]),
                "traffic_type": sta.traffic_type,
            }
            item.update(
                {
                    f"rssi_dbm_ap{ap_id}": float(state.rssi_dbm[row_index, ap_id])
                    for ap_id in range(max_ap)
                }
            )
            sta_rows.append(item)
            sta_tick_rows.append(item)

        decisions = [_action_record(action) for action in controller.last_actions]
        step_events: list[dict[str, Any]] = []
        for decision in decisions:
            step_events.append(
                _event_row(run_id, state.t, "policy_action", str(decision["entity"]), decision)
            )

        new_handovers = network.handover_events[handover_start:]
        explicit_handover_ids = {int(str(event["sta_id"])) for event in new_handovers}
        for event in new_handovers:
            step_events.append(
                _event_row(
                    run_id,
                    int(str(event.get("t", state.t))),
                    "handover",
                    f"STA {event.get('sta_id', '?')}",
                    event,
                )
            )
        for event in network.channel_events[channel_change_start:]:
            step_events.append(
                _event_row(
                    run_id,
                    int(str(event.get("t", state.t))),
                    "channel_change",
                    f"AP {event.get('ap_id', '?')}",
                    event,
                )
            )
        for event in network.invalid_action_events[invalid_action_start:]:
            entity = (
                f"STA {event['sta_id']}"
                if "sta_id" in event
                else f"AP {event['ap_id']}"
                if "ap_id" in event
                else "controller"
            )
            step_events.append(
                _event_row(
                    run_id,
                    int(str(event.get("t", state.t))),
                    "invalid_action",
                    entity,
                    event,
                )
            )

        current_stas = {sta.sta_id: sta for sta in state.stas}
        for sta_id in sorted(current_stas.keys() - previous_sta_ids):
            sta = current_stas[sta_id]
            step_events.append(
                _event_row(
                    run_id,
                    state.t,
                    "station_joined",
                    f"STA {sta_id}",
                    {"ap_id": sta.ap_id, "traffic_type": sta.traffic_type},
                )
            )
        for sta_id in sorted(previous_sta_ids - current_stas.keys()):
            step_events.append(
                _event_row(
                    run_id,
                    state.t,
                    "station_left",
                    f"STA {sta_id}",
                    {"previous_ap_id": previous_ap_by_sta[sta_id]},
                )
            )
        for sta_id in sorted(previous_sta_ids & current_stas.keys()):
            sta = current_stas[sta_id]
            previous_ap = previous_ap_by_sta[sta_id]
            if sta.ap_id != previous_ap and sta_id not in explicit_handover_ids:
                step_events.append(
                    _event_row(
                        run_id,
                        state.t,
                        "association_fallback",
                        f"STA {sta_id}",
                        {"from_ap": previous_ap, "to_ap": sta.ap_id},
                    )
                )

        run_event_rows.extend(step_events)
        if progress_callback is not None:
            progress_callback(
                ExperimentProgress(
                    elapsed_s=state.t,
                    duration_s=duration,
                    metrics={
                        "aggregate_tput_mbps": float(step_metrics["agg_tput_mbps"]),
                        "offered_load_mbps": float(step_metrics["offered_mbps"]),
                        "jain_fairness": float(step_metrics["jfi"]),
                        "mean_ap_occupancy": float(step_metrics["occ_mean"]),
                        "occupancy_std": float(step_metrics["occ_std"]),
                        "mean_retries_per_frame": float(step_metrics["retx_per_frame"]),
                        "decision_latency_ms": float(step_metrics["decision_latency_ms"]),
                        "active_stations": len(state.stas),
                        "handovers_this_step": int(step_metrics["handovers"]),
                        "invalid_actions_this_step": int(step_metrics["invalid_actions"]),
                    },
                    ap_indicators=ap_tick_rows,
                    stations=sta_tick_rows,
                    decisions=decisions,
                    events=step_events,
                )
            )

    handover_rows = [
        {
            "run_id": run_id,
            "t": event["t"],
            "sta_id": event["sta_id"],
            "from_ap": event["from_ap"],
            "to_ap": event["to_ap"],
            "policy": policy_name,
            "trigger": event["reason"],
        }
        for event in controller.network.handover_events
    ]
    _write_csv(root / "ap_indicators.csv", AP_FIELDS, ap_rows)
    _write_csv(root / "sta_log.csv", sta_fields, sta_rows)
    _write_csv(root / "handover_log.csv", HANDOVER_FIELDS, handover_rows)
    channel_change_rows = [
        {
            "run_id": run_id,
            "t": event["t"],
            "ap_id": event["ap_id"],
            "from_channel": event["from_channel"],
            "to_channel": event["to_channel"],
            "policy": policy_name,
            "trigger": event["reason"],
        }
        for event in controller.network.channel_events
    ]
    _write_csv(root / "channel_change_log.csv", CHANNEL_CHANGE_FIELDS, channel_change_rows)
    invalid_rows = [
        {"run_id": run_id, **event} for event in controller.network.invalid_action_events
    ]
    _write_csv(
        root / "invalid_actions.csv",
        ["run_id", "t", "sta_id", "target_ap", "ap_id", "channel", "reason"],
        invalid_rows,
    )
    summary = summarize_run(
        run_id=run_id,
        scenario=scenario.name,
        policy=policy_name,
        seed=seed,
        step_rows=step_rows,
        reports=report_rows,
        aps=state.aps,
        handover_events=controller.network.handover_events,
        decision_latencies_ms=controller.decision_latencies_ms,
    )
    run_event_rows.append(
        _event_row(
            run_id,
            duration,
            "run_completed",
            "controller",
            {
                "total_events": len(run_event_rows),
                "aggregate_tput_mbps": summary["agg_tput_mbps"],
                "jain_fairness": summary["jfi"],
                "handovers": summary["handovers"],
                "invalid_actions": summary["invalid_actions"],
            },
        )
    )
    _write_csv(root / "run_events.csv", RUN_EVENT_FIELDS, run_event_rows)
    _write_csv(root / "metrics_summary.csv", list(summary), [summary])
    resolved = {
        "run_id": run_id,
        "scenario": scenario_to_mapping(scenario),
        "policy": policy_name,
        "policy_settings": settings,
        "seed": seed,
        "duration_s": duration,
        "git_commit": _git_commit(),
        "library_versions": _versions(),
    }
    (root / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    if make_plots:
        plot_run(root)
    LOGGER.info(
        "Finished %s: throughput=%.3f Mbps, JFI=%.3f",
        run_id,
        summary["agg_tput_mbps"],
        summary["jfi"],
    )
    return summary


def _parse_seeds(value: str) -> list[int]:
    """Parse comma-separated seeds or an inclusive ``start-end`` range."""

    if "-" in value:
        start, end = value.split("-", maxsplit=1)
        return list(range(int(start), int(end) + 1))
    return [int(item) for item in value.split(",") if item]


def run_matrix(
    matrix_path: str | Path,
    *,
    output_root: str | Path | None = None,
    seeds_override: list[int] | None = None,
    jobs: int = 1,
    duration_s: int | None = None,
    make_plots: bool = True,
) -> list[dict[str, Any]]:
    """Run the configured scenario x policy x seed matrix, optionally in parallel."""

    if jobs < 1:
        raise ValueError("jobs must be at least one")
    with Path(matrix_path).open(encoding="utf-8") as stream:
        matrix = yaml.safe_load(stream) or {}
    scenarios = list(matrix.get("scenarios", []))
    policies = list(matrix.get("policies", []))
    seeds_cfg = matrix.get("seeds", {"start": 0, "stop": 1})
    seeds = seeds_override or (
        list(range(int(seeds_cfg["start"]), int(seeds_cfg["stop"])))
        if isinstance(seeds_cfg, dict)
        else [int(seed) for seed in seeds_cfg]
    )
    root = Path(output_root or matrix.get("out", "results"))
    tasks: list[dict[str, Any]] = []
    for scenario_path in scenarios:
        scenario = load_scenario(scenario_path)
        for policy_entry in policies:
            if isinstance(policy_entry, str):
                policy_name, policy_options = policy_entry, {}
            else:
                policy_name = str(policy_entry["name"])
                policy_options = dict(policy_entry)
            for seed in seeds:
                tasks.append(
                    {
                        "scenario_path": scenario_path,
                        "policy_name": policy_name,
                        "seed": int(seed),
                        "output_dir": root / f"{scenario.name}_{policy_name}_s{seed}",
                        "model_path": policy_options.get("model"),
                        "forecast_model_path": policy_options.get("forecast_model"),
                        "forecast_model": policy_options.get("forecaster", "ewma"),
                        "policy_config": policy_options.get("config"),
                        "k_max": policy_options.get("k_max"),
                        "duration_s": duration_s or matrix.get("duration_s"),
                    }
                )
    summaries: list[dict[str, Any]] = []
    if jobs == 1:
        for task in tasks:
            summaries.append(run_experiment(**task, make_plots=make_plots))
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(run_experiment, **task, make_plots=False): task for task in tasks
            }
            for future in as_completed(futures):
                summaries.append(future.result())
        if make_plots:
            for task in tasks:
                plot_run(task["output_dir"])
    return summaries


def main() -> None:
    """Run one experiment or a reproducible scenario/policy/seed matrix."""

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scenario", help="scenario YAML file")
    mode.add_argument("--matrix", help="YAML scenario/policy/seed matrix")
    parser.add_argument("--policy", default="client_driven", help="registered policy name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="run output directory, or results root with --matrix")
    parser.add_argument("--duration", type=int, help="override run duration in seconds")
    parser.add_argument("--model", help="trained RL checkpoint for rl_* policies")
    parser.add_argument("--forecast-model", help="saved forecaster directory")
    parser.add_argument(
        "--forecaster",
        choices=["ewma", "moving_average", "arima", "random_forest", "xgboost", "xgb", "lstm"],
        default="ewma",
    )
    parser.add_argument("--policy-config", help="optional YAML policy settings")
    parser.add_argument(
        "--k-max", type=int, help="RL observation AP count (default: max(5, scenario APs))"
    )
    parser.add_argument("--seeds", help="override matrix seeds, e.g. 200-219 or 2,4,6")
    parser.add_argument("--jobs", type=int, default=1, help="parallel matrix workers")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--wifi-adapter",
        help="run a Linux scan-capability preflight on this interface before starting",
    )
    parser.add_argument(
        "--wifi-scan-timeout", type=float, default=15.0, help="per-command preflight timeout"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    wifi_adapter = args.wifi_adapter or os.environ.get("FA_RLWA_WIFI_ADAPTER")
    if wifi_adapter:
        scan_check = check_wifi_scan(wifi_adapter, timeout_s=args.wifi_scan_timeout)
        status = "PASS" if scan_check.can_scan else "FAIL"
        print(f"Wi-Fi preflight {status}: {scan_check.interface}: {scan_check.detail}")
        if not scan_check.can_scan:
            parser.exit(2)
    if args.matrix:
        run_matrix(
            args.matrix,
            output_root=args.out,
            seeds_override=_parse_seeds(args.seeds) if args.seeds else None,
            jobs=args.jobs,
            duration_s=args.duration,
            make_plots=not args.no_plots,
        )
        return
    if not args.out:
        parser.error("--out is required when running a single --scenario")
    run_experiment(
        args.scenario,
        args.policy,
        args.seed,
        args.out,
        duration_s=args.duration,
        model_path=args.model,
        forecast_model_path=args.forecast_model,
        forecast_model=args.forecaster,
        policy_config=args.policy_config,
        k_max=args.k_max,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
