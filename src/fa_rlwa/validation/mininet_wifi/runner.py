"""Build and run a closed-loop FA-RLWA controller on Mininet-WiFi."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import logging
import os
import shlex
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fa_rlwa.config import ScenarioConfig, load_scenario, scenario_to_mapping
from fa_rlwa.experiments.metrics import channel_occupancy_std, summarize_run
from fa_rlwa.experiments.plots import plot_run
from fa_rlwa.forecast.base import Forecaster
from fa_rlwa.policies.factory import make_policy
from fa_rlwa.types import IndicatorReport, NetworkState
from fa_rlwa.validation.mininet_wifi.backend import MininetWifiBackend

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


def _positions(
    scenario: ScenarioConfig, seed: int
) -> tuple[list[tuple[float, float]], list[float], list[str]]:
    """Generate reproducible static station positions and offered traffic from a scenario."""

    if scenario.flash_crowd is not None and scenario.flash_crowd.count:
        raise ValueError("Mininet-WiFi backend currently supports static scenario populations only")
    rng = np.random.default_rng(seed)
    positions: list[tuple[float, float]] = []
    offered_mbps: list[float] = []
    traffic_types: list[str] = []
    for group in scenario.groups:
        if group.mobility != "static":
            raise ValueError(
                "Mininet-WiFi backend currently supports static station groups; "
                f"scenario requests {group.mobility!r} mobility"
            )
        for _ in range(group.count):
            if group.placement == "hotspot":
                center = scenario.aps[len(positions) % len(scenario.aps)].pos
                x, y = rng.normal(center[0], 3.0), rng.normal(center[1], 3.0)
                position = (
                    float(np.clip(x, 0.0, scenario.area_m[0])),
                    float(np.clip(y, 0.0, scenario.area_m[1])),
                )
            else:
                position = (
                    float(rng.uniform(0.0, scenario.area_m[0])),
                    float(rng.uniform(0.0, scenario.area_m[1])),
                )
            positions.append(position)
            offered_mbps.append(float(group.offered_mbps))
            traffic_types.append(group.traffic_type)
    if len(positions) > 100:
        raise ValueError("Mininet-WiFi validation is limited to 100 stations")
    return positions, offered_mbps, traffic_types


def _build_network(
    scenario: ScenarioConfig,
    seed: int,
    *,
    _mininet_factory: Any | None = None,
) -> tuple[Any, list[Any], list[Any], Any, list[float], list[str]]:
    """Create the SC1-style Mininet-WiFi AP/STA topology and a wired traffic sink."""

    if _mininet_factory is None:
        try:
            from mn_wifi.net import Mininet_wifi
        except ImportError as exc:
            raise RuntimeError(
                "Mininet-WiFi is not installed. Install it on Linux using the upstream "
                "installation instructions, then run this command with sudo."
            ) from exc
        _mininet_factory = Mininet_wifi

    positions, offered_mbps, traffic_types = _positions(scenario, seed)
    net = _mininet_factory(autoAssociation=True, allAutoAssociation=True)
    try:
        ap_nodes = [
            net.addAccessPoint(
                f"fa_ap{ap.ap_id}",
                ssid=f"fa-rlwa-{scenario.name}",
                mode="n",
                channel=str(ap.channel),
                position=f"{ap.pos[0]},{ap.pos[1]},0",
                failMode="standalone",
            )
            for ap in scenario.aps
        ]
        sta_nodes = [
            net.addStation(
                f"fa_sta{sta_id}",
                position=f"{position[0]},{position[1]},0",
                mode="n",
                ip=f"10.0.0.{sta_id + 1}/8",
            )
            for sta_id, position in enumerate(positions)
        ]
        net.setPropagationModel(model="logDistance", exp=scenario.propagation.exponent)
        core_switch = net.addSwitch("fa_s1", failMode="standalone")
        traffic_sink = net.addHost("fa_server", ip="10.0.0.254/8")

        # Match the documented Mininet-WiFi lifecycle: configure wireless interfaces,
        # add wired backhaul links, build, then explicitly start the APs and switch.
        net.configureNodes()
        for ap_node in ap_nodes:
            net.addLink(ap_node, core_switch)
        net.addLink(core_switch, traffic_sink)
        net.build()
        for ap_node in ap_nodes:
            ap_node.start([])
        core_switch.start([])
        return net, ap_nodes, sta_nodes, traffic_sink, offered_mbps, traffic_types
    except BaseException:
        try:
            net.stop()
        except Exception:
            LOGGER.exception("Mininet-WiFi cleanup failed after topology setup error")
        raise


def _start_iperf3_traffic(
    traffic_sink: Any,
    sta_nodes: list[Any],
    offered_mbps: list[float],
    duration_s: int,
    output_dir: Path,
) -> None:
    """Start isolated per-station UDP iperf3 flows toward a wired server node."""

    if not shutil.which("iperf3"):
        raise RuntimeError(
            "iperf3 is required for traffic generation; install it or pass --no-traffic"
        )
    if not str(traffic_sink.cmd("command -v iperf3")).strip():
        raise RuntimeError("iperf3 is not available inside the Mininet traffic-sink node")
    server_ip = str(traffic_sink.IP())
    for sta_id, (station, rate_mbps) in enumerate(zip(sta_nodes, offered_mbps, strict=True)):
        if rate_mbps <= 0:
            continue
        port = 5201 + sta_id
        server_log = shlex.quote(str(output_dir / f"iperf_server_sta{sta_id}.log"))
        client_log = shlex.quote(str(output_dir / f"iperf_client_sta{sta_id}.log"))
        traffic_sink.cmd(f"iperf3 -s -p {port} > {server_log} 2>&1 &")
        rate_arg = f"{rate_mbps:g}M"
        station.cmd(
            f"iperf3 -c {server_ip} -p {port} -u -b {rate_arg} "
            f"-t {duration_s + 5} -i 1 > {client_log} 2>&1 &"
        )


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    """Write a trace with a stable schema, including empty event files."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _append_state_rows(
    run_id: str,
    state: NetworkState,
    latency_ms: float,
    invalid_count: int,
    ap_rows: list[dict[str, Any]],
    sta_rows: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
    report_rows: list[IndicatorReport],
    warmup_s: int,
) -> None:
    """Translate one backend snapshot to the same core trace/KPI fields as the simulator."""

    report = state.reports[-1]
    for ap_id, ap in enumerate(state.aps):
        ap_rows.append(
            {
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
        )
    max_ap = len(state.aps)
    for row, sta in enumerate(state.stas):
        item: dict[str, Any] = {
            "run_id": run_id,
            "t": state.t,
            "sta_id": sta.sta_id,
            "ap_id": sta.ap_id,
            "x_m": sta.pos[0],
            "y_m": sta.pos[1],
            "phy_mbps": float(state.sta_phy_mbps[row]),
            "offered_mbps": float(state.sta_demand_mbps[row]),
            "tput_mbps": float(state.sta_tput_mbps[row]),
            "p_fail": float(state.sta_loss_frac[row]),
            "retx_per_frame": float(state.sta_retx_per_frame[row]),
            "traffic_type": sta.traffic_type,
        }
        item.update({f"rssi_dbm_ap{i}": float(state.rssi_dbm[row, i]) for i in range(max_ap)})
        sta_rows.append(item)
    if state.t >= warmup_s:
        aggregate_tput = float(np.sum(state.sta_tput_mbps))
        aggregate_demand = float(np.sum(state.sta_demand_mbps))
        occupancy = report.occ_frac
        step_rows.append(
            {
                "agg_tput_mbps": aggregate_tput,
                "offered_mbps": aggregate_demand,
                "jfi": _fairness(state),
                "retx_per_frame": float(np.mean(state.sta_retx_per_frame))
                if len(state.stas)
                else 0.0,
                "occ_mean": float(np.mean(occupancy)) if len(occupancy) else 0.0,
                "occ_std": channel_occupancy_std(report, state.aps),
                "phy_mean_mbps": float(np.mean(state.sta_phy_mbps))
                if len(state.stas)
                else 0.0,
                "invalid_actions": invalid_count,
                "decision_latency_ms": latency_ms,
            }
        )
        report_rows.append(report)


def _fairness(state: NetworkState) -> float:
    """Compute Jain fairness for normalized per-station throughput satisfaction."""

    from fa_rlwa.monitor.indicators import satisfaction_fairness

    return satisfaction_fairness(state.sta_tput_mbps, state.sta_demand_mbps)


def _load_policy_settings(path: str | Path | None) -> dict[str, Any]:
    """Load optional YAML policy parameters while keeping the policy name in the CLI."""

    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    return {key: value for key, value in values.items() if key != "policy"}


def _load_forecaster_if_needed(
    policy_name: str, model_path: str | None, forecaster_name: str, model_dir: str | None
) -> Forecaster | None:
    """Reuse the experiment runner's lazy forecaster loader for forecast-enabled policies."""

    if policy_name not in {"tiwa_forecast", "rl_forecast"}:
        return None
    from fa_rlwa.experiments.runner import _load_forecaster

    return _load_forecaster(model_dir, forecaster_name)


def run_mininet_validation(
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
    tick_s: float = 1.0,
    forever: bool = False,
    generate_traffic: bool = True,
    allow_channel_changes: bool = False,
    make_plots: bool = True,
    _mininet_factory: Any | None = None,
) -> dict[str, Any]:
    """Run a timed live policy loop against a Mininet-WiFi AP/STA topology.

    Requires Linux, Mininet-WiFi, root privileges, ``iw`` in Mininet namespaces, and
    optionally iperf3. The loop is interruptible with Ctrl-C and always tears down the
    emulated network. No host Wi-Fi interfaces are modified by this backend.
    """

    if _mininet_factory is None and not sys.platform.startswith("linux"):
        raise RuntimeError("Mininet-WiFi validation is Linux-only; use the simulator on Windows")
    if _mininet_factory is None and (
        not hasattr(os, "geteuid") or os.geteuid() != 0
    ):
        raise PermissionError("Mininet-WiFi requires root privileges; run with sudo on Linux")
    if _mininet_factory is None and not shutil.which("iw"):
        raise RuntimeError("The Linux `iw` utility is required for Mininet-WiFi telemetry")
    if not np.isfinite(tick_s) or tick_s <= 0:
        raise ValueError("tick_s must be finite and greater than zero")
    scenario = load_scenario(scenario_path)
    duration = int(duration_s if duration_s is not None else scenario.duration_s)
    if duration < 1:
        if forever:
            duration = max(1, scenario.duration_s)
        else:
            raise ValueError("duration_s must be positive unless --forever is enabled")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    network, ap_nodes, sta_nodes, traffic_sink, offered, traffic_types = _build_network(
        scenario, seed, _mininet_factory=_mininet_factory
    )
    try:
        backend = MininetWifiBackend(
            network,
            scenario,
            ap_nodes,
            sta_nodes,
            offered,
            traffic_types,
            allow_channel_changes=allow_channel_changes,
        )
    except BaseException:
        network.stop()
        raise
    try:
        if generate_traffic:
            traffic_duration = 86_400 if forever else duration
            _start_iperf3_traffic(traffic_sink, sta_nodes, offered, traffic_duration, root)
        settings = _load_policy_settings(policy_config)
        forecaster = _load_forecaster_if_needed(
            policy_name, model_path, forecast_model, forecast_model_path
        )
        policy = make_policy(
            policy_name,
            forecaster=forecaster,
            model_path=model_path,
            k_max=int(k_max or max(5, len(scenario.aps))),
            use_forecast=policy_name != "rl_noforecast",
            settings=settings,
        )
        run_id = f"{scenario.name}_{policy_name}_s{seed}_mininet"
        state = backend.snapshot(t=0)
        policy.reset(state, np.random.default_rng(seed))
        ap_rows: list[dict[str, Any]] = []
        sta_rows: list[dict[str, Any]] = []
        step_rows: list[dict[str, Any]] = []
        report_rows: list[IndicatorReport] = []
        decision_latencies_ms: list[float] = []
        invalid_seen = 0
        _append_state_rows(
            run_id,
            state,
            0.0,
            0,
            ap_rows,
            sta_rows,
            step_rows,
            report_rows,
            scenario.warmup_s,
        )
        start = time.monotonic()
        completed_steps = 0
        step = 0
        try:
            while forever or step < duration:
                cycle_start = time.monotonic()
                actions = policy.decide(state)
                backend.apply_actions(actions)
                latency_ms = (time.monotonic() - cycle_start) * 1000.0
                decision_latencies_ms.append(latency_ms)
                deadline = start + (step + 1) * tick_s
                time.sleep(max(0.0, deadline - time.monotonic()))
                state = backend.snapshot(t=step + 1)
                invalid_count = len(backend.invalid_action_events) - invalid_seen
                invalid_seen = len(backend.invalid_action_events)
                _append_state_rows(
                    run_id,
                    state,
                    latency_ms,
                    invalid_count,
                    ap_rows,
                    sta_rows,
                    step_rows,
                    report_rows,
                    scenario.warmup_s,
                )
                completed_steps = step + 1
                step += 1
        except KeyboardInterrupt:
            LOGGER.warning("Mininet-WiFi validation interrupted after %d cycles", completed_steps)

        handover_rows = [
            {
                "run_id": run_id,
                "t": event.get("t", 0),
                "sta_id": event.get("sta_id", ""),
                "from_ap": event.get("from_ap", ""),
                "to_ap": event.get("to_ap", ""),
                "policy": policy_name,
                "trigger": event.get("reason", ""),
            }
            for event in backend.handover_events
        ]
        invalid_rows = [{"run_id": run_id, **event} for event in backend.invalid_action_events]
        _write_csv(root / "ap_indicators.csv", AP_FIELDS, ap_rows)
        sta_fields = STA_BASE_FIELDS + [f"rssi_dbm_ap{i}" for i in range(len(scenario.aps))]
        _write_csv(root / "sta_log.csv", sta_fields, sta_rows)
        _write_csv(root / "handover_log.csv", HANDOVER_FIELDS, handover_rows)
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
            handover_events=backend.handover_events,
            decision_latencies_ms=decision_latencies_ms,
        )
        summary["backend"] = "mininet_wifi"
        summary["completed_cycles"] = completed_steps
        summary["traffic_generated"] = bool(generate_traffic)
        summary["run_until_interrupted"] = bool(forever)
        _write_csv(root / "metrics_summary.csv", list(summary), [summary])
        try:
            mininet_version = importlib.metadata.version("mininet-wifi")
        except importlib.metadata.PackageNotFoundError:
            mininet_version = "source-install-or-unknown"
        resolved = {
            "run_id": run_id,
            "backend": "mininet_wifi",
            "scenario": scenario_to_mapping(scenario),
            "policy": policy_name,
            "policy_settings": settings,
            "seed": int(seed),
            "duration_s": None if forever else duration,
            "forever": forever,
            "completed_cycles": completed_steps,
            "tick_s": tick_s,
            "traffic_generator": "iperf3/UDP" if generate_traffic else "disabled",
            "mininet_wifi_version": mininet_version,
            "allow_channel_changes": allow_channel_changes,
        }
        (root / "config_resolved.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
        )
        if make_plots:
            plot_run(root)
        return summary
    finally:
        backend.close(stop_network=True)
        close_method = getattr(forecaster if "forecaster" in locals() else None, "close", None)
        if callable(close_method):
            close_method()


def main() -> None:
    """CLI entry point for the Mininet-WiFi validation loop."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="configs/scenarios/sc1_steady.yaml")
    parser.add_argument(
        "--policy",
        default="client_driven",
        choices=[
            "client_driven",
            "tiwa_original",
            "tiwa_fixed",
            "tiwa_forecast",
            "random",
            "rl_forecast",
            "rl_noforecast",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=int, help="override scenario duration in seconds")
    parser.add_argument(
        "--forever",
        action="store_true",
        help="keep the controller loop running until Ctrl-C (traffic runs for up to 24 h)",
    )
    parser.add_argument("--out", default="results/mininet_validation")
    parser.add_argument("--model", help="trained RL checkpoint when selecting an RL policy")
    parser.add_argument("--forecast-model", help="saved forecaster folder")
    parser.add_argument(
        "--forecaster",
        default="ewma",
        choices=["ewma", "moving_average", "arima", "random_forest", "xgboost", "lstm"],
    )
    parser.add_argument("--policy-config", help="optional YAML policy parameters")
    parser.add_argument("--k-max", type=int, help="RL AP count used during training")
    parser.add_argument("--tick", type=float, default=1.0, help="control period in seconds")
    parser.add_argument(
        "--no-traffic", action="store_true", help="do not start iperf3 UDP flows"
    )
    parser.add_argument(
        "--allow-channel-changes",
        action="store_true",
        help="allow TiWA channel actions (may interrupt all clients on an AP)",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    try:
        result = run_mininet_validation(
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
            tick_s=args.tick,
            forever=args.forever,
            generate_traffic=not args.no_traffic,
            allow_channel_changes=args.allow_channel_changes,
            make_plots=not args.no_plots,
        )
    except (PermissionError, RuntimeError, ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(
        f"Completed {result['completed_cycles']} cycles on Mininet-WiFi: "
        f"throughput={result['agg_tput_mbps']:.3f} Mbps, "
        f"JFI={result['jfi']:.3f}, handovers={result['handovers']}"
    )


if __name__ == "__main__":
    main()
