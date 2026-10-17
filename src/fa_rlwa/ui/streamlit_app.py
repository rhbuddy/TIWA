"""Simple local Streamlit dashboard for FA-RLWA experiments."""

from __future__ import annotations

import csv
import sys
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from fa_rlwa.config import load_scenario
from fa_rlwa.experiments.runner import ExperimentProgress, run_experiment
from fa_rlwa.validation.wifi_scan import check_wifi_scan

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCENARIO_DIR = REPOSITORY_ROOT / "configs" / "scenarios"
SCENARIOS = {path.stem: path for path in sorted(SCENARIO_DIR.glob("*.yaml"))}
POLICIES = (
    "client_driven",
    "tiwa_original",
    "tiwa_fixed",
    "tiwa_forecast",
    "random",
    "rl_forecast",
    "rl_noforecast",
)


def _wireless_interfaces() -> list[str]:
    """Return Linux network interfaces exposing the wireless sysfs marker."""

    if not sys.platform.startswith("linux"):
        return []
    interfaces = Path("/sys/class/net")
    if not interfaces.is_dir():
        return []
    return sorted(
        interface.name
        for interface in interfaces.iterdir()
        if (interface / "wireless").exists() or (interface / "phy80211").exists()
    )


def _resolve_output_root(value: str) -> Path:
    """Resolve a user-entered output path relative to the repository root."""

    path = Path(value).expanduser()
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _resolve_input_path(value: str | None) -> str | None:
    """Resolve an optional model path relative to the repository root."""

    if not value:
        return None
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else REPOSITORY_ROOT / path)


def _show_wifi_preflight(interface: str) -> bool:
    """Run the scan check and display its result without exposing SSIDs."""

    if not interface.strip():
        st.error("Enter a Wi-Fi interface name first, for example `wlan0`.")
        return False
    with st.spinner(f"Checking whether `{interface}` can scan..."):
        result = check_wifi_scan(interface.strip())
    if result.can_scan:
        st.success(f"Scan passed: {result.detail}.")
        return True
    st.error(f"Scan failed: {result.detail}.")
    return False


def _display_trace_file(path: Path, title: str, key: str) -> None:
    """Display a selectable tail of a trace and offer the complete CSV download."""

    if not path.is_file():
        st.info(f"`{path.name}` was not generated for this run.")
        return
    row_limit = st.selectbox(
        "Rows to display",
        [100, 500, 1_000, 5_000, "All"],
        index=2,
        key=f"trace-limit-{key}",
    )
    preview: list[dict[str, str]] = []
    total_rows = 0
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if row_limit == "All":
            preview = list(reader)
            total_rows = len(preview)
        else:
            limit = row_limit if isinstance(row_limit, int) else 1_000
            recent_rows: deque[dict[str, str]] = deque(maxlen=limit)
            for row in reader:
                total_rows += 1
                recent_rows.append(row)
            preview = list(recent_rows)
    if not preview:
        st.info(f"{title} has no records for this run.")
    else:
        st.caption(
            f"Showing {len(preview):,} of {total_rows:,} rows from `{path.name}`. "
            "The download always contains the complete trace."
        )
        st.dataframe(preview, hide_index=True, use_container_width=True, height=500)


def _display_result(run: dict[str, Any]) -> None:
    """Show summary metrics, full traces, events, resolved config, and artifacts."""

    summary = run["summary"]
    run_dir = Path(run["output_dir"])
    st.subheader("Latest run — complete run trace")
    st.caption(f"Output folder: `{run_dir}`")
    detail = st.selectbox(
        "Run details",
        [
            "Overview",
            "AP telemetry",
            "Client telemetry",
            "Decisions & events",
            "Resolved config",
            "Files",
        ],
        key="run-details-section",
    )

    if detail == "Overview":
        metric_columns = st.columns(6)
        metric_columns[0].metric("Throughput", f"{summary['agg_tput_mbps']:.2f} Mbps")
        metric_columns[1].metric("Jain fairness", f"{summary['jfi']:.3f}")
        metric_columns[2].metric("Handovers", str(summary["handovers"]))
        metric_columns[3].metric("Mean occupancy", f"{summary['occ_mean']:.1%}")
        metric_columns[4].metric("Invalid actions", str(summary["invalid_actions"]))
        metric_columns[5].metric("Decision p95", f"{summary['decision_ms_p95']:.2f} ms")
        st.dataframe([summary], hide_index=True, use_container_width=True)
        figures = run_dir / "figures"
        plot_files = (
            ("Throughput", figures / "throughput_over_time.png"),
            ("AP occupancy", figures / "per_ap_load.png"),
            ("Fairness", figures / "fairness_over_time.png"),
        )
        available_plots = [(title, path) for title, path in plot_files if path.exists()]
        if available_plots:
            st.subheader("Run plots")
            columns = st.columns(len(available_plots))
            for column, (title, path) in zip(columns, available_plots, strict=True):
                column.image(str(path), caption=title, use_column_width=True)
    elif detail == "AP telemetry":
        st.caption("Per-second channel, load, signal, throughput, and occupancy measurements.")
        _display_trace_file(run_dir / "ap_indicators.csv", "AP telemetry", "ap")
    elif detail == "Client telemetry":
        st.caption("Per-client association, location, load, PHY rate, retries, loss, and AP RSSI.")
        _display_trace_file(run_dir / "sta_log.csv", "Client telemetry", "stations")
    elif detail == "Decisions & events":
        st.caption(
            "Policy requests, executed handovers/channel changes, lifecycle changes, "
            "and rejected actions."
        )
        event_views = {
            "All events": ("run_events.csv", "All run events", "events"),
            "Handovers": ("handover_log.csv", "Handovers", "handovers"),
            "Channel changes": ("channel_change_log.csv", "Channel changes", "channels"),
            "Rejected actions": ("invalid_actions.csv", "Rejected actions", "invalid"),
        }
        selected_view = st.selectbox("Event view", list(event_views), key="event-view")
        filename, title, key = event_views[selected_view]
        _display_trace_file(run_dir / filename, title, key)
    elif detail == "Resolved config":
        config_path = run_dir / "config_resolved.yaml"
        if config_path.is_file():
            st.code(config_path.read_text(encoding="utf-8"), language="yaml")
        else:
            st.info("Resolved configuration was not written for this run.")
    else:
        artifact_names = (
            "metrics_summary.csv",
            "ap_indicators.csv",
            "sta_log.csv",
            "handover_log.csv",
            "channel_change_log.csv",
            "invalid_actions.csv",
            "run_events.csv",
            "config_resolved.yaml",
        )
        available_artifacts = [name for name in artifact_names if (run_dir / name).is_file()]
        if not available_artifacts:
            st.info("No downloadable files were written for this run.")
            return
        st.caption("Choose one exact output file to download.")
        selected_artifact = st.selectbox(
            "Run file", available_artifacts, key="run-artifact-selection"
        )
        artifact_path = run_dir / selected_artifact
        st.caption(f"File size: {artifact_path.stat().st_size:,} bytes")
        st.download_button(
            f"Download {selected_artifact}",
            data=artifact_path.read_bytes(),
            file_name=selected_artifact,
            mime="text/yaml" if artifact_path.suffix == ".yaml" else "text/csv",
            key="selected-artifact-download",
        )


def _render_live_snapshot(
    progress: ExperimentProgress, recent_events: list[dict[str, Any]]
) -> None:
    """Render the current controller cycle while the simulation is executing."""

    st.subheader(f"Live simulator state · second {progress.elapsed_s:,}")
    primary_metrics = st.columns(5)
    primary_metrics[0].metric(
        "Aggregate throughput", f"{progress.metrics['aggregate_tput_mbps']:.2f} Mbps"
    )
    primary_metrics[1].metric("Offered load", f"{progress.metrics['offered_load_mbps']:.2f} Mbps")
    primary_metrics[2].metric("Jain fairness", f"{progress.metrics['jain_fairness']:.3f}")
    primary_metrics[3].metric("Mean AP occupancy", f"{progress.metrics['mean_ap_occupancy']:.1%}")
    primary_metrics[4].metric("Occupancy spread", f"{progress.metrics['occupancy_std']:.3f}")

    control_metrics = st.columns(5)
    control_metrics[0].metric("Active clients", str(progress.metrics["active_stations"]))
    control_metrics[1].metric(
        "Mean retries/frame", f"{progress.metrics['mean_retries_per_frame']:.3f}"
    )
    control_metrics[2].metric(
        "Decision latency", f"{progress.metrics['decision_latency_ms']:.2f} ms"
    )
    control_metrics[3].metric("Handovers this tick", str(progress.metrics["handovers_this_step"]))
    control_metrics[4].metric(
        "Rejected actions", str(progress.metrics["invalid_actions_this_step"])
    )

    st.markdown("**Current AP telemetry**")
    st.dataframe(progress.ap_indicators, hide_index=True, use_container_width=True, height=180)
    st.markdown("**Current client state**")
    st.dataframe(progress.stations, hide_index=True, use_container_width=True, height=240)
    st.markdown("**Policy requests and events**")
    if progress.decisions:
        st.dataframe(progress.decisions, hide_index=True, use_container_width=True)
    else:
        st.caption("The policy requested no handover or channel change on this cycle.")
    if progress.events:
        st.dataframe(progress.events, hide_index=True, use_container_width=True)
    else:
        st.caption("No network events or rejected actions on this cycle.")
    if recent_events:
        st.caption("Recent event history (up to 50 newest records)")
        st.dataframe(recent_events[-50:], hide_index=True, use_container_width=True, height=220)


def main() -> None:
    """Render the experiment launcher and most recent run results."""

    st.set_page_config(page_title="FA-RLWA", page_icon="📶", layout="wide")
    st.title("FA-RLWA")
    st.caption("Forecast-aware Wi-Fi association experiments")
    st.info(
        "This dashboard runs the abstract simulator. A physical adapter is only scanned "
        "when you explicitly request the optional preflight."
    )
    if not SCENARIOS:
        st.error(f"No scenario YAML files were found in `{SCENARIO_DIR}`.")
        return

    with st.sidebar:
        st.header("Wi-Fi adapter")
        if sys.platform.startswith("linux"):
            interfaces = _wireless_interfaces()
            if interfaces:
                adapter = st.selectbox("Interface", interfaces)
            else:
                adapter = st.text_input("Interface", placeholder="wlan0")
            if st.button("Test scan capability", use_container_width=True):
                _show_wifi_preflight(adapter)
            require_scan = st.checkbox("Require a successful scan before each run", value=False)
            st.caption("Linux `iw` is required. Scanning is never started automatically.")
        else:
            adapter = ""
            require_scan = False
            st.info("Wi-Fi scan preflight is Linux-only; simulation runs normally on Windows.")

    with st.form("experiment_form"):
        st.subheader("Experiment settings")
        first, second, third = st.columns(3)
        scenario_key = first.selectbox("Scenario", list(SCENARIOS))
        policy = second.selectbox("Policy", POLICIES)
        seed = third.number_input("Seed", min_value=0, max_value=2**31 - 1, value=0, step=1)

        scenario = load_scenario(SCENARIOS[scenario_key])
        duration = st.number_input(
            "Duration (seconds)", min_value=1, max_value=86_400, value=120, step=60
        )
        output_root = st.text_input("Output folder", value="results/ui_runs")

        forecast_model = "ewma"
        forecast_model_path: str | None = None
        model_path: str | None = None
        k_max = max(5, len(scenario.aps))
        if policy == "tiwa_forecast":
            forecast_model = st.selectbox("Built-in forecaster", ["ewma", "moving_average"])
            forecast_model_path = (
                st.text_input(
                    "Optional saved forecaster folder", placeholder="models/forecast/ewma_v1"
                )
                or None
            )
        if policy in {"rl_forecast", "rl_noforecast"}:
            model_path = (
                st.text_input("Trained RL checkpoint", placeholder="models/rl/dqn_forecast_s0")
                or None
            )
            k_max = int(
                st.number_input(
                    "RL K_max (must match training)",
                    min_value=1,
                    max_value=10,
                    value=min(10, k_max),
                    step=1,
                )
            )
            if policy == "rl_forecast":
                forecast_model_path = (
                    st.text_input(
                        "Optional saved forecaster folder", placeholder="models/forecast/ewma_v1"
                    )
                    or None
                )

        submitted = st.form_submit_button("Run experiment", type="primary")

    if submitted:
        if policy in {"rl_forecast", "rl_noforecast"} and not model_path:
            st.error("Select a trained RL checkpoint before running an RL policy.")
        elif require_scan and not _show_wifi_preflight(adapter):
            st.error("The experiment was not started because the adapter scan failed.")
        else:
            output_dir = _resolve_output_root(output_root) / (
                f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_"
                f"{scenario_key}_{policy}_s{int(seed)}"
            )
            if duration < scenario.warmup_s:
                st.warning(
                    f"This run is shorter than the {scenario.warmup_s}s warm-up; "
                    "post-warm-up summary metrics will be zero."
                )
            try:
                recent_events: list[dict[str, Any]] = []
                refresh_every = max(1, int(duration) // 40)
                last_rendered_step = 0
                with st.status(
                    "Preparing scenario, policy, and simulator…", expanded=True
                ) as status:
                    st.write("Loading the selected configuration and starting the controller loop.")
                    progress_bar = st.progress(0.0, text="Initializing simulator…")
                    live_panel = st.empty()

                    def update_progress(progress: ExperimentProgress) -> None:
                        nonlocal last_rendered_step
                        recent_events.extend(progress.events)
                        if len(recent_events) > 250:
                            del recent_events[:-250]
                        should_render = (
                            progress.elapsed_s == progress.duration_s
                            or progress.events
                            or progress.elapsed_s - last_rendered_step >= refresh_every
                        )
                        if should_render:
                            last_rendered_step = progress.elapsed_s
                            progress_bar.progress(
                                progress.elapsed_s / progress.duration_s,
                                text=(
                                    f"Simulated {progress.elapsed_s:,} / "
                                    f"{progress.duration_s:,} seconds"
                                ),
                            )
                            status.update(
                                label=(
                                    f"Controller running · second {progress.elapsed_s:,} "
                                    f"of {progress.duration_s:,}"
                                ),
                                state="running",
                                expanded=True,
                            )
                            with live_panel.container():
                                _render_live_snapshot(progress, recent_events)

                    summary = run_experiment(
                        SCENARIOS[scenario_key],
                        policy,
                        int(seed),
                        output_dir,
                        duration_s=int(duration),
                        model_path=_resolve_input_path(model_path),
                        forecast_model_path=_resolve_input_path(forecast_model_path),
                        forecast_model=forecast_model,
                        k_max=k_max,
                        make_plots=True,
                        progress_callback=update_progress,
                    )
                    progress_bar.progress(1.0, text="Run artifacts written")
                    status.update(
                        label="Experiment completed · full trace and artifacts are available below",
                        state="complete",
                        expanded=False,
                    )
                st.session_state["last_run"] = {
                    "summary": summary,
                    "output_dir": str(output_dir),
                }
                st.success(
                    "Experiment completed. Inspect telemetry and decisions in the tabs below."
                )
            except Exception as exc:  # The UI should report a failed run without crashing the app.
                st.error(f"Experiment failed: {exc}")
                with st.expander("Technical details", expanded=True):
                    st.code(traceback.format_exc(), language="text")

    last_run = st.session_state.get("last_run")
    if last_run:
        _display_result(last_run)


if __name__ == "__main__":
    main()
