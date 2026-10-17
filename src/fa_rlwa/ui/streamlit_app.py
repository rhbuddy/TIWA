"""Simple local Streamlit dashboard for FA-RLWA experiments."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from fa_rlwa.config import load_scenario
from fa_rlwa.experiments.runner import run_experiment
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


def _display_result(run: dict[str, Any]) -> None:
    """Show summary metrics, plots, and a CSV download for the last run."""

    summary = run["summary"]
    run_dir = Path(run["output_dir"])
    st.subheader("Latest run")
    st.caption(f"Output: `{run_dir}`")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Aggregate throughput", f"{summary['agg_tput_mbps']:.2f} Mbps")
    metric_columns[1].metric("Jain fairness", f"{summary['jfi']:.3f}")
    metric_columns[2].metric("Handovers", str(summary["handovers"]))
    metric_columns[3].metric("Mean occupancy", f"{summary['occ_mean']:.1%}")
    st.dataframe([summary], hide_index=True, use_container_width=True)

    summary_path = run_dir / "metrics_summary.csv"
    st.download_button(
        "Download summary CSV",
        data=summary_path.read_bytes(),
        file_name=summary_path.name,
        mime="text/csv",
    )
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
                with st.spinner("Running the experiment…"):
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
                    )
                st.session_state["last_run"] = {
                    "summary": summary,
                    "output_dir": str(output_dir),
                }
                st.success("Experiment completed.")
            except Exception as exc:  # The UI should report a failed run without crashing the app.
                st.error(f"Experiment failed: {exc}")

    last_run = st.session_state.get("last_run")
    if last_run:
        _display_result(last_run)


if __name__ == "__main__":
    main()
