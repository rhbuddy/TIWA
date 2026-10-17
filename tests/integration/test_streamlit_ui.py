"""The optional dashboard exposes live and completed-run diagnostics."""

import sys
from pathlib import Path

import pytest


def _app_path() -> Path:
    repository_root = Path(__file__).resolve().parents[2]
    return repository_root / "src" / "fa_rlwa" / "ui" / "streamlit_app.py"


def test_streamlit_dashboard_renders_without_exceptions() -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(_app_path()), default_timeout=15).run()
    assert len(app.exception) == 0


def test_windows_dashboard_disables_wifi_preflight(monkeypatch) -> None:
    pytest.importorskip("streamlit")
    # Load platform-sensitive optional dependencies before simulating Windows.
    from streamlit.testing.v1 import AppTest

    import fa_rlwa.experiments.runner  # noqa: F401

    app_path = _app_path()
    monkeypatch.setattr(sys, "platform", "win32")
    app = AppTest.from_file(str(app_path), default_timeout=15).run()

    assert len(app.exception) == 0
    assert any("Wi-Fi scan preflight is Linux-only" in info.value for info in app.info)


def test_dashboard_displays_completed_run_traces(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from fa_rlwa.experiments.runner import run_experiment

    run_dir = tmp_path / "ui-run"
    summary = run_experiment(
        "configs/scenarios/sc1_steady.yaml",
        "client_driven",
        11,
        run_dir,
        duration_s=5,
        make_plots=False,
    )
    app = AppTest.from_file(str(_app_path()), default_timeout=30).run()
    app.session_state["last_run"] = {"summary": summary, "output_dir": str(run_dir)}
    app.run()

    assert len(app.exception) == 0
    detail_selector = next(widget for widget in app.selectbox if widget.label == "Run details")
    assert {
        "AP telemetry",
        "Client telemetry",
        "Decisions & events",
        "Resolved config",
    } <= set(detail_selector.options)

    detail_selector.set_value("AP telemetry").run()
    assert len(app.exception) == 0
    assert len(app.dataframe) == 1

    detail_selector = next(widget for widget in app.selectbox if widget.label == "Run details")
    detail_selector.set_value("Files").run()
    artifact_selector = next(widget for widget in app.selectbox if widget.label == "Run file")
    artifact_selector.set_value("run_events.csv").run()
    download_labels = {button.label for button in app.get("download_button")}
    assert "Download run_events.csv" in download_labels


def test_dashboard_renders_live_controller_snapshot(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    snapshot = {
        "elapsed_s": 1,
        "duration_s": 3,
        "metrics": {
            "aggregate_tput_mbps": 12.5,
            "offered_load_mbps": 15.0,
            "jain_fairness": 0.98,
            "mean_ap_occupancy": 0.4,
            "occupancy_std": 0.05,
            "mean_retries_per_frame": 0.2,
            "decision_latency_ms": 1.2,
            "active_stations": 1,
            "handovers_this_step": 1,
            "invalid_actions_this_step": 0,
        },
        "ap_indicators": [{"t": 1, "ap_id": 0, "n_clients": 1, "tput_mbps": 12.5}],
        "stations": [{"t": 1, "sta_id": 0, "ap_id": 0, "tput_mbps": 12.5}],
        "decisions": [{"action": "handover", "entity": "STA 0", "target": "AP 0"}],
        "events": [{"t": 1, "event_type": "handover", "entity": "STA 0", "details": "{}"}],
    }
    script = tmp_path / "live_progress_app.py"
    script.write_text(
        "import streamlit as st\n"
        "from fa_rlwa.experiments.runner import ExperimentProgress\n"
        "from fa_rlwa.ui.streamlit_app import _render_live_snapshot\n"
        f"snapshot = ExperimentProgress(**{snapshot!r})\n"
        "with st.status('Controller running', expanded=True) as status:\n"
        "    st.progress(snapshot.elapsed_s / snapshot.duration_s, text='Test progress')\n"
        "    _render_live_snapshot(snapshot, snapshot.events)\n"
        "    status.update(label='Run complete', state='complete', expanded=False)\n",
        encoding="utf-8",
    )
    app = AppTest.from_file(str(script), default_timeout=15).run()

    assert len(app.exception) == 0
    assert len(app.dataframe) >= 4
