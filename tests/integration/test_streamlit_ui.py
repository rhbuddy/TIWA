"""The optional dashboard should render without requiring a Linux radio on Windows."""

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
