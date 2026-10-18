"""Platform-specific failures should be explicit while the simulator stays portable."""

from types import SimpleNamespace

import pytest

from fa_rlwa.validation import wifi_scan
from fa_rlwa.validation.mininet_wifi import runner as mininet_runner


def test_wifi_scan_preflight_reports_windows_as_unsupported(monkeypatch) -> None:
    monkeypatch.setattr(wifi_scan, "sys", SimpleNamespace(platform="win32"))
    def fail_lookup(_name: str) -> str:
        raise AssertionError("iw lookup should be skipped")

    monkeypatch.setattr(wifi_scan.shutil, "which", fail_lookup)

    result = wifi_scan.check_wifi_scan("Wi-Fi")

    assert not result.can_scan
    assert "only on Linux" in result.detail


def test_mininet_backend_reports_windows_as_unsupported(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(mininet_runner, "sys", SimpleNamespace(platform="win32"))

    with pytest.raises(RuntimeError, match="Linux-only"):
        mininet_runner.run_mininet_validation(
            "configs/scenarios/sc1_steady.yaml",
            "client_driven",
            0,
            tmp_path / "mininet",
            generate_traffic=False,
        )
