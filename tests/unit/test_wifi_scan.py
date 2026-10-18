"""Tests for the optional Linux Wi-Fi adapter scan preflight."""

from types import SimpleNamespace

from fa_rlwa.validation import wifi_scan


def test_scan_preflight_counts_bss_without_exposing_network_names(monkeypatch) -> None:
    monkeypatch.setattr(wifi_scan, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(wifi_scan.shutil, "which", lambda name: "/usr/sbin/iw")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[-1] == "info":
            return SimpleNamespace(
                returncode=0, stdout="Interface wlan0\ntype managed\n", stderr=""
            )
        return SimpleNamespace(
            returncode=0,
            stdout="BSS 00:11:22:33:44:55(on wlan0)\n\tSSID: private-name\nBSS aa:bb:cc:dd:ee:ff\n",
            stderr="",
        )

    monkeypatch.setattr(wifi_scan.subprocess, "run", fake_run)
    result = wifi_scan.check_wifi_scan("wlan0")

    assert result.can_scan
    assert result.visible_bss_count == 2
    assert "private-name" not in result.detail
    assert calls == [
        ["/usr/sbin/iw", "dev", "wlan0", "info"],
        ["/usr/sbin/iw", "dev", "wlan0", "scan"],
    ]


def test_empty_scan_is_supported_but_scan_error_fails(monkeypatch) -> None:
    monkeypatch.setattr(wifi_scan, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(wifi_scan.shutil, "which", lambda name: "/usr/sbin/iw")
    responses = iter(
        [
            SimpleNamespace(returncode=0, stdout="Interface wlan0\n", stderr=""),
            SimpleNamespace(returncode=0, stdout="", stderr=""),
        ]
    )
    monkeypatch.setattr(wifi_scan.subprocess, "run", lambda *args, **kwargs: next(responses))
    empty = wifi_scan.check_wifi_scan("wlan0")
    assert empty.can_scan and empty.visible_bss_count == 0

    responses = iter(
        [
            SimpleNamespace(returncode=0, stdout="Interface wlan0\n", stderr=""),
            SimpleNamespace(returncode=1, stdout="", stderr="Operation not permitted"),
        ]
    )
    monkeypatch.setattr(wifi_scan.subprocess, "run", lambda *args, **kwargs: next(responses))
    failed = wifi_scan.check_wifi_scan("wlan0")
    assert not failed.can_scan
    assert "Operation not permitted" in failed.detail


def test_invalid_interface_and_missing_iw_fail_safely(monkeypatch) -> None:
    monkeypatch.setattr(wifi_scan, "sys", SimpleNamespace(platform="linux"))
    assert not wifi_scan.check_wifi_scan("wlan0; echo nope").can_scan
    assert not wifi_scan.check_wifi_scan("wlan0", timeout_s=float("nan")).can_scan
    monkeypatch.setattr(wifi_scan.shutil, "which", lambda name: None)
    missing_tool = wifi_scan.check_wifi_scan("wlan0")
    assert not missing_tool.can_scan
    assert "iw" in missing_tool.detail
