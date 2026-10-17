"""Command-line dispatch and adapter-preflight tests."""

import sys

import pytest

from fa_rlwa.experiments import runner
from fa_rlwa.validation.wifi_scan import WifiScanCheck


def test_subcommand_help_is_forwarded_to_runner(capsys) -> None:
    from fa_rlwa.cli import main

    with pytest.raises(SystemExit) as error:
        main(["run", "--help"])
    assert error.value.code == 0
    assert "--wifi-adapter" in capsys.readouterr().out


def test_mininet_validation_help_is_exposed_by_top_level_cli(capsys) -> None:
    from fa_rlwa.cli import main

    with pytest.raises(SystemExit) as error:
        main(["validate-mininet", "--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "--no-traffic" in output
    assert "--allow-channel-changes" in output
    assert "--forever" in output


def test_runner_checks_adapter_before_starting(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fa-rlwa",
            "--scenario",
            "configs/scenarios/sc1_steady.yaml",
            "--policy",
            "client_driven",
            "--out",
            "/tmp/fa-rlwa-preflight-test",
            "--wifi-adapter",
            "wlan0",
        ],
    )
    checked = []
    started = []
    monkeypatch.setattr(
        runner,
        "check_wifi_scan",
        lambda interface, timeout_s: (
            checked.append((interface, timeout_s))
            or WifiScanCheck(interface, False, None, "permission denied")
        ),
    )
    monkeypatch.setattr(runner, "run_experiment", lambda *args, **kwargs: started.append(args))

    with pytest.raises(SystemExit) as error:
        runner.main()

    assert error.value.code == 2
    assert checked == [("wlan0", 15.0)]
    assert started == []
