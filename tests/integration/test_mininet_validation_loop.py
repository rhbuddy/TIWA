"""A fake Mininet topology exercises the live controller/trace lifecycle without root."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from fa_rlwa.validation.mininet_wifi import runner as mininet_runner
from fa_rlwa.validation.mininet_wifi.runner import run_mininet_validation


class _APInterface:
    def __init__(self, node: _AP, ap_id: int) -> None:
        self.node = node
        self.name = f"{node.name}-wlan1"
        self.ap_id = ap_id
        self.mac = f"02:00:00:00:00:{ap_id + 1:02x}"


class _STAInterface:
    def __init__(self, node: _Station, sta_id: int) -> None:
        self.node = node
        self.name = f"{node.name}-wlan0"
        self.mac = f"02:00:00:01:00:{sta_id + 1:02x}"
        self.associatedTo = None

    def get_rssi(self, ap_interface: _APInterface, distance: float) -> float:
        return -50.0 - ap_interface.ap_id * 5.0 - min(distance, 20.0) / 10.0


class _AP:
    def __init__(self, name: str, position: str, ap_id: int) -> None:
        self.name = name
        self.position = tuple(float(part) for part in position.split(","))
        self.wintfs = {0: _APInterface(self, ap_id)}
        self.stations: list[_Station] = []

    def start(self, controllers: list[Any]) -> None:
        pass

    def pexec(self, command: str) -> tuple[str, str, int]:
        if "survey dump" in command:
            return (
                "Survey data\n frequency: 2412 MHz [in use]\n"
                " channel active time: 1000 ms\n channel busy time: 200 ms\n",
                "",
                0,
            )
        return "", "", 0


class _Station:
    def __init__(self, name: str, position: str, sta_id: int) -> None:
        self.name = name
        self.position = tuple(float(part) for part in position.split(","))
        self.wintfs = {0: _STAInterface(self, sta_id)}

    def setAssociation(self, ap: _AP) -> None:
        self.wintfs[0].associatedTo = ap.wintfs[0]


class _Switch:
    def start(self, controllers: list[Any]) -> None:
        pass


class _Server:
    def IP(self) -> str:
        return "10.0.0.254"


class _FakeMininet:
    def __init__(self, **options: Any) -> None:
        self.aps: list[_AP] = []
        self.stations: list[_Station] = []
        self.stopped = False

    def addAccessPoint(self, name: str, **options: Any) -> _AP:
        node = _AP(name, options["position"], len(self.aps))
        node.stations = self.stations
        self.aps.append(node)
        return node

    def addStation(self, name: str, **options: Any) -> _Station:
        node = _Station(name, options["position"], len(self.stations))
        self.stations.append(node)
        for ap in self.aps:
            ap.stations = self.stations
        return node

    def setPropagationModel(self, **options: Any) -> None:
        pass

    def addSwitch(self, name: str, **options: Any) -> _Switch:
        return _Switch()

    def addHost(self, name: str, **options: Any) -> _Server:
        return _Server()

    def configureNodes(self) -> None:
        pass

    def addLink(self, node1: Any, node2: Any) -> None:
        pass

    def build(self) -> None:
        pass

    def stop(self) -> None:
        self.stopped = True


def _factory_recorder() -> tuple[Any, list[_FakeMininet]]:
    networks: list[_FakeMininet] = []

    def factory(**options: Any) -> _FakeMininet:
        network = _FakeMininet(**options)
        networks.append(network)
        return network

    return factory, networks


def test_live_loop_writes_emulator_traces_and_stops_network(tmp_path: Path) -> None:
    factory, networks = _factory_recorder()
    result = run_mininet_validation(
        "configs/scenarios/sc1_steady.yaml",
        "client_driven",
        3,
        tmp_path / "mininet_run",
        duration_s=2,
        tick_s=0.001,
        generate_traffic=False,
        make_plots=False,
        _mininet_factory=factory,
    )

    output = tmp_path / "mininet_run"
    assert result["backend"] == "mininet_wifi"
    assert result["completed_cycles"] == 2
    assert networks[0].stopped
    for name in (
        "ap_indicators.csv",
        "sta_log.csv",
        "handover_log.csv",
        "invalid_actions.csv",
        "metrics_summary.csv",
        "config_resolved.yaml",
    ):
        assert (output / name).is_file()
    with (output / "ap_indicators.csv").open(encoding="utf-8", newline="") as stream:
        ap_rows = list(csv.DictReader(stream))
    assert len(ap_rows) == 3 * 3  # Three APs at t=0, 1, and 2.


def test_forever_loop_stops_cleanly_on_keyboard_interrupt(monkeypatch, tmp_path: Path) -> None:
    factory, networks = _factory_recorder()
    def interrupt(_delay: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(mininet_runner.time, "sleep", interrupt)
    result = run_mininet_validation(
        "configs/scenarios/sc1_steady.yaml",
        "client_driven",
        3,
        tmp_path / "mininet_forever",
        forever=True,
        tick_s=0.001,
        generate_traffic=False,
        make_plots=False,
        _mininet_factory=factory,
    )
    assert result["run_until_interrupted"] is True
    assert result["completed_cycles"] == 0
    assert networks[0].stopped
    assert (tmp_path / "mininet_forever" / "config_resolved.yaml").is_file()
