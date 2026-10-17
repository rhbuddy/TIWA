"""Unit tests for Mininet-WiFi telemetry parsing and guarded actions."""

from __future__ import annotations

from fa_rlwa.config import scenario_from_mapping
from fa_rlwa.types import Handover
from fa_rlwa.validation.mininet_wifi.backend import (
    MininetWifiBackend,
    parse_station_dump,
    parse_survey_dump,
)


class FakeAPInterface:
    def __init__(self, node: FakeAP, ap_id: int) -> None:
        self.node = node
        self.name = f"{node.name}-wlan1"
        self.ap_id = ap_id
        self.mac = f"02:00:00:00:00:{ap_id + 1:02x}"


class FakeStationInterface:
    def __init__(self, node: FakeStation, mac: str) -> None:
        self.node = node
        self.name = f"{node.name}-wlan0"
        self.mac = mac
        self.associatedTo: FakeAPInterface | None = None

    def get_rssi(self, ap_interface: FakeAPInterface, distance: float) -> float:
        return -50.0 - (ap_interface.ap_id * 5.0) - min(distance, 20.0) / 10.0


class FakeAP:
    def __init__(self, ap_id: int, position: tuple[float, float]) -> None:
        self.name = f"ap{ap_id}"
        self.position = (*position, 0.0)
        self.wintfs = {0: FakeAPInterface(self, ap_id)}
        self.stations: list[FakeStation] = []
        self.active_ms = 1000
        self.busy_ms = 250 + ap_id * 100

    def pexec(self, command: str) -> tuple[str, str, int]:
        if "station dump" in command:
            reports = []
            for station in self.stations:
                interface = station.wintfs[0]
                if interface.associatedTo is not self.wintfs[0]:
                    continue
                counters = station.counters
                reports.append(
                    f"Station {interface.mac}\n"
                    f"\tsignal: {counters['signal']} dBm\n"
                    f"\ttx bytes: {counters['tx_bytes']}\n"
                    f"\trx bytes: {counters['rx_bytes']}\n"
                    f"\ttx packets: {counters['tx_packets']}\n"
                    f"\ttx retries: {counters['tx_retries']}\n"
                    f"\ttx failed: {counters['tx_failed']}\n"
                    "\ttx bitrate: 54.0 MBit/s\n"
                )
            return "".join(reports), "", 0
        return (
            "Survey data from ap\n"
            " frequency: 2412 MHz [in use]\n"
            f" channel active time: {self.active_ms} ms\n"
            f" channel busy time: {self.busy_ms} ms\n",
            "",
            0,
        )


class FakeStation:
    def __init__(self, sta_id: int, position: tuple[float, float]) -> None:
        self.name = f"sta{sta_id}"
        self.position = (*position, 0.0)
        self.counters: dict[str, int] = {
            "signal": -55,
            "tx_bytes": 0,
            "rx_bytes": 0,
            "tx_packets": 0,
            "tx_retries": 0,
            "tx_failed": 0,
        }
        self.wintfs = {
            0: FakeStationInterface(self, f"02:00:00:01:00:{sta_id + 1:02x}")
        }
        self.association_calls: list[FakeAP] = []

    def setAssociation(self, ap: FakeAP) -> None:
        self.association_calls.append(ap)
        self.wintfs[0].associatedTo = ap.wintfs[0]


def _scenario():
    return scenario_from_mapping(
        {
            "name": "mininet_test",
            "area_m": [10, 10],
            "duration_s": 30,
            "warmup_s": 0,
            "channels": [1, 6, 11],
            "aps": [
                {"id": 0, "pos": [0, 5], "channel": 1},
                {"id": 1, "pos": [10, 5], "channel": 6},
            ],
            "background_stas": {"count": 1, "traffic": {"type": "cbr", "mbps": 1.0}},
            "handover": {"dwell_s": 10, "max_moves_per_cycle": 1},
        }
    )


def _backend() -> tuple[MininetWifiBackend, list[FakeAP], list[FakeStation]]:
    aps = [FakeAP(0, (0.0, 5.0)), FakeAP(1, (10.0, 5.0))]
    stations = [FakeStation(0, (5.0, 5.0))]
    for ap in aps:
        ap.stations = stations
    stations[0].wintfs[0].associatedTo = aps[0].wintfs[0]
    backend = MininetWifiBackend(
        net=object(),
        scenario=_scenario(),
        ap_nodes=aps,
        sta_nodes=stations,
        offered_mbps=[1.0],
        traffic_types=["cbr"],
    )
    return backend, aps, stations


def test_iw_parsers_capture_station_counters_and_survey_occupancy() -> None:
    parsed = parse_station_dump(
        "Station 02:00:00:00:00:01 (on wlan1)\n"
        "\tsignal: -57 dBm\n"
        "\ttx bytes: 1200\n"
        "\trx bytes: 2400\n"
        "\ttx packets: 10\n"
        "\ttx retries: 2\n"
        "\ttx failed: 1\n"
        "\ttx bitrate: 6.5 MBit/s\n"
    )
    stats = parsed["02:00:00:00:00:01"]
    assert stats.signal_dbm == -57.0
    assert stats.tx_bytes == 1200 and stats.rx_bytes == 2400
    assert stats.tx_retries == 2 and stats.tx_failed == 1
    assert stats.tx_bitrate_mbps == 6.5
    assert stats.rx_bitrate_mbps is None

    survey = parse_survey_dump(
        "Survey data from wlan1\n"
        " frequency: 2412 MHz\n"
        " channel active time: 900 ms\n"
        " channel busy time: 300 ms\n"
        " frequency: 2437 MHz [in use]\n"
        " channel active time: 1000 ms\n"
        " channel busy time: 450 ms\n"
    )
    assert survey.active_ms == 1000 and survey.busy_ms == 450


def test_snapshot_builds_state_and_estimates_counter_delta_throughput() -> None:
    backend, aps, stations = _backend()
    first = backend.snapshot(t=0, now=10.0)
    assert first.rssi_dbm.shape == (1, 2)
    assert first.stas[0].ap_id == 0
    assert first.reports[-1].n_clients.tolist() == [1, 0]
    assert first.reports[-1].occ_frac.tolist() == [0.25, 0.35]

    stations[0].counters.update(tx_bytes=62_500, rx_bytes=62_500, tx_packets=100)
    second = backend.snapshot(t=1, now=11.0)
    assert second.sta_tput_mbps[0] == 1.0
    assert second.reports[-1].tput_mbps[0] == 1.0
    assert second.reports[-1].demand_mbps[0] == 1.0
    backend.close()
    assert aps  # Keep the fake AP objects live through both snapshots.


def test_handover_executes_once_and_dwell_guard_rejects_ping_pong() -> None:
    backend, _, stations = _backend()
    backend.snapshot(t=0, now=1.0)
    accepted = backend.apply_actions([Handover(sta_id=0, target_ap=1, reason="test")])
    assert len(accepted) == 1
    assert stations[0].association_calls[-1].name == "ap1"

    after = backend.snapshot(t=1, now=2.0)
    assert after.stas[0].ap_id == 1
    assert after.stas[0].last_handover_t == 0
    assert len(backend.handover_events) == 1
    backend.apply_actions([Handover(sta_id=0, target_ap=0)])
    assert backend.invalid_action_events[-1]["reason"] == "dwell_time"
    backend.close()
