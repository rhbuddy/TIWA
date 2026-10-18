"""Mininet-WiFi telemetry and association-control adapter.

The adapter targets Mininet-WiFi's Python API and Linux ``iw`` counters. It is kept
separate from the portable simulator so importing FA-RLWA does not require Mininet-WiFi.
"""

from __future__ import annotations

import math
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from fa_rlwa.config import ScenarioConfig
from fa_rlwa.types import (
    Action,
    APState,
    ChannelChange,
    Handover,
    IndicatorReport,
    NetworkState,
    STAState,
)


@dataclass(frozen=True)
class StationRadioStats:
    """Cumulative counters reported by ``iw station dump`` for one station."""

    mac: str
    signal_dbm: float | None = None
    tx_bytes: int = 0
    rx_bytes: int = 0
    tx_packets: int = 0
    tx_retries: int = 0
    tx_failed: int = 0
    tx_bitrate_mbps: float = 6.0
    rx_bitrate_mbps: float | None = None


@dataclass(frozen=True)
class SurveyStats:
    """Channel active/busy counters from ``iw survey dump``."""

    active_ms: int = 0
    busy_ms: int = 0


def parse_station_dump(output: str) -> dict[str, StationRadioStats]:
    """Parse station records from Linux ``iw dev <iface> station dump`` output."""

    result: dict[str, StationRadioStats] = {}
    current: dict[str, Any] | None = None

    def commit() -> None:
        if current is None or not current.get("mac"):
            return
        result[str(current["mac"]).lower()] = StationRadioStats(
            mac=str(current["mac"]).lower(),
            signal_dbm=current.get("signal_dbm"),
            tx_bytes=int(current.get("tx_bytes", 0)),
            rx_bytes=int(current.get("rx_bytes", 0)),
            tx_packets=int(current.get("tx_packets", 0)),
            tx_retries=int(current.get("tx_retries", 0)),
            tx_failed=int(current.get("tx_failed", 0)),
            tx_bitrate_mbps=float(current.get("tx_bitrate_mbps", 6.0)),
            rx_bitrate_mbps=current.get("rx_bitrate_mbps"),
        )

    for line in output.splitlines():
        station = re.match(r"\s*Station\s+([0-9a-fA-F:]{17})\b", line)
        if station:
            commit()
            current = {"mac": station.group(1)}
            continue
        if current is None or ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", maxsplit=1))
        if key in {"tx bytes", "rx bytes", "tx packets", "tx retries", "tx failed"}:
            parsed = re.match(r"\d+", value)
            if parsed:
                current[key.replace(" ", "_")] = int(parsed.group())
        elif key == "signal":
            parsed = re.match(r"-?\d+(?:\.\d+)?", value)
            if parsed:
                current["signal_dbm"] = float(parsed.group())
        elif key in {"tx bitrate", "rx bitrate"}:
            parsed = re.match(r"([\d.]+)\s*(MBit/s|KBit/s|GBit/s|Mbit/s|Kbit/s|Gbit/s)", value)
            if parsed:
                scale = {
                    "Kbit/s": 0.001,
                    "KBit/s": 0.001,
                    "MBit/s": 1.0,
                    "Mbit/s": 1.0,
                    "GBit/s": 1000.0,
                    "Gbit/s": 1000.0,
                }
                direction = key.split()[0]
                current[f"{direction}_bitrate_mbps"] = (
                    float(parsed.group(1)) * scale[parsed.group(2)]
                )
    commit()
    return result


def parse_survey_dump(output: str) -> SurveyStats:
    """Parse active and busy channel-time counters from ``iw survey dump`` output."""

    blocks = re.split(r"(?=^\s*frequency:)", output, flags=re.MULTILINE)
    candidates: list[SurveyStats] = []
    for block in blocks:
        active = re.search(r"(?m)^\s*channel active time:\s*(\d+)\s*ms", block)
        busy = re.search(r"(?m)^\s*channel busy time:\s*(\d+)\s*ms", block)
        if active and busy:
            candidates.append(SurveyStats(int(active.group(1)), int(busy.group(1))))
    if not candidates:
        return SurveyStats()
    in_use = [block for block in blocks if "[in use]" in block]
    if in_use:
        active = re.search(r"(?m)^\s*channel active time:\s*(\d+)\s*ms", in_use[0])
        busy = re.search(r"(?m)^\s*channel busy time:\s*(\d+)\s*ms", in_use[0])
        if active and busy:
            return SurveyStats(int(active.group(1)), int(busy.group(1)))
    return candidates[0]


def _wireless_interface(node: Any) -> Any:
    """Return a node's first Mininet-WiFi wireless interface."""

    interfaces = getattr(node, "wintfs", None)
    if isinstance(interfaces, dict):
        if not interfaces:
            raise RuntimeError(f"Mininet-WiFi node {getattr(node, 'name', node)!r} has no WLAN")
        return interfaces[min(interfaces)]
    if interfaces:
        return interfaces[0]
    raise RuntimeError(f"Mininet-WiFi node {getattr(node, 'name', node)!r} has no WLAN")


def _position(node: Any) -> tuple[float, float]:
    """Extract a Mininet-WiFi node's two-dimensional position in metres."""

    value = getattr(node, "position", None)
    if value is None:
        params = getattr(node, "params", {})
        value = params.get("position") if isinstance(params, dict) else None
    if isinstance(value, str):
        value = [float(part.strip()) for part in value.split(",")]
    if value is None or len(value) < 2:
        raise RuntimeError(f"Mininet-WiFi node {getattr(node, 'name', node)!r} has no position")
    return float(value[0]), float(value[1])


def _read_node(node: Any, command: str) -> str:
    """Run a read-only command in a Mininet node's namespace."""

    result = node.pexec(command)
    if isinstance(result, tuple):
        stdout = str(result[0])
        return_code = int(result[2]) if len(result) > 2 else 0
        return stdout if return_code == 0 else ""
    return str(result)


def _counter_delta(current: int, previous: int | None) -> int:
    """Return a non-negative counter delta, accounting for interface resets."""

    return current if previous is None or current < previous else current - previous


class MininetWifiBackend:
    """Translate a running Mininet-WiFi topology into FA-RLWA states and actions.

    The RSSI for the serving AP is read from ``iw``; candidate-AP RSSI is obtained from
    Mininet-WiFi's configured propagation model. Occupancy uses survey counters where
    available and falls back to estimated own-airtime when the driver lacks survey data.
    """

    def __init__(
        self,
        net: Any,
        scenario: ScenarioConfig,
        ap_nodes: list[Any],
        sta_nodes: list[Any],
        offered_mbps: list[float],
        traffic_types: list[str],
        *,
        allow_channel_changes: bool = False,
    ) -> None:
        if len(ap_nodes) != len(scenario.aps):
            raise ValueError("Mininet AP count must match scenario AP count")
        if not (len(sta_nodes) == len(offered_mbps) == len(traffic_types)):
            raise ValueError("each Mininet station requires a traffic profile")
        self.net = net
        self.scenario = scenario
        self.ap_nodes = list(ap_nodes)
        self.sta_nodes = list(sta_nodes)
        self.offered_mbps = list(map(float, offered_mbps))
        self.traffic_types = list(traffic_types)
        self.allow_channel_changes = bool(allow_channel_changes)
        self._ap_interfaces = [_wireless_interface(node) for node in self.ap_nodes]
        self._sta_interfaces = [_wireless_interface(node) for node in self.sta_nodes]
        self._channels = [int(ap.channel) for ap in scenario.aps]
        sta_macs = [str(interface.mac).lower() for interface in self._sta_interfaces]
        if len(set(sta_macs)) != len(self.sta_nodes):
            raise ValueError("Mininet station WLAN MAC addresses must be unique")
        self._tick = -1
        self._last_sample_time: float | None = None
        self._previous_station_counters: dict[tuple[int, str], StationRadioStats] = {}
        self._previous_survey: dict[int, SurveyStats] = {}
        self._previous_ap: dict[int, int] = {}
        self._last_handover_t = {sta_id: -10_000 for sta_id in range(len(self.sta_nodes))}
        self._previous_ap_for_sta = {sta_id: -1 for sta_id in range(len(self.sta_nodes))}
        self._last_channel_change_t = {ap_id: -10_000 for ap_id in range(len(self.ap_nodes))}
        self.handover_events: list[dict[str, object]] = []
        self.invalid_action_events: list[dict[str, object]] = []
        self._reports: deque[IndicatorReport] = deque(maxlen=300)
        self._latest_state: NetworkState | None = None
        self._closed = False

    def _read_ap_stats(self) -> tuple[list[dict[str, StationRadioStats]], list[SurveyStats]]:
        """Collect per-AP station and survey reports using in-namespace ``iw`` calls."""

        station_stats: list[dict[str, StationRadioStats]] = []
        surveys: list[SurveyStats] = []
        for ap_node, interface in zip(self.ap_nodes, self._ap_interfaces, strict=True):
            station_stats.append(
                parse_station_dump(_read_node(ap_node, f"iw dev {interface.name} station dump"))
            )
            surveys.append(
                parse_survey_dump(_read_node(ap_node, f"iw dev {interface.name} survey dump"))
            )
        return station_stats, surveys

    def _associated_ap(
        self,
        sta_id: int,
        station_stats: list[dict[str, StationRadioStats]],
    ) -> int:
        """Resolve serving AP from actual AP station tables, not Mininet's attempted link."""

        mac = str(self._sta_interfaces[sta_id].mac).lower()
        return next((ap_id for ap_id, stats in enumerate(station_stats) if mac in stats), -1)

    def snapshot(self, t: int | None = None, *, now: float | None = None) -> NetworkState:
        """Read a timed observation and append its AP report to the bounded history."""

        if self._closed:
            raise RuntimeError("Mininet-WiFi backend has been closed")
        self._tick = self._tick + 1 if t is None else int(t)
        sample_time = time.monotonic() if now is None else float(now)
        elapsed_s = (
            1.0
            if self._last_sample_time is None
            else max(sample_time - self._last_sample_time, 1e-6)
        )
        self._last_sample_time = sample_time
        station_stats_by_ap, surveys = self._read_ap_stats()
        n_sta, n_ap = len(self.sta_nodes), len(self.ap_nodes)
        positions = [_position(node) for node in self.sta_nodes]
        rssi = np.full((n_sta, n_ap), -100.0, dtype=float)
        sta_tput = np.zeros(n_sta, dtype=float)
        sta_phy = np.full(n_sta, 6.0, dtype=float)
        sta_loss = np.zeros(n_sta, dtype=float)
        sta_retx = np.zeros(n_sta, dtype=float)
        ap_counts = np.zeros(n_ap, dtype=int)
        ap_throughput = np.zeros(n_ap, dtype=float)
        ap_demand = np.zeros(n_ap, dtype=float)
        ap_loss_samples: list[list[float]] = [[] for _ in range(n_ap)]
        ap_rssi_samples: list[list[float]] = [[] for _ in range(n_ap)]
        ap_own_airtime = np.zeros(n_ap, dtype=float)
        sta_states: list[STAState] = []

        for sta_id, (_node, interface, position) in enumerate(
            zip(self.sta_nodes, self._sta_interfaces, positions, strict=True)
        ):
            current_ap = self._associated_ap(sta_id, station_stats_by_ap)
            mac = str(interface.mac).lower()
            current_stats = (
                station_stats_by_ap[current_ap].get(mac) if current_ap >= 0 else None
            )
            for ap_id, (ap_node, ap_interface) in enumerate(
                zip(self.ap_nodes, self._ap_interfaces, strict=True)
            ):
                dx, dy = position[0] - _position(ap_node)[0], position[1] - _position(ap_node)[1]
                distance = max(math.hypot(dx, dy), 0.01)
                try:
                    rssi[sta_id, ap_id] = float(interface.get_rssi(ap_interface, distance))
                except (AttributeError, TypeError, ValueError, RuntimeError):
                    rssi[sta_id, ap_id] = -100.0
            if current_ap >= 0:
                ap_counts[current_ap] += 1
                if current_stats is not None and current_stats.signal_dbm is not None:
                    rssi[sta_id, current_ap] = current_stats.signal_dbm
                ap_rssi_samples[current_ap].append(float(rssi[sta_id, current_ap]))

            if current_stats is not None and current_ap >= 0:
                key = (current_ap, mac)
                previous = self._previous_station_counters.get(key)
                counters = (
                    current_stats.tx_bytes + current_stats.rx_bytes,
                    current_stats.tx_packets,
                    current_stats.tx_retries,
                    current_stats.tx_failed,
                )
                previous_values = (
                    None
                    if previous is None
                    else (
                        previous.tx_bytes + previous.rx_bytes,
                        previous.tx_packets,
                        previous.tx_retries,
                        previous.tx_failed,
                    )
                )
                deltas = (
                    [0, 0, 0, 0]
                    if previous_values is None
                    else [
                        _counter_delta(value, old)
                        for value, old in zip(counters, previous_values, strict=True)
                    ]
                )
                byte_delta, packet_delta, retry_delta, failed_delta = deltas
                sta_tput[sta_id] = byte_delta * 8.0 / elapsed_s / 1_000_000.0
                sta_phy[sta_id] = max(
                    current_stats.rx_bitrate_mbps or current_stats.tx_bitrate_mbps, 0.1
                )
                sta_loss[sta_id] = failed_delta / max(packet_delta + failed_delta, 1)
                sta_retx[sta_id] = retry_delta / max(packet_delta, 1)
                ap_throughput[current_ap] += sta_tput[sta_id]
                ap_demand[current_ap] += self.offered_mbps[sta_id]
                ap_loss_samples[current_ap].append(sta_loss[sta_id])
                ap_own_airtime[current_ap] += sta_tput[sta_id] / sta_phy[sta_id]
                self._previous_station_counters[key] = current_stats

            previous_ap = self._previous_ap.get(sta_id)
            if previous_ap is not None and previous_ap != current_ap:
                self._previous_ap_for_sta[sta_id] = previous_ap
                if not any(
                    event["sta_id"] == sta_id
                    and event["to_ap"] == current_ap
                    and event["t"] == self._tick
                    for event in self.handover_events
                ):
                    self.handover_events.append(
                        {
                            "t": self._tick,
                            "sta_id": sta_id,
                            "from_ap": previous_ap,
                            "to_ap": current_ap,
                            "reason": "external_association",
                        }
                    )
                    self._last_handover_t[sta_id] = self._tick
            self._previous_ap[sta_id] = current_ap
            previous_for_sta = self._previous_ap_for_sta[sta_id]
            if previous_ap != current_ap:
                previous_for_sta = previous_ap if previous_ap is not None else -1
            sta_states.append(
                STAState(
                    sta_id=sta_id,
                    ap_id=current_ap,
                    pos=position,
                    traffic_type=self.traffic_types[sta_id],
                    offered_mbps=self.offered_mbps[sta_id],
                    base_offered_mbps=self.offered_mbps[sta_id],
                    last_handover_t=self._last_handover_t[sta_id],
                    previous_ap_id=previous_for_sta,
                )
            )

        ap_states: list[APState] = []
        occupancy = np.zeros(n_ap, dtype=float)
        for ap_id, (config, ap_node, survey) in enumerate(
            zip(self.scenario.aps, self.ap_nodes, surveys, strict=True)
        ):
            previous_survey = self._previous_survey.get(ap_id)
            previous_active = previous_survey.active_ms if previous_survey else None
            previous_busy = previous_survey.busy_ms if previous_survey else None
            active_delta = _counter_delta(survey.active_ms, previous_active)
            busy_delta = _counter_delta(survey.busy_ms, previous_busy)
            if active_delta > 0:
                occupancy[ap_id] = float(np.clip(busy_delta / active_delta, 0.0, 1.0))
            elif survey.active_ms > 0:
                occupancy[ap_id] = float(np.clip(survey.busy_ms / survey.active_ms, 0.0, 1.0))
            else:
                occupancy[ap_id] = float(np.clip(ap_own_airtime[ap_id], 0.0, 1.0))
            own_airtime = float(np.clip(ap_own_airtime[ap_id], 0.0, 1.0))
            if not np.isfinite(occupancy[ap_id]):
                occupancy[ap_id] = own_airtime
            self._previous_survey[ap_id] = survey
            ap_states.append(
                APState(
                    ap_id=ap_id,
                    channel=self._channels[ap_id],
                    pos=_position(ap_node),
                    tx_dbm=float(config.tx_dbm),
                    last_channel_change_t=self._last_channel_change_t[ap_id],
                )
            )

        report = IndicatorReport(
            t=self._tick,
            rssi_dbm=np.asarray(
                [float(np.mean(values)) if values else -60.0 for values in ap_rssi_samples]
            ),
            loss_frac=np.asarray(
                [float(np.mean(values)) if values else 0.0 for values in ap_loss_samples]
            ),
            tput_mbps=ap_throughput.copy(),
            occ_frac=occupancy.copy(),
            n_clients=ap_counts.copy(),
            airtime_own_frac=np.asarray(ap_own_airtime, dtype=float).clip(0.0, 1.0),
            demand_mbps=ap_demand.copy(),
        )
        self._reports.append(report)
        state = NetworkState(
            t=self._tick,
            aps=ap_states,
            stas=sta_states,
            rssi_dbm=rssi,
            reports=list(self._reports),
            sta_tput_mbps=sta_tput,
            sta_phy_mbps=sta_phy,
            sta_loss_frac=sta_loss,
            sta_retx_per_frame=sta_retx,
            sta_demand_mbps=np.asarray(self.offered_mbps, dtype=float),
        )
        self._latest_state = state
        return state

    def apply_actions(self, actions: list[Action]) -> list[dict[str, object]]:
        """Execute validated handovers; channel changes require an explicit opt-in."""

        if self._closed:
            raise RuntimeError("Mininet-WiFi backend has been closed")
        if self._latest_state is None:
            raise RuntimeError("Call snapshot() before applying policy actions")
        accepted: list[dict[str, object]] = []
        moved: set[int] = set()
        for action in actions:
            if isinstance(action, Handover):
                reason = self._validate_handover(action, moved)
                if reason:
                    self.invalid_action_events.append(
                        {
                            "t": self._latest_state.t,
                            "sta_id": action.sta_id,
                            "target_ap": action.target_ap,
                            "reason": reason,
                        }
                    )
                    continue
                sta_node = self.sta_nodes[action.sta_id]
                source_ap = self._latest_state.stas[action.sta_id].ap_id
                try:
                    sta_node.setAssociation(self.ap_nodes[action.target_ap])
                except (AttributeError, RuntimeError, ValueError) as exc:
                    reason = f"association_error:{type(exc).__name__}"
                    self.invalid_action_events.append(
                        {
                            "t": self._latest_state.t,
                            "sta_id": action.sta_id,
                            "target_ap": action.target_ap,
                            "reason": reason,
                        }
                    )
                    continue
                current = getattr(self._sta_interfaces[action.sta_id], "associatedTo", None)
                target_node = getattr(current, "node", None)
                target_interface = self._ap_interfaces[action.target_ap]
                observed_stations = parse_station_dump(
                    _read_node(
                        self.ap_nodes[action.target_ap],
                        f"iw dev {target_interface.name} station dump",
                    )
                )
                sta_mac = str(self._sta_interfaces[action.sta_id].mac).lower()
                if (
                    target_node is not self.ap_nodes[action.target_ap]
                    or sta_mac not in observed_stations
                ):
                    self.invalid_action_events.append(
                        {
                            "t": self._latest_state.t,
                            "sta_id": action.sta_id,
                            "target_ap": action.target_ap,
                            "reason": "association_not_confirmed",
                        }
                    )
                    continue
                moved.add(action.sta_id)
                self._last_handover_t[action.sta_id] = self._latest_state.t
                self._previous_ap_for_sta[action.sta_id] = source_ap
                self._previous_ap[action.sta_id] = action.target_ap
                event = {
                    "t": self._latest_state.t,
                    "sta_id": action.sta_id,
                    "from_ap": source_ap,
                    "to_ap": action.target_ap,
                    "reason": action.reason,
                }
                self.handover_events.append(event)
                accepted.append(event)
            elif isinstance(action, ChannelChange):
                reason = self._validate_channel_change(action)
                if reason:
                    self.invalid_action_events.append(
                        {
                            "t": self._latest_state.t,
                            "ap_id": action.ap_id,
                            "channel": action.channel,
                            "reason": reason,
                        }
                    )
                    continue
                interface = self._ap_interfaces[action.ap_id]
                try:
                    interface.setAPChannel(int(action.channel))
                except (AttributeError, RuntimeError, ValueError) as exc:
                    self.invalid_action_events.append(
                        {
                            "t": self._latest_state.t,
                            "ap_id": action.ap_id,
                            "channel": action.channel,
                            "reason": f"channel_change_error:{type(exc).__name__}",
                        }
                    )
                    continue
                self._last_channel_change_t[action.ap_id] = self._latest_state.t
                self._channels[action.ap_id] = int(action.channel)
                accepted.append(
                    {
                        "t": self._latest_state.t,
                        "ap_id": action.ap_id,
                        "channel": action.channel,
                        "reason": action.reason,
                    }
                )
        return accepted

    def _validate_handover(self, action: Handover, moved: set[int]) -> str | None:
        """Apply range, dwell, and per-cycle safety guards to one requested handover."""

        state = self._latest_state
        assert state is not None
        if action.sta_id < 0 or action.sta_id >= len(self.sta_nodes):
            return "unknown_station"
        if action.target_ap < 0 or action.target_ap >= len(self.ap_nodes):
            return "unknown_access_point"
        if action.sta_id in moved:
            return "multiple_moves_same_cycle"
        station = state.stas[action.sta_id]
        if station.ap_id == action.target_ap:
            return "already_associated"
        if state.t - self._last_handover_t[action.sta_id] < self.scenario.handover.dwell_s:
            return "dwell_time"
        row = action.sta_id
        if (
            state.rssi_dbm[row, action.target_ap]
            < self.scenario.propagation.association_threshold_dbm
        ):
            return "out_of_range"
        if len(moved) >= self.scenario.handover.max_moves_per_cycle:
            return "max_moves_per_cycle"
        return None

    def _validate_channel_change(self, action: ChannelChange) -> str | None:
        """Validate channel requests and enforce the explicit safety opt-in."""

        state = self._latest_state
        assert state is not None
        if not self.allow_channel_changes:
            return "channel_changes_disabled"
        if action.ap_id < 0 or action.ap_id >= len(self.ap_nodes):
            return "unknown_access_point"
        if action.channel not in self.scenario.channels:
            return "unsupported_channel"
        if (
            state.t - self._last_channel_change_t[action.ap_id]
            < self.scenario.handover.channel_change_dwell_s
        ):
            return "channel_dwell_time"
        if action.channel == state.aps[action.ap_id].channel:
            return "already_on_channel"
        return None

    def close(self, *, stop_network: bool = False) -> None:
        """Optionally stop the Mininet-WiFi network; safe to call more than once."""

        if self._closed:
            return
        self._closed = True
        if stop_network:
            self.net.stop()
