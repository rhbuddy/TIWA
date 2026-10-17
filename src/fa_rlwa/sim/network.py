"""Fast, seedable one-second Wi-Fi network simulator."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import numpy as np

from fa_rlwa.config import ScenarioConfig, StationGroupConfig
from fa_rlwa.monitor.indicators import build_indicator_report, satisfaction_fairness
from fa_rlwa.monitor.store import IndicatorStore
from fa_rlwa.sim.mac import MacResult, compute_mac_metrics
from fa_rlwa.sim.mobility import advance_position
from fa_rlwa.sim.propagation import path_loss_db
from fa_rlwa.sim.traffic import update_offered_load
from fa_rlwa.types import (
    Action,
    APState,
    ChannelChange,
    Handover,
    NetworkState,
    STAState,
)


class WifiNetwork:
    """Abstract SDN Wi-Fi network with airtime sharing, mobility, and handovers."""

    def __init__(self, scenario: ScenarioConfig, seed: int = 0) -> None:
        self.scenario = scenario
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.t = 0
        self.aps: list[APState] = []
        self.stas: list[STAState] = []
        self.store = IndicatorStore(max_reports=300)
        self.state: NetworkState | None = None
        self.last_step_metrics: dict[str, Any] = {}
        self.handover_events: list[dict[str, object]] = []
        self.channel_events: list[dict[str, object]] = []
        self.invalid_action_events: list[dict[str, object]] = []
        self._rssi_matrix = np.empty((0, 0), dtype=float)
        self._last_mac: MacResult | None = None
        self._last_measured_tput = np.empty(0, dtype=float)
        self._next_sta_id = 0
        self._flash_arrivals: list[tuple[int, int]] = []
        self._outage_remaining: dict[int, float] = {}
        self._channel_outage_remaining: dict[int, float] = {}
        self._current_handover_ids: set[int] = set()
        self._current_channel_change_ids: set[int] = set()
        self.reset(seed=self.seed)

    def reset(self, seed: int | None = None) -> NetworkState:
        """Reset the simulation; identical configuration and seed reproduce the same trace."""

        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.t = 0
        self.aps = [APState(ap.ap_id, ap.channel, ap.pos, ap.tx_dbm) for ap in self.scenario.aps]
        self.stas = []
        self.store = IndicatorStore(max_reports=300)
        self.handover_events = []
        self.channel_events = []
        self.invalid_action_events = []
        self._next_sta_id = 0
        self._outage_remaining = {}
        self._channel_outage_remaining = {}
        self._current_handover_ids = set()
        self._current_channel_change_ids = set()
        self._flash_arrivals = []

        for group in self.scenario.groups:
            for _ in range(group.count):
                self.stas.append(self._spawn_sta(group, self.t))
        flash = self.scenario.flash_crowd
        if flash is not None and flash.count:
            times = self.rng.integers(
                flash.start_s,
                flash.start_s + max(flash.window_s, 1),
                size=flash.count,
            )
            self._flash_arrivals = sorted((int(t), index) for index, t in enumerate(times))

        self._refresh_snapshot(added_or_moved=set())
        self.last_step_metrics = self._make_step_metrics(set(), 0, 0)
        return self.state  # type: ignore[return-value]

    def step(self, actions: Iterable[Action] = ()) -> NetworkState:
        """Apply controller requests and advance the simulated network by one second."""

        next_t = self.t + self.scenario.step_s
        self._current_handover_ids = set()
        self._current_channel_change_ids = set()
        invalid_before = len(self.invalid_action_events)
        self._apply_actions(list(actions), next_t)
        self.t = next_t

        for sta in list(self.stas):
            if sta.departure_t is not None and self.t >= sta.departure_t:
                self.stas.remove(sta)
                self._outage_remaining.pop(sta.sta_id, None)
                continue
            advance_position(sta, self.rng, self.scenario.area_m)
            update_offered_load(sta, self.rng)
        self._spawn_flash_arrivals()
        self._update_shadowing()
        self._refresh_snapshot(added_or_moved=self._current_handover_ids)
        self.last_step_metrics = self._make_step_metrics(
            self._current_handover_ids,
            sum(1 for event in self.handover_events if event.get("t") == self.t),
            len(self.invalid_action_events) - invalid_before,
        )
        return self.state  # type: ignore[return-value]

    def _spawn_sta(
        self,
        group: StationGroupConfig,
        arrival_t: int,
        position: tuple[float, float] | None = None,
        departure_t: int | None = None,
    ) -> STAState:
        """Create an STA using this network's seeded random generator."""

        if position is None:
            if group.placement in {"hotspot", "per_ap", "overlap"}:
                if group.center_m is not None:
                    center = group.center_m
                elif group.placement == "overlap":
                    center_array = np.mean(np.asarray([ap.pos for ap in self.aps]), axis=0)
                    center = (float(center_array[0]), float(center_array[1]))
                else:
                    center = self.aps[self._next_sta_id % len(self.aps)].pos
                x = float(self.rng.normal(center[0], group.spread_m))
                y = float(self.rng.normal(center[1], group.spread_m))
                position = (
                    float(np.clip(x, 0.0, self.scenario.area_m[0])),
                    float(np.clip(y, 0.0, self.scenario.area_m[1])),
                )
            else:
                position = (
                    float(self.rng.uniform(0.0, self.scenario.area_m[0])),
                    float(self.rng.uniform(0.0, self.scenario.area_m[1])),
                )
        sta_id = self._next_sta_id
        self._next_sta_id += 1
        sigma = self.scenario.propagation.shadow_sigma_db
        sta = STAState(
            sta_id=sta_id,
            ap_id=-1,
            pos=position,
            traffic_type=group.traffic_type,
            offered_mbps=group.offered_mbps,
            base_offered_mbps=group.offered_mbps,
            mobility=group.mobility,
            arrival_t=arrival_t,
            departure_t=departure_t,
            speed_mps=float(self.rng.uniform(*group.speed_mps)),
            path=group.scripted_path,
            shadow_db=self.rng.normal(0.0, sigma, size=len(self.aps)),
        )
        sta.ap_id = self._strongest_reachable_ap(sta)
        if sta.mobility == "scripted" and sta.path:
            sta.path_index = 0
            sta.path_hold_s = 0
        return sta

    def _spawn_flash_arrivals(self) -> None:
        """Add scheduled burst clients and sample their exponential dwell times."""

        flash = self.scenario.flash_crowd
        if flash is None:
            return
        while self._flash_arrivals and self._flash_arrivals[0][0] <= self.t:
            arrival_t, _ = self._flash_arrivals.pop(0)
            x = float(self.rng.normal(flash.hotspot[0], flash.spread_m))
            y = float(self.rng.normal(flash.hotspot[1], flash.spread_m))
            position = (
                float(np.clip(x, 0.0, self.scenario.area_m[0])),
                float(np.clip(y, 0.0, self.scenario.area_m[1])),
            )
            dwell = max(1, int(self.rng.exponential(flash.dwell_mean_s)))
            group = StationGroupConfig(
                count=1,
                placement="uniform",
                mobility="static",
                traffic_type=flash.traffic_type,
                offered_mbps=flash.offered_mbps,
            )
            self.stas.append(self._spawn_sta(group, arrival_t, position, self.t + dwell))

    def _update_shadowing(self) -> None:
        """Advance each STA/AP shadowing value as a temporally correlated AR(1) process."""

        prop = self.scenario.propagation
        innovation_scale = math.sqrt(max(0.0, 1.0 - prop.shadow_rho**2)) * prop.shadow_sigma_db
        for sta in self.stas:
            eps = self.rng.normal(0.0, 1.0, size=len(self.aps))
            sta.shadow_db = prop.shadow_rho * sta.shadow_db + innovation_scale * eps

    def _rssi_for_position(self, pos: tuple[float, float], shadow_db: np.ndarray) -> np.ndarray:
        """Return physical RSSI from one STA to every AP, before report noise."""

        distances = np.asarray(
            [np.hypot(pos[0] - ap.pos[0], pos[1] - ap.pos[1]) for ap in self.aps], dtype=float
        )
        loss = path_loss_db(
            distances,
            self.scenario.propagation.pl0_db,
            self.scenario.propagation.exponent,
            shadow_db,
        )
        return 15.0 - np.asarray(loss, dtype=float)

    def _strongest_reachable_ap(self, sta: STAState) -> int:
        """Choose the AP with strongest uplink RSSI, or -1 when no AP is in range."""

        values = self._rssi_for_position(sta.pos, sta.shadow_db)
        reachable = values >= self.scenario.propagation.association_threshold_dbm
        return int(np.argmax(np.where(reachable, values, -np.inf))) if np.any(reachable) else -1

    def _apply_actions(self, actions: list[Action], at_t: int) -> None:
        """Validate and execute handover/channel requests with dwell-time enforcement."""

        accepted_moves = 0
        for action in actions:
            if isinstance(action, Handover):
                sta = next((item for item in self.stas if item.sta_id == action.sta_id), None)
                if sta is None:
                    self.invalid_action_events.append(
                        {
                            "t": at_t,
                            "sta_id": action.sta_id,
                            "target_ap": action.target_ap,
                            "reason": "unknown_sta",
                        }
                    )
                    continue
                invalid_reason: str | None = None
                if not 0 <= action.target_ap < len(self.aps):
                    invalid_reason = "invalid_ap"
                elif action.target_ap == sta.ap_id:
                    continue
                elif at_t - sta.last_handover_t < self.scenario.handover.dwell_s:
                    invalid_reason = "dwell_time"
                elif (
                    self._rssi_matrix.size
                    and self._rssi_matrix[
                        next(i for i, item in enumerate(self.stas) if item.sta_id == sta.sta_id),
                        action.target_ap,
                    ]
                    < self.scenario.propagation.association_threshold_dbm
                ):
                    invalid_reason = "out_of_range"
                elif accepted_moves >= self.scenario.handover.max_moves_per_cycle:
                    invalid_reason = "move_limit"
                if invalid_reason:
                    self.invalid_action_events.append(
                        {
                            "t": at_t,
                            "sta_id": action.sta_id,
                            "target_ap": action.target_ap,
                            "reason": invalid_reason,
                        }
                    )
                    continue
                source = sta.ap_id
                sta.previous_ap_id = source
                sta.ap_id = action.target_ap
                sta.last_handover_t = at_t
                self._current_handover_ids.add(sta.sta_id)
                self._outage_remaining[sta.sta_id] = float(self.scenario.handover.outage_s)
                self.handover_events.append(
                    {
                        "t": at_t,
                        "sta_id": sta.sta_id,
                        "from_ap": source,
                        "to_ap": action.target_ap,
                        "reason": action.reason,
                    }
                )
                accepted_moves += 1
            elif isinstance(action, ChannelChange):
                if (
                    not 0 <= action.ap_id < len(self.aps)
                    or action.channel not in self.scenario.channels
                ):
                    self.invalid_action_events.append(
                        {
                            "t": at_t,
                            "ap_id": action.ap_id,
                            "channel": action.channel,
                            "reason": "invalid_channel_change",
                        }
                    )
                    continue
                ap = self.aps[action.ap_id]
                if action.channel == ap.channel:
                    continue
                if at_t - ap.last_channel_change_t < self.scenario.handover.channel_change_dwell_s:
                    self.invalid_action_events.append(
                        {
                            "t": at_t,
                            "ap_id": action.ap_id,
                            "channel": action.channel,
                            "reason": "channel_dwell_time",
                        }
                    )
                    continue
                old_channel = ap.channel
                ap.channel = action.channel
                ap.last_channel_change_t = at_t
                self._current_channel_change_ids.add(ap.ap_id)
                self._channel_outage_remaining[ap.ap_id] = 0.5
                self.channel_events.append(
                    {
                        "t": at_t,
                        "ap_id": ap.ap_id,
                        "from_channel": old_channel,
                        "to_channel": action.channel,
                        "reason": action.reason,
                    }
                )
            else:
                raise TypeError(f"Unsupported action type: {type(action).__name__}")

    def _refresh_snapshot(self, added_or_moved: set[int]) -> None:
        """Recompute RSSI, PHY/MAC metrics, indicator reports, and public state."""

        # Disconnected clients automatically use the strongest reachable AP; stale associations
        # disappear on link loss, matching the client-driven fallback in the SRS.
        for sta in self.stas:
            values = self._rssi_for_position(sta.pos, sta.shadow_db)
            if (
                sta.ap_id < 0
                or values[sta.ap_id] < self.scenario.propagation.association_threshold_dbm
            ):
                sta.ap_id = self._strongest_reachable_ap(sta)

        physical = (
            np.stack([self._rssi_for_position(sta.pos, sta.shadow_db) for sta in self.stas], axis=0)
            if self.stas
            else np.empty((0, len(self.aps)), dtype=float)
        )
        if self.scenario.measurement_noise and self.scenario.propagation.rssi_noise_db:
            observed = physical + self.rng.normal(
                0.0, self.scenario.propagation.rssi_noise_db, size=physical.shape
            )
        else:
            observed = physical.copy()
        sta_ap_ids = np.asarray([sta.ap_id for sta in self.stas], dtype=int)
        row_indices = np.arange(len(self.stas))
        if self.stas:
            valid = (sta_ap_ids >= 0) & (sta_ap_ids < len(self.aps))
            link_rssi = np.full(len(self.stas), -100.0, dtype=float)
            link_rssi[valid] = observed[row_indices[valid], sta_ap_ids[valid]]
        else:
            link_rssi = np.empty(0, dtype=float)
        demand = np.asarray([sta.offered_mbps for sta in self.stas], dtype=float)
        mac_result = compute_mac_metrics(
            self.aps,
            sta_ap_ids,
            link_rssi,
            demand,
            self.scenario.propagation,
            self.scenario.mac,
        )
        measured_tput = mac_result.throughput_sta.copy()
        if self.scenario.measurement_noise and self.scenario.mac.tput_noise_fraction:
            noise = self.rng.uniform(
                1.0 - self.scenario.mac.tput_noise_fraction,
                1.0 + self.scenario.mac.tput_noise_fraction,
                size=measured_tput.shape,
            )
            measured_tput *= noise
            measured_tput = np.minimum(measured_tput, demand)

        # LVAP-style handovers lose 0.2 seconds of service in a one-second step; longer
        # configured outages suppress later steps as well.
        dt = float(self.scenario.step_s)
        for index, sta in enumerate(self.stas):
            remaining = self._outage_remaining.get(sta.sta_id, 0.0)
            if sta.sta_id in added_or_moved:
                outage = max(0.0, self.scenario.handover.outage_s)
                factor = max(0.0, 1.0 - outage / dt) if outage < dt else 0.0
                self._outage_remaining[sta.sta_id] = max(0.0, outage - dt)
            elif remaining > 0:
                factor = 0.0
                self._outage_remaining[sta.sta_id] = max(0.0, remaining - dt)
            else:
                factor = 1.0
            measured_tput[index] *= factor
        for ap_id, remaining in list(self._channel_outage_remaining.items()):
            ap_sta_indices = np.flatnonzero(sta_ap_ids == ap_id)
            if ap_id in self._current_channel_change_ids:
                factor = max(0.0, 1.0 - 0.5 / dt)
                self._channel_outage_remaining[ap_id] = max(0.0, 0.5 - dt)
            elif remaining > 0:
                factor = 0.0
                self._channel_outage_remaining[ap_id] = max(0.0, remaining - dt)
            else:
                factor = 1.0
            measured_tput[ap_sta_indices] *= factor

        report = build_indicator_report(
            self.t,
            self.aps,
            self.stas,
            observed,
            link_rssi,
            demand,
            mac_result,
            measured_tput,
        )
        self.store.append(report)
        self._rssi_matrix = observed
        self._last_mac = mac_result
        self._last_measured_tput = measured_tput
        self.state = NetworkState(
            t=self.t,
            aps=self.aps,
            stas=self.stas,
            rssi_dbm=observed,
            reports=self.store.reports,
            sta_tput_mbps=measured_tput,
            sta_phy_mbps=mac_result.phy_mbps,
            sta_loss_frac=mac_result.p_fail,
            sta_retx_per_frame=mac_result.retx_per_frame,
            sta_demand_mbps=demand,
        )

    def _make_step_metrics(
        self, moved_ids: set[int], handover_count: int, invalid_count: int
    ) -> dict[str, Any]:
        """Prepare the current-step metrics used by Gymnasium and the experiment runner."""

        state = self.state
        if state is None:
            return {}
        total_demand = float(state.sta_demand_mbps.sum())
        aggregate_tput = float(state.sta_tput_mbps.sum())
        active = (
            (state.sta_demand_mbps > 0)
            & (np.asarray([sta.ap_id for sta in state.stas]) >= 0)
            & (state.sta_phy_mbps > 0)
        )
        mean_retx = float(np.mean(state.sta_retx_per_frame[active])) if np.any(active) else 0.0
        return {
            "t": self.t,
            "agg_tput_mbps": aggregate_tput,
            "offered_mbps": total_demand,
            "jfi": satisfaction_fairness(state.sta_tput_mbps, state.sta_demand_mbps),
            "retx_per_frame": mean_retx,
            "occ_mean": float(np.mean(state.reports[-1].occ_frac)),
            "occ_std": float(np.std(state.reports[-1].occ_frac)),
            "phy_mean_mbps": float(np.mean(state.sta_phy_mbps[active])) if np.any(active) else 0.0,
            "handovers": handover_count,
            "moved_sta_ids": sorted(moved_ids),
            "invalid_actions": invalid_count,
            "throughput": state.sta_tput_mbps.copy(),
            "demand": state.sta_demand_mbps.copy(),
            "sta_ids": [sta.sta_id for sta in state.stas],
            "sta_ap_ids": [sta.ap_id for sta in state.stas],
            "sta_loss": state.sta_loss_frac.copy(),
            "sta_retx": state.sta_retx_per_frame.copy(),
            "sta_phy": state.sta_phy_mbps.copy(),
            "jfi_value": satisfaction_fairness(state.sta_tput_mbps, state.sta_demand_mbps),
        }

    def moving_average(self, field: str, window: int = 10) -> np.ndarray:
        """Return the recent moving average of a per-AP indicator field."""

        return self.store.average(field, window)

    def snapshot(self) -> NetworkState:
        """Return the latest public state, failing if the network has not been reset."""

        if self.state is None:
            raise RuntimeError("Network must be reset before requesting a snapshot")
        return self.state

    def clone_scenario(self, scenario: ScenarioConfig) -> WifiNetwork:
        """Construct a fresh network with the same seed and a changed scenario."""

        return WifiNetwork(replace(scenario), self.seed)
