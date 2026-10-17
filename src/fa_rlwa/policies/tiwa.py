"""Published-TiWA interpretation, normalized correction, and forecast wrapper."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from fa_rlwa.config import PropagationConfig
from fa_rlwa.forecast.base import Forecaster
from fa_rlwa.monitor.normalize import (
    normalize_loss,
    normalize_occupancy,
    normalize_rssi,
    normalize_throughput_headroom,
)
from fa_rlwa.monitor.store import IndicatorStore
from fa_rlwa.policies.base import Policy
from fa_rlwa.sim.mac import collision_neighbourhoods
from fa_rlwa.types import Action, ChannelChange, Handover, IndicatorReport, NetworkState


def _row_for_sta(state: NetworkState, sta_id: int) -> int | None:
    """Find the state-array row for a stable STA identifier."""

    return next((index for index, sta in enumerate(state.stas) if sta.sta_id == sta_id), None)


def _smoothed_report(state: NetworkState, window: int = 10) -> IndicatorReport:
    """Build the most-recent per-AP report averaged over up to ten samples."""

    store = IndicatorStore(max_reports=max(window, 1))
    for report in state.reports[-window:]:
        store.append(report)
    last = state.reports[-1]
    fields = {
        name: store.average(name, window)
        for name in (
            "rssi_dbm",
            "loss_frac",
            "tput_mbps",
            "occ_frac",
            "n_clients",
            "airtime_own_frac",
            "demand_mbps",
        )
    }
    return IndicatorReport(t=last.t, **fields)


def _imbalance(values: np.ndarray) -> bool:
    """Apply the paper/SRS max-minus-min versus median trigger literally."""

    data = np.asarray(values, dtype=float)
    return bool(data.size > 1 and float(np.max(data) - np.min(data)) > float(np.median(data)))


def assign_channels(state: NetworkState) -> list[ChannelChange]:
    """Greedily apply the TiWA Algorithm 2 channel assignment (channels 1/6/11)."""

    channels = (1, 6, 11)
    aps = state.aps
    neighbours = collision_neighbourhoods(aps, PropagationConfig())
    occupancies = state.reports[-1].occ_frac
    order = sorted(
        range(len(aps)),
        key=lambda ap_id: (-len(neighbours[ap_id]), float(occupancies[ap_id]), ap_id),
    )
    assigned: dict[int, int] = {}
    actions: list[ChannelChange] = []
    for ap_id in order:
        used = {
            assigned[neighbor]
            for neighbor in neighbours[ap_id]
            if neighbor != ap_id and neighbor in assigned
        }
        available = [channel for channel in channels if channel not in used]
        if aps[ap_id].channel in available:
            choice = aps[ap_id].channel
        elif available:
            choice = available[0]
        else:
            scores = {}
            for channel in channels:
                nearby_load = sum(
                    float(occupancies[other])
                    for other in range(len(aps))
                    if other != ap_id
                    and assigned.get(other, aps[other].channel) == channel
                    and other in neighbours[ap_id]
                )
                scores[channel] = nearby_load
            choice = min(scores, key=lambda channel: (scores[channel], channel))
        assigned[ap_id] = choice
        if choice != aps[ap_id].channel:
            actions.append(ChannelChange(ap_id, choice, reason="tiwa_algorithm_2"))
    return actions


class TiwaOriginalPolicy(Policy):
    """Reproduction of TiWA's four-indicator heuristic with logged interpretations."""

    name = "tiwa_original"

    def __init__(
        self,
        window: int = 10,
        threshold_mode: str = "spatial",
        migration_margin: float = 0.05,
        max_moves_per_cycle: int = 2,
        dwell_s: int = 10,
        channel_reassign_interval_s: int = 60,
    ) -> None:
        if threshold_mode not in {"spatial", "temporal"}:
            raise ValueError("threshold_mode must be 'spatial' or 'temporal'")
        self.window = int(window)
        self.threshold_mode = threshold_mode
        self.migration_margin = float(migration_margin)
        self.max_moves_per_cycle = int(max_moves_per_cycle)
        self.dwell_s = int(dwell_s)
        self.channel_reassign_interval_s = int(channel_reassign_interval_s)

    def decide(self, state: NetworkState) -> list[Action]:
        """Reassociate candidates on imbalance and run channel assignment on occupancy trigger."""

        if not state.aps or not state.reports:
            return []
        report = _smoothed_report(state, self.window)
        indicator_values = {
            "rssi": report.rssi_dbm,
            "loss": report.loss_frac,
            "tput": report.tput_mbps,
            "occ": report.occ_frac,
        }
        if self.threshold_mode == "spatial":
            triggers = {name: _imbalance(values) for name, values in indicator_values.items()}
        else:
            recent = state.reports[-self.window :]
            triggers = {}
            for name, field_name in (
                ("rssi", "rssi_dbm"),
                ("loss", "loss_frac"),
                ("tput", "tput_mbps"),
                ("occ", "occ_frac"),
            ):
                matrix = np.stack([getattr(item, field_name) for item in recent])
                triggers[name] = any(
                    _imbalance(matrix[:, ap_id]) for ap_id in range(matrix.shape[1])
                )

        actions: list[Action] = []
        # I-2 and I-4: an indicator trigger re-evaluates STAs on the AP with the
        # maximum value, strongest alternative first, with at most two moves per cycle.
        if any(triggers[name] for name in ("rssi", "loss", "tput")):
            triggered = [name for name in ("rssi", "loss", "tput") if triggers[name]]
            overloaded = sorted(
                {int(np.argmax(indicator_values[name])) for name in triggered},
                key=lambda ap_id: ap_id,
            )
            potential: list[tuple[float, int, int]] = []
            for ap_id in overloaded:
                for row, sta in enumerate(state.stas):
                    if sta.ap_id != ap_id or state.t - sta.last_handover_t < self.dwell_s:
                        continue
                    available = [
                        (float(state.rssi_dbm[row, target]), target)
                        for target in range(len(state.aps))
                        if target != ap_id and state.rssi_dbm[row, target] >= -82.0
                    ]
                    if available:
                        alternative_rssi, target = max(available)
                        potential.append((alternative_rssi, sta.sta_id, target))
            for _, sta_id, target_ap in sorted(potential, reverse=True):
                sta_row = _row_for_sta(state, sta_id)
                if sta_row is None:
                    continue
                current_ap = state.stas[sta_row].ap_id
                cost = (
                    report.occ_frac
                    + report.airtime_own_frac
                    + report.loss_frac
                    + report.tput_mbps / 39.0
                )
                scores = state.rssi_dbm[sta_row] * cost
                target_score = float(scores[target_ap])
                current_score = float(scores[current_ap]) if current_ap >= 0 else -np.inf
                threshold = (
                    current_score + self.migration_margin * abs(current_score)
                    if np.isfinite(current_score)
                    else -np.inf
                )
                if target_score > threshold:
                    actions.append(Handover(sta_id, target_ap, reason="tiwa_original"))
                    if len(actions) >= self.max_moves_per_cycle:
                        return actions

        # Algorithm 1 triggers Algorithm 2 only when channel-occupancy indicators are imbalanced.
        if triggers["occ"]:
            for action in assign_channels(state):
                ap = state.aps[action.ap_id]
                if state.t - ap.last_channel_change_t >= self.channel_reassign_interval_s:
                    actions.append(action)
        return actions


class CorrectedTiwaPolicy(Policy):
    """Normalized weighted TiWA policy with explicit overload and migration thresholds."""

    name = "tiwa_fixed"

    def __init__(
        self,
        weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
        delta: float = 0.05,
        occupancy_threshold: float = 0.7,
        loss_threshold: float = 0.15,
        max_moves_per_cycle: int = 2,
        dwell_s: int = 10,
    ) -> None:
        if len(weights) != 4 or not np.isclose(sum(weights), 1.0) or min(weights) < 0:
            raise ValueError("corrected TiWA weights must be four non-negative values summing to 1")
        self.weights = tuple(float(value) for value in weights)
        self.delta = float(delta)
        self.occupancy_threshold = float(occupancy_threshold)
        self.loss_threshold = float(loss_threshold)
        self.max_moves_per_cycle = int(max_moves_per_cycle)
        self.dwell_s = int(dwell_s)

    def decide(self, state: NetworkState) -> list[Action]:
        """Move at most two eligible STAs when a normalized target score improves by delta."""

        if not state.stas or not state.reports:
            return []
        report = _smoothed_report(state, 10)
        overloaded = set(
            np.flatnonzero(
                (report.occ_frac > self.occupancy_threshold)
                | (report.loss_frac > self.loss_threshold)
            ).tolist()
        )
        if not overloaded:
            return []
        weight_r, weight_l, weight_o, weight_t = self.weights
        candidates: list[tuple[float, int, int]] = []
        for row, sta in enumerate(state.stas):
            if sta.ap_id not in overloaded or state.t - sta.last_handover_t < self.dwell_s:
                continue
            rssi_score = np.asarray(normalize_rssi(state.rssi_dbm[row]), dtype=float)
            ap_scores = np.asarray(
                weight_r * rssi_score
                + weight_l * normalize_loss(report.loss_frac)
                + weight_o * normalize_occupancy(report.occ_frac)
                + weight_t * normalize_throughput_headroom(report.tput_mbps),
                dtype=float,
            )
            eligible = [
                ap_id
                for ap_id in range(len(state.aps))
                if ap_id != sta.ap_id and state.rssi_dbm[row, ap_id] >= -82.0
            ]
            if not eligible:
                continue
            target = max(eligible, key=lambda ap_id: (float(ap_scores[ap_id]), -ap_id))
            current = float(ap_scores[sta.ap_id]) if sta.ap_id >= 0 else -np.inf
            improvement = float(ap_scores[target]) - current
            if improvement > self.delta:
                candidates.append((improvement, sta.sta_id, target))
        return [
            Handover(sta_id, target, reason="tiwa_normalized")
            for _, sta_id, target in sorted(candidates, reverse=True)[: self.max_moves_per_cycle]
        ]


class TiwaForecastPolicy(Policy):
    """Blend current TiWA indicators with a frozen forecaster's +10 s prediction."""

    name = "tiwa_forecast"

    def __init__(
        self,
        forecaster: Forecaster,
        base_policy: Policy | None = None,
        beta: float = 0.5,
    ) -> None:
        if not 0.0 <= beta <= 1.0:
            raise ValueError("forecast beta must be in [0, 1]")
        self.forecaster = forecaster
        self.base_policy = base_policy or CorrectedTiwaPolicy()
        self.beta = float(beta)

    def reset(self, state: NetworkState, rng: np.random.Generator) -> None:
        """Reset the wrapped policy; forecaster weights stay frozen."""

        self.base_policy.reset(state, rng)

    def decide(self, state: NetworkState) -> list[Action]:
        """Call the base TiWA policy using current/predicted indicator blends."""

        if not state.reports:
            return []
        history_length = max(30, int(getattr(self.forecaster, "history_window_s", 30)))
        history_reports = state.reports[-history_length:]
        series: list[np.ndarray] = []
        for report in history_reports:
            series.append(
                np.stack(
                    [
                        report.rssi_dbm,
                        report.loss_frac,
                        report.tput_mbps,
                        report.occ_frac,
                        report.n_clients.astype(float),
                        report.demand_mbps,
                    ],
                    axis=1,
                )
            )
        history = np.stack(series, axis=1)  # [AP, time, feature]
        if history.shape[1] < 30:
            history = np.pad(history, ((0, 0), (30 - history.shape[1], 0), (0, 0)), mode="edge")
        predicted = self.forecaster.predict(history)
        horizons = list(self.forecaster.horizons)
        horizon_index = horizons.index(10) if 10 in horizons else min(1, len(horizons) - 1)
        forecast = predicted[:, horizon_index, :]
        report = state.reports[-1]
        current = np.stack(
            [report.rssi_dbm, report.loss_frac, report.tput_mbps, report.occ_frac], axis=1
        )
        used = (1.0 - self.beta) * current + self.beta * forecast
        blended = replace(
            report,
            rssi_dbm=used[:, 0],
            loss_frac=used[:, 1],
            tput_mbps=used[:, 2],
            occ_frac=used[:, 3],
        )
        # The blended report is the decision input for this cycle; averaging it with nine
        # older raw reports would silently shrink the configured forecast weight by 10x.
        forecast_state = replace(state, reports=[blended])
        return self.base_policy.decide(forecast_state)
