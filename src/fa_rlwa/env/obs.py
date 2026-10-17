"""Fixed-size Gymnasium observation vector and candidate-STA selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from fa_rlwa.forecast.base import Forecaster
from fa_rlwa.monitor.indicators import satisfaction_fairness
from fa_rlwa.monitor.normalize import (
    normalize_client_count,
    normalize_loss,
    normalize_occupancy,
    normalize_rssi,
    normalize_throughput_headroom,
)
from fa_rlwa.types import NetworkState


def observation_size(k_max: int, use_forecast: bool = True) -> int:
    """Return the observation length defined by SRS Section 12.2."""

    if k_max < 1:
        raise ValueError("k_max must be positive")
    return (12 if use_forecast else 8) * k_max + 4


def select_candidate_sta(state: NetworkState, dwell_s: int = 10) -> int | None:
    """Choose the eligible STA with highest SRS priority, breaking ties by lowest ID."""

    if not state.stas or not state.reports:
        return None
    report = state.reports[-1]
    ranked: list[tuple[float, int]] = []
    for row, sta in enumerate(state.stas):
        if sta.ap_id < 0 or state.t - sta.last_handover_t < dwell_s:
            continue
        ap_id = sta.ap_id
        current_rssi = float(state.rssi_dbm[row, ap_id])
        rssi_badness = 1.0 - float(normalize_rssi(current_rssi))
        priority = (
            0.5 * float(report.occ_frac[ap_id])
            + 0.3 * rssi_badness
            + 0.2 * float(report.loss_frac[ap_id])
        )
        ranked.append((priority, sta.sta_id))
    if not ranked:
        return None
    return min(ranked, key=lambda item: (-item[0], item[1]))[1]


def _forecast_features(state: NetworkState, forecaster: Forecaster | None) -> np.ndarray:
    """Obtain the +10-second (or nearest available) per-AP forecast."""

    report = state.reports[-1]
    if forecaster is None:
        return np.stack(
            [report.rssi_dbm, report.loss_frac, report.occ_frac, report.tput_mbps], axis=1
        )
    history_length = max(30, int(getattr(forecaster, "history_window_s", 30)))
    report_history = state.reports[-history_length:]
    features = [
        np.stack(
            [
                item.rssi_dbm,
                item.loss_frac,
                item.tput_mbps,
                item.occ_frac,
                item.n_clients.astype(float),
                item.demand_mbps,
            ],
            axis=1,
        )
        for item in report_history
    ]
    history = np.stack(features, axis=1)
    if history.shape[1] < 30:
        history = np.pad(history, ((0, 0), (30 - history.shape[1], 0), (0, 0)), mode="edge")
    prediction = forecaster.predict(history)
    horizons = list(forecaster.horizons)
    horizon_index = horizons.index(10) if 10 in horizons else min(1, len(horizons) - 1)
    # Forecaster target order is RSSI, loss, throughput, occupancy; observation order is
    # RSSI, loss, occupancy, throughput.
    return prediction[:, horizon_index, :][:, [0, 1, 3, 2]]


def build_observation(
    state: NetworkState,
    k_max: int = 5,
    use_forecast: bool = True,
    forecaster: Forecaster | None = None,
    candidate_sta_id: int | None = None,
    rssi_history: Sequence[tuple[int, Mapping[int, np.ndarray]]] = (),
    fairness_history: Sequence[float] = (),
) -> np.ndarray:
    """Build a normalized ``[0, 1]`` observation for one association candidate."""

    n_ap = len(state.aps)
    if n_ap > k_max:
        raise ValueError(f"Scenario has {n_ap} APs but policy k_max={k_max}")
    if not state.reports:
        raise ValueError("Network state has no indicator reports")
    latest = state.reports[-1]
    forecast = _forecast_features(state, forecaster) if use_forecast else None
    values: list[float] = []
    for ap_id in range(k_max):
        if ap_id >= n_ap:
            values.extend([0.0] * (9 if use_forecast else 5))
            continue
        base = [
            float(normalize_rssi(latest.rssi_dbm[ap_id])),
            float(normalize_loss(latest.loss_frac[ap_id])),
            float(normalize_occupancy(latest.occ_frac[ap_id])),
            float(normalize_throughput_headroom(latest.tput_mbps[ap_id])),
            float(normalize_client_count(latest.n_clients[ap_id])),
        ]
        if use_forecast and forecast is not None:
            base.extend(
                [
                    float(normalize_rssi(forecast[ap_id, 0])),
                    float(normalize_loss(forecast[ap_id, 1])),
                    float(normalize_occupancy(forecast[ap_id, 2])),
                    float(normalize_throughput_headroom(forecast[ap_id, 3])),
                ]
            )
        values.extend(base)

    candidate = (
        next((sta for sta in state.stas if sta.sta_id == candidate_sta_id), None)
        if candidate_sta_id is not None
        else None
    )
    row = (
        next(
            (index for index, sta in enumerate(state.stas) if sta.sta_id == candidate_sta_id), None
        )
        if candidate is not None
        else None
    )
    if candidate is None or row is None:
        values.extend([0.0] * (3 * k_max + 4))
    else:
        rssi = np.asarray(state.rssi_dbm[row], dtype=float)
        values.extend(
            [
                float(normalize_rssi(rssi[ap_id])) if ap_id < n_ap and rssi[ap_id] >= -82.0 else 0.0
                for ap_id in range(k_max)
            ]
        )
        values.extend([1.0 if candidate.ap_id == ap_id else 0.0 for ap_id in range(k_max)])
        slopes = np.zeros(k_max, dtype=float)
        samples: list[tuple[int, np.ndarray]] = []
        for t, snapshot in rssi_history[-10:]:
            if candidate.sta_id in snapshot:
                samples.append((int(t), np.asarray(snapshot[candidate.sta_id], dtype=float)))
        if len(samples) >= 2:
            times = np.asarray([sample[0] for sample in samples], dtype=float)
            times -= times[0]
            matrix = np.stack([sample[1] for sample in samples])
            for ap_id in range(min(k_max, n_ap)):
                slopes[ap_id] = float(np.polyfit(times, matrix[:, ap_id], 1)[0])
        slopes = np.clip(slopes / 2.0, -1.0, 1.0)
        trend_features = (slopes + 1.0) / 2.0
        trend_features[n_ap:] = 0.0
        values.extend(trend_features.tolist())
        values.extend(
            [
                float(np.clip(candidate.offered_mbps / 10.0, 0.0, 1.0)),
                float(np.clip((state.t - candidate.last_handover_t) / 60.0, 0.0, 1.0)),
                float(
                    np.mean(fairness_history[-10:])
                    if fairness_history
                    else satisfaction_fairness(state.sta_tput_mbps, state.sta_demand_mbps)
                ),
                float(np.clip(len(state.stas) / 100.0, 0.0, 1.0)),
            ]
        )
    observation = np.asarray(values, dtype=np.float32)
    expected = observation_size(k_max, use_forecast)
    if observation.shape != (expected,):
        raise RuntimeError(f"Observation has shape {observation.shape}; expected ({expected},)")
    return np.clip(observation, 0.0, 1.0).astype(np.float32)
