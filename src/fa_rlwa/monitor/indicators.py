"""Indicator aggregation and network-performance metrics."""

from __future__ import annotations

import numpy as np

from fa_rlwa.types import IndicatorReport, NetworkState


def jain_fairness(values: np.ndarray | list[float]) -> float:
    """Return Jain's fairness index for non-negative per-STA values (1 is perfectly fair)."""

    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0 or np.all(data == 0):
        return 1.0 if data.size == 0 else 0.0
    total = float(data.sum())
    squares = float(np.square(data).sum())
    if squares == 0.0:
        return 0.0
    return float(total * total / (data.size * squares))


def satisfaction_fairness(throughput: np.ndarray, demand: np.ndarray) -> float:
    """Compute Jain fairness over per-STA throughput/offered-load satisfaction."""

    ratio = np.divide(
        throughput, demand, out=np.zeros_like(throughput, dtype=float), where=demand > 0
    )
    return jain_fairness(ratio)


def build_indicator_report(
    t: int,
    aps: list,
    stas: list,
    rssi_matrix: np.ndarray,
    sta_rssi_dbm: np.ndarray,
    offered_mbps: np.ndarray,
    mac_result,
    measured_tput_mbps: np.ndarray,
) -> IndicatorReport:
    """Aggregate per-STA measurements into the four SRS per-AP indicators."""

    n_ap = len(aps)
    rssi = np.full(n_ap, -60.0, dtype=float)
    loss = np.zeros(n_ap, dtype=float)
    throughput = np.zeros(n_ap, dtype=float)
    n_clients = np.zeros(n_ap, dtype=int)
    demand = np.zeros(n_ap, dtype=float)
    for ap_id in range(n_ap):
        indices = np.flatnonzero(np.asarray([sta.ap_id for sta in stas]) == ap_id)
        n_clients[ap_id] = int(indices.size)
        if indices.size:
            rssi[ap_id] = float(np.mean(sta_rssi_dbm[indices]))
            loss[ap_id] = float(np.mean(mac_result.p_fail[indices]))
            throughput[ap_id] = float(np.sum(measured_tput_mbps[indices]))
            demand[ap_id] = float(np.sum(offered_mbps[indices]))
    return IndicatorReport(
        t=t,
        rssi_dbm=rssi,
        loss_frac=loss,
        tput_mbps=throughput,
        occ_frac=np.asarray(mac_result.occupancy_ap, dtype=float).copy(),
        n_clients=n_clients,
        airtime_own_frac=np.asarray(mac_result.airtime_ap, dtype=float).copy(),
        demand_mbps=demand,
    )


def moving_average(reports: list[IndicatorReport], field: str, window: int = 10) -> np.ndarray:
    """Average a report field over the most recent ``window`` reports."""

    if window < 1:
        raise ValueError("window must be at least one report")
    if not reports:
        raise ValueError("at least one indicator report is required")
    values = [np.asarray(getattr(report, field), dtype=float) for report in reports[-window:]]
    return np.mean(np.stack(values, axis=0), axis=0)


def network_jfi(state: NetworkState) -> float:
    """Compute current fairness over each STA's served fraction of offered load."""

    return satisfaction_fairness(state.sta_tput_mbps, state.sta_demand_mbps)
