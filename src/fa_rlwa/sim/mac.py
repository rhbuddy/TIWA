"""Airtime-sharing MAC approximation from SRS Sections 9.4-9.5."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fa_rlwa.config import MacConfig, PropagationConfig
from fa_rlwa.sim.propagation import channel_error_probability, path_loss_db, phy_rate_mbps
from fa_rlwa.types import APState


@dataclass
class MacResult:
    """Per-STA MAC outputs and per-AP airtime aggregates for one control cycle."""

    phy_mbps: np.ndarray
    p_fail: np.ndarray
    retx_per_frame: np.ndarray
    airtime_sta: np.ndarray
    airtime_ap: np.ndarray
    occupancy_ap: np.ndarray
    scale_ap: np.ndarray
    throughput_sta: np.ndarray


def collision_probability(n_active: int, tau: float = 0.01, cap: float = 0.5) -> float:
    """Return the capped unslotted collision approximation for an AP neighbourhood."""

    if n_active <= 1:
        return 0.0
    return float(min(cap, 1.0 - (1.0 - tau) ** (n_active - 1)))


def airtime_demand(offered_mbps: float, phy_mbps: float, efficiency: float, p_fail: float) -> float:
    """Return airtime fraction required to serve offered load at the effective goodput."""

    goodput = phy_mbps * efficiency * (1.0 - p_fail)
    if offered_mbps <= 0 or goodput <= 0:
        return 0.0
    return float(offered_mbps / goodput)


def collision_neighbourhoods(aps: list[APState], propagation: PropagationConfig) -> list[set[int]]:
    """Build same-channel carrier-sense neighbourhoods using AP-to-AP path loss."""

    neighbours: list[set[int]] = [{ap.ap_id} for ap in aps]
    for i, ap_i in enumerate(aps):
        for j, ap_j in enumerate(aps):
            if i == j or ap_i.channel != ap_j.channel:
                continue
            distance = float(np.hypot(ap_i.pos[0] - ap_j.pos[0], ap_i.pos[1] - ap_j.pos[1]))
            rx_dbm = ap_i.tx_dbm - path_loss_db(distance, propagation.pl0_db, propagation.exponent)
            if rx_dbm >= propagation.ap_carrier_sense_dbm:
                neighbours[i].add(j)
    return neighbours


def compute_mac_metrics(
    aps: list[APState],
    sta_ap_ids: np.ndarray,
    sta_rssi_dbm: np.ndarray,
    offered_mbps: np.ndarray,
    propagation: PropagationConfig,
    mac: MacConfig,
) -> MacResult:
    """Compute PHY rates, failures, airtime, occupancy, and delivered Mbps."""

    n_ap = len(aps)
    n_sta = len(sta_ap_ids)
    neighbours = collision_neighbourhoods(aps, propagation)
    p_coll_ap = np.zeros(n_ap, dtype=float)
    for ap_id, group in enumerate(neighbours):
        active = int(np.count_nonzero(np.isin(sta_ap_ids, list(group)) & (offered_mbps > 0.0)))
        p_coll_ap[ap_id] = collision_probability(active, mac.tau, mac.collision_cap)

    p_coll_sta = np.zeros(n_sta, dtype=float)
    associated = (sta_ap_ids >= 0) & (sta_ap_ids < n_ap)
    if np.any(associated):
        p_coll_sta[associated] = p_coll_ap[sta_ap_ids[associated]]
    phy = np.asarray(phy_rate_mbps(sta_rssi_dbm, p_coll_sta), dtype=float)
    p_err = np.asarray(channel_error_probability(sta_rssi_dbm), dtype=float)
    p_fail = 1.0 - (1.0 - p_coll_sta) * (1.0 - p_err)
    p_fail[~associated] = 1.0

    goodput = phy * mac.efficiency * (1.0 - p_fail)
    airtime = np.divide(
        offered_mbps,
        goodput,
        out=np.zeros(n_sta, dtype=float),
        where=(goodput > 0.0) & associated,
    )
    airtime_ap = np.zeros(n_ap, dtype=float)
    for ap_id in range(n_ap):
        airtime_ap[ap_id] = float(airtime[sta_ap_ids == ap_id].sum())

    occupancy = np.zeros(n_ap, dtype=float)
    scale = np.ones(n_ap, dtype=float)
    for ap_id, group in enumerate(neighbours):
        neighbourhood_airtime = float(airtime_ap[list(group)].sum())
        occupancy[ap_id] = min(1.0, neighbourhood_airtime)
        scale[ap_id] = min(1.0, mac.c_max / max(neighbourhood_airtime, 1e-12))

    throughput = np.zeros(n_sta, dtype=float)
    serving = associated & (goodput > 0.0)
    throughput[serving] = scale[sta_ap_ids[serving]] * offered_mbps[serving]
    retransmissions = np.divide(
        p_fail,
        1.0 - p_fail,
        out=np.zeros(n_sta, dtype=float),
        where=p_fail < 1.0,
    )
    return MacResult(
        phy_mbps=phy,
        p_fail=p_fail,
        retx_per_frame=retransmissions,
        airtime_sta=airtime,
        airtime_ap=airtime_ap,
        occupancy_ap=occupancy,
        scale_ap=scale,
        throughput_sta=throughput,
    )
