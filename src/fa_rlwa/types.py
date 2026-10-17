"""Typed simulator and controller data structures (distances in m, power in dBm)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np


@dataclass
class APState:
    """One simulated access point."""

    ap_id: int
    channel: int
    pos: tuple[float, float]
    tx_dbm: float = 20.0
    last_channel_change_t: int = -10_000


@dataclass
class STAState:
    """One wireless station; throughput and offered load are in Mbps."""

    sta_id: int
    ap_id: int
    pos: tuple[float, float]
    traffic_type: str
    offered_mbps: float
    base_offered_mbps: float | None = None
    mobility: str = "static"
    last_handover_t: int = -10_000
    previous_ap_id: int = -1
    arrival_t: int = 0
    departure_t: int | None = None
    speed_mps: float = 0.0
    destination: tuple[float, float] | None = None
    pause_s: int = 0
    path: tuple[tuple[float, float, int], ...] = ()
    path_index: int = 0
    path_hold_s: int = 0
    shadow_db: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float), repr=False)
    video_delta: float = 0.0


@dataclass
class IndicatorReport:
    """Per-AP observations for one second; loss and occupancy are fractions."""

    t: int
    rssi_dbm: np.ndarray
    loss_frac: np.ndarray
    tput_mbps: np.ndarray
    occ_frac: np.ndarray
    n_clients: np.ndarray
    airtime_own_frac: np.ndarray
    demand_mbps: np.ndarray


@dataclass
class NetworkState:
    """Snapshot passed to policies; RSSI matrix has shape [n_sta, n_ap]."""

    t: int
    aps: list[APState]
    stas: list[STAState]
    rssi_dbm: np.ndarray
    reports: list[IndicatorReport]
    sta_tput_mbps: np.ndarray
    sta_phy_mbps: np.ndarray
    sta_loss_frac: np.ndarray
    sta_retx_per_frame: np.ndarray
    sta_demand_mbps: np.ndarray


@dataclass(frozen=True)
class Handover:
    """Request to move a STA to a target AP."""

    sta_id: int
    target_ap: int
    reason: str = "policy"


@dataclass(frozen=True)
class ChannelChange:
    """Request to change an AP's 2.4 GHz channel."""

    ap_id: int
    channel: int
    reason: str = "policy"


Action: TypeAlias = Handover | ChannelChange
