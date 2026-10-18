"""SRS indicator normalization; all normalized indicators use higher-is-better."""

from __future__ import annotations

import numpy as np


def normalize_rssi(rssi_dbm: float | np.ndarray) -> float | np.ndarray:
    """Map RSSI from the -90 to -40 dBm reference interval into [0, 1]."""

    value = np.clip((np.asarray(rssi_dbm, dtype=float) + 90.0) / 50.0, 0.0, 1.0)
    return float(value) if value.ndim == 0 else value


def normalize_loss(loss_frac: float | np.ndarray) -> float | np.ndarray:
    """Map a failure fraction into a higher-is-better link-quality score."""

    value = 1.0 - np.clip(np.asarray(loss_frac, dtype=float) / 0.5, 0.0, 1.0)
    return float(value) if value.ndim == 0 else value


def normalize_throughput_headroom(tput_mbps: float | np.ndarray) -> float | np.ndarray:
    """Map throughput into remaining headroom against the 39 Mbps reference."""

    value = 1.0 - np.clip(np.asarray(tput_mbps, dtype=float) / 39.0, 0.0, 1.0)
    return float(value) if value.ndim == 0 else value


def normalize_occupancy(occupancy_frac: float | np.ndarray) -> float | np.ndarray:
    """Convert channel occupancy into available airtime in [0, 1]."""

    value = 1.0 - np.clip(np.asarray(occupancy_frac, dtype=float), 0.0, 1.0)
    return float(value) if value.ndim == 0 else value


def normalize_client_count(count: float | np.ndarray, maximum: int = 100) -> float | np.ndarray:
    """Scale client counts against the SRS 100-STA reference maximum."""

    if maximum <= 0:
        raise ValueError("maximum client count must be positive")
    value = np.clip(np.asarray(count, dtype=float) / maximum, 0.0, 1.0)
    return float(value) if value.ndim == 0 else value


def normalize_ap_airtime(airtime_frac: float | np.ndarray) -> float | np.ndarray:
    """Return the AP's unused airtime fraction."""

    value = 1.0 - np.clip(np.asarray(airtime_frac, dtype=float), 0.0, 1.0)
    return float(value) if value.ndim == 0 else value
