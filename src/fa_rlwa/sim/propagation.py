"""Log-distance path loss, shadowing, RSSI, and 802.11n PHY-rate lookup."""

from __future__ import annotations

import numpy as np

PHY_THRESHOLDS_DBM = np.asarray([-82.0, -79.0, -77.0, -74.0, -70.0, -66.0, -65.0, -64.0])
PHY_RATES_MBPS = np.asarray([6.5, 13.0, 19.5, 26.0, 39.0, 52.0, 58.5, 65.0])


def path_loss_db(
    distance_m: float | np.ndarray,
    pl0_db: float = 40.0,
    exponent: float = 3.0,
    shadow_db: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """Return path loss in dB; distances below one metre use the one-metre reference."""

    distance = np.maximum(np.asarray(distance_m, dtype=float), 1.0)
    loss = pl0_db + 10.0 * exponent * np.log10(distance) + np.asarray(shadow_db)
    return float(loss) if loss.ndim == 0 else loss


def rssi_dbm(
    tx_dbm: float,
    distance_m: float | np.ndarray,
    pl0_db: float = 40.0,
    exponent: float = 3.0,
    shadow_db: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """Compute received signal strength in dBm from transmit power and path loss."""

    result = tx_dbm - path_loss_db(distance_m, pl0_db, exponent, shadow_db)
    return float(result) if np.ndim(result) == 0 else np.asarray(result)


def phy_rate_mbps(
    rssi: float | np.ndarray,
    collision_probability: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """Map effective RSSI (dBm) to the highest supported 802.11n 20 MHz rate."""

    effective = np.asarray(rssi, dtype=float) - 10.0 * np.asarray(collision_probability)
    indices = np.searchsorted(PHY_THRESHOLDS_DBM, effective, side="right") - 1
    # The threshold array is ascending from -82 dBm; values above -64 use 65 Mbps.
    indices = np.clip(indices, -1, len(PHY_RATES_MBPS) - 1)
    rates = np.where(indices >= 0, PHY_RATES_MBPS[np.maximum(indices, 0)], 0.0)
    return float(rates) if rates.ndim == 0 else rates


def channel_error_probability(rssi: float | np.ndarray) -> float | np.ndarray:
    """Estimate per-attempt channel-error probability from uplink RSSI."""

    signal = np.asarray(rssi, dtype=float)
    result = 0.01 + 0.19 * np.clip((-65.0 - signal) / 17.0, 0.0, 1.0)
    return float(result) if result.ndim == 0 else result
