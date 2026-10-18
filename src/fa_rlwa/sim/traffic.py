"""Synthetic traffic profiles used by the abstract simulator."""

from __future__ import annotations

import numpy as np

from fa_rlwa.types import STAState


def update_offered_load(sta: STAState, rng: np.random.Generator) -> float:
    """Update and return the STA offered load in Mbps for the current second."""

    if sta.traffic_type == "video":
        # Bounded AR(1) modulation around the configured four-Mbps mean.
        sta.video_delta = float(np.clip(0.75 * sta.video_delta + rng.normal(0.0, 0.08), -0.3, 0.3))
        baseline = sta.base_offered_mbps if sta.base_offered_mbps is not None else sta.offered_mbps
        sta.offered_mbps = max(0.0, baseline * (1.0 + sta.video_delta))
    elif sta.traffic_type not in {"cbr", "voip", "bulk"}:
        raise ValueError(f"Unsupported traffic profile {sta.traffic_type!r}")
    return sta.offered_mbps
