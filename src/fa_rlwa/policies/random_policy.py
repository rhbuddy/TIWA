"""Random handover baseline for sanity checks."""

from __future__ import annotations

import numpy as np

from fa_rlwa.policies.base import Policy
from fa_rlwa.types import Action, Handover, NetworkState


class RandomPolicy(Policy):
    """At each cycle, move a random eligible STA with probability 0.02."""

    name = "random"

    def __init__(self, probability: float = 0.02, dwell_s: int = 10) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("random move probability must be in [0, 1]")
        self.probability = float(probability)
        self.dwell_s = int(dwell_s)
        self.rng = np.random.default_rng(0)

    def reset(self, state: NetworkState, rng: np.random.Generator) -> None:
        """Use the run's explicit random-number generator."""

        self.rng = rng

    def decide(self, state: NetworkState) -> list[Action]:
        """Choose one random eligible STA/AP pair, or stay."""

        if not state.stas or self.rng.random() >= self.probability:
            return []
        eligible: list[tuple[int, list[int]]] = []
        for row, sta in enumerate(state.stas):
            if state.t - sta.last_handover_t < self.dwell_s:
                continue
            targets = [
                ap_id
                for ap_id, rssi in enumerate(state.rssi_dbm[row])
                if ap_id != sta.ap_id and rssi >= -82.0
            ]
            if targets:
                eligible.append((sta.sta_id, targets))
        if not eligible:
            return []
        sta_id, targets = eligible[int(self.rng.integers(len(eligible)))]
        target = targets[int(self.rng.integers(len(targets)))]
        return [Handover(sta_id, target, reason="random")]
