"""Shared interface for association and channel-assignment policies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from fa_rlwa.types import Action, NetworkState


class Policy(ABC):
    """Stateful network policy; all policy randomness is supplied as a Generator."""

    name = "policy"

    def reset(self, state: NetworkState, rng: np.random.Generator) -> None:
        """Reset policy state at the beginning of a seeded run."""

        return None

    def initial_ap(self, sta_id: int, state: NetworkState) -> int:
        """Return the strongest reachable AP for a new station by default."""

        try:
            row = next(index for index, sta in enumerate(state.stas) if sta.sta_id == sta_id)
        except StopIteration:
            return -1
        if row >= len(state.rssi_dbm):
            return -1
        rssi = state.rssi_dbm[row]
        reachable = rssi >= -82.0
        return int(np.argmax(np.where(reachable, rssi, -np.inf))) if np.any(reachable) else -1

    @abstractmethod
    def decide(self, state: NetworkState) -> list[Action]:
        """Return zero or more handover/channel-change requests for the current cycle."""


class ClientDrivenPolicy(Policy):
    """RSSI-only client baseline; the simulator handles initial selection and link loss."""

    name = "client_driven"

    def decide(self, state: NetworkState) -> list[Action]:
        """Client-driven association is decentralized and issues no controller actions."""

        return []
