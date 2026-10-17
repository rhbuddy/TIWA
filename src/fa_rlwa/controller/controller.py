"""Policy dispatch, cycle timing, and controller/network integration."""

from __future__ import annotations

import time

import numpy as np

from fa_rlwa.policies.base import Policy
from fa_rlwa.sim.network import WifiNetwork
from fa_rlwa.types import NetworkState


class SDNController:
    """Small controller abstraction that calls a policy once per network step."""

    def __init__(self, network: WifiNetwork, policy: Policy, seed: int = 0) -> None:
        self.network = network
        self.policy = policy
        self.seed = int(seed)
        self.decision_latencies_ms: list[float] = []
        self.state: NetworkState = network.snapshot()
        self.policy.reset(self.state, np.random.default_rng(self.seed))

    def reset(self, seed: int | None = None) -> NetworkState:
        """Reset network and policy state using the provided run seed."""

        if seed is not None:
            self.seed = int(seed)
        self.state = self.network.reset(seed=self.seed)
        self.decision_latencies_ms = []
        self.policy.reset(self.state, np.random.default_rng(self.seed))
        return self.state

    def step(self) -> NetworkState:
        """Make one policy decision, execute it, and record decision-cycle latency."""

        started = time.perf_counter()
        actions = self.policy.decide(self.state)
        self.state = self.network.step(actions)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.decision_latencies_ms.append(latency_ms)
        self.network.last_step_metrics["decision_latency_ms"] = latency_ms
        return self.state
