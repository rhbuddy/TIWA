"""Configurable SRS reward calculation and component logging."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Weights for throughput, fairness, handover, retransmission, and invalid actions."""

    w1: float = 1.0
    w2: float = 0.5
    w3: float = 0.2
    w4: float = 0.3
    w5: float = 1.0


def calculate_reward(
    throughput_norm: float,
    fairness: float,
    handover: float,
    retransmissions: float,
    invalid_action: float,
    config: RewardConfig,
) -> tuple[float, dict[str, float]]:
    """Return the weighted scalar reward and its unweighted/loggable components."""

    terms = {
        "throughput_norm": float(throughput_norm),
        "fairness": float(fairness),
        "handover": float(handover),
        "retransmissions": float(min(1.0, max(0.0, retransmissions))),
        "invalid_action": float(invalid_action),
    }
    reward = (
        config.w1 * terms["throughput_norm"]
        + config.w2 * terms["fairness"]
        - config.w3 * terms["handover"]
        - config.w4 * terms["retransmissions"]
        - config.w5 * terms["invalid_action"]
    )
    return float(reward), terms
