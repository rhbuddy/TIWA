"""Gymnasium environment for discrete Wi-Fi association control."""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fa_rlwa.config import ScenarioConfig
from fa_rlwa.env.obs import build_observation, observation_size, select_candidate_sta
from fa_rlwa.env.reward import RewardConfig, calculate_reward
from fa_rlwa.forecast.base import Forecaster
from fa_rlwa.forecast.ewma import EWMAForecaster
from fa_rlwa.sim.network import WifiNetwork
from fa_rlwa.types import Handover, NetworkState


class WifiAssocEnv(gym.Env):
    """Single-candidate, one-second-step association-control MDP."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_mix: list[ScenarioConfig] | ScenarioConfig,
        reward_cfg: RewardConfig | None = None,
        forecaster: Forecaster | None = None,
        k_max: int = 5,
        use_forecast: bool = True,
        max_episode_steps: int = 600,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.scenario_mix = (
            [scenario_mix] if isinstance(scenario_mix, ScenarioConfig) else list(scenario_mix)
        )
        if not self.scenario_mix:
            raise ValueError("scenario_mix must contain at least one scenario")
        if any(len(scenario.aps) > k_max for scenario in self.scenario_mix):
            raise ValueError("k_max must be at least the largest scenario AP count")
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        self.reward_cfg = reward_cfg or RewardConfig()
        self.forecaster = forecaster or (EWMAForecaster() if use_forecast else None)
        self.k_max = int(k_max)
        self.use_forecast = bool(use_forecast)
        self.max_episode_steps = int(max_episode_steps)
        self.initial_seed = int(seed)
        self.action_space = spaces.Discrete(self.k_max + 1)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(observation_size(self.k_max, self.use_forecast),),
            dtype=np.float32,
        )
        self.network: WifiNetwork | None = None
        self.state: NetworkState | None = None
        self.candidate_sta_id: int | None = None
        self.episode_steps = 0
        self.rssi_history: deque[tuple[int, dict[int, np.ndarray]]] = deque(maxlen=30)
        self.fairness_history: deque[float] = deque(maxlen=30)
        self._episode_seed = self.initial_seed

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new randomized scenario episode and return its initial observation."""

        super().reset(seed=seed)
        selected = self.scenario_mix[int(self.np_random.integers(len(self.scenario_mix)))]
        self._episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.network = WifiNetwork(selected, seed=self._episode_seed)
        self.state = self.network.snapshot()
        self.episode_steps = 0
        self.rssi_history.clear()
        self.fairness_history.clear()
        self._record_auxiliary_history(self.state)
        self.candidate_sta_id = select_candidate_sta(self.state, selected.handover.dwell_s)
        observation = self._observation()
        return observation, {
            "scenario": selected.name,
            "seed": self._episode_seed,
            "candidate_sta_id": self.candidate_sta_id,
            "action_mask": self.action_masks(),
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute stay/move, advance the simulator, and calculate the SRS reward."""

        if self.network is None or self.state is None:
            raise RuntimeError("Call reset() before step()")
        action_value = int(np.asarray(action).item())
        mask = self.action_masks()
        invalid_action = not (0 <= action_value < len(mask))
        actions: list[Handover] = []
        if not invalid_action and action_value > 0:
            if mask[action_value]:
                target_ap = action_value - 1
                candidate = next(
                    (sta for sta in self.state.stas if sta.sta_id == self.candidate_sta_id), None
                )
                if candidate is not None and candidate.ap_id != target_ap:
                    actions.append(Handover(candidate.sta_id, target_ap, reason="rl_agent"))
            else:
                invalid_action = True
                candidate = next(
                    (sta for sta in self.state.stas if sta.sta_id == self.candidate_sta_id), None
                )
                if candidate is not None and 0 <= action_value - 1 < self.k_max:
                    actions.append(
                        Handover(candidate.sta_id, action_value - 1, reason="invalid_rl_action")
                    )
                else:
                    self.network.invalid_action_events.append(
                        {
                            "t": self.state.t + 1,
                            "sta_id": self.candidate_sta_id,
                            "target_ap": action_value - 1,
                            "reason": "invalid_action",
                        }
                    )

        self.state = self.network.step(actions)
        self.episode_steps += 1
        self._record_auxiliary_history(self.state)
        self.candidate_sta_id = select_candidate_sta(
            self.state, self.network.scenario.handover.dwell_s
        )
        metrics = self.network.last_step_metrics
        invalid_action = bool(invalid_action or metrics.get("invalid_actions", 0))
        aggregate = float(metrics.get("agg_tput_mbps", 0.0))
        demand = float(metrics.get("offered_mbps", 0.0))
        throughput_norm = aggregate / demand if demand > 0 else 0.0
        handover = float(bool(metrics.get("handovers", 0)))
        retx = float(metrics.get("retx_per_frame", 0.0))
        fairness = float(metrics.get("jfi", 0.0))
        reward, components = calculate_reward(
            throughput_norm, fairness, handover, retx, float(invalid_action), self.reward_cfg
        )
        truncated = self.episode_steps >= self.max_episode_steps
        observation = self._observation()
        info = {
            "scenario": self.network.scenario.name,
            "t": self.state.t,
            "candidate_sta_id": self.candidate_sta_id,
            "executed_handovers": int(metrics.get("handovers", 0)),
            "invalid_action": bool(invalid_action),
            "reward_components": components,
            "action_mask": self.action_masks(),
        }
        return observation, reward, False, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return a Boolean mask; stay and the candidate's current AP are always valid."""

        mask = np.zeros(self.k_max + 1, dtype=bool)
        mask[0] = True
        if self.state is None or self.candidate_sta_id is None:
            return mask
        row = next(
            (
                index
                for index, sta in enumerate(self.state.stas)
                if sta.sta_id == self.candidate_sta_id
            ),
            None,
        )
        if row is None:
            return mask
        current_ap = self.state.stas[row].ap_id
        for ap_id in range(min(len(self.state.aps), self.k_max)):
            if ap_id == current_ap or self.state.rssi_dbm[row, ap_id] >= -82.0:
                mask[ap_id + 1] = True
        return mask

    def _record_auxiliary_history(self, state: NetworkState) -> None:
        """Store short RSSI/fairness histories for the candidate observation features."""

        snapshot = {sta.sta_id: state.rssi_dbm[row].copy() for row, sta in enumerate(state.stas)}
        self.rssi_history.append((state.t, snapshot))
        if state.sta_demand_mbps.size:
            ratios = np.divide(
                state.sta_tput_mbps,
                state.sta_demand_mbps,
                out=np.zeros_like(state.sta_tput_mbps),
                where=state.sta_demand_mbps > 0,
            )
            from fa_rlwa.monitor.indicators import jain_fairness

            self.fairness_history.append(jain_fairness(ratios))

    def _observation(self) -> np.ndarray:
        """Assemble the current candidate's feature vector."""

        if self.state is None:
            return np.zeros((observation_size(self.k_max, self.use_forecast),), dtype=np.float32)
        return build_observation(
            self.state,
            k_max=self.k_max,
            use_forecast=self.use_forecast,
            forecaster=self.forecaster,
            candidate_sta_id=self.candidate_sta_id,
            rssi_history=list(self.rssi_history),
            fairness_history=list(self.fairness_history),
        )
