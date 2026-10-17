"""Stable-Baselines3 policy adapter for the experiment-runner interface."""

from __future__ import annotations

import zipfile
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from fa_rlwa.env.obs import build_observation, select_candidate_sta
from fa_rlwa.forecast.base import Forecaster
from fa_rlwa.forecast.ewma import EWMAForecaster
from fa_rlwa.policies.base import Policy
from fa_rlwa.types import Action, Handover, NetworkState


class RLPolicy(Policy):
    """Load a trained discrete policy and translate actions into controller handovers."""

    name = "rl_forecast"

    def __init__(
        self,
        model_path: str | Path,
        k_max: int = 5,
        use_forecast: bool = True,
        forecaster: Forecaster | None = None,
    ) -> None:
        try:
            from stable_baselines3 import DQN, PPO
        except ImportError as exc:
            raise ImportError(
                "Install the RL extra with pip install -r requirements-ml.txt"
            ) from exc
        path = str(model_path)
        try:
            self.model: Any = DQN.load(path, device="cpu")
        except (FileNotFoundError, ValueError, zipfile.BadZipFile):
            # PPO checkpoints are supported as an alternative when DQN loading rejects the archive.
            self.model = PPO.load(path, device="cpu")
        self.k_max = int(k_max)
        self.use_forecast = bool(use_forecast)
        self.forecaster = forecaster or (EWMAForecaster() if use_forecast else None)
        self.rssi_history: deque[tuple[int, dict[int, np.ndarray]]] = deque(maxlen=30)
        self.fairness_history: deque[float] = deque(maxlen=30)
        self._last_recorded_t: int | None = None

    def reset(self, state: NetworkState, rng: np.random.Generator) -> None:
        """Clear per-episode observation histories."""

        self.rssi_history.clear()
        self.fairness_history.clear()
        self._last_recorded_t = None
        self._record(state)

    def _record(self, state: NetworkState) -> None:
        """Append RSSI snapshots and current Jain fairness for the observation builder."""

        if self._last_recorded_t == state.t:
            return
        self._last_recorded_t = state.t
        self.rssi_history.append(
            (
                state.t,
                {sta.sta_id: state.rssi_dbm[row].copy() for row, sta in enumerate(state.stas)},
            )
        )
        ratios = np.divide(
            state.sta_tput_mbps,
            state.sta_demand_mbps,
            out=np.zeros_like(state.sta_tput_mbps),
            where=state.sta_demand_mbps > 0,
        )
        if ratios.size:
            total = float(ratios.sum())
            squares = float(np.square(ratios).sum())
            self.fairness_history.append(
                total * total / (len(ratios) * squares) if squares else 0.0
            )

    def decide(self, state: NetworkState) -> list[Action]:
        """Query the learned policy greedily for the selected candidate STA."""

        self._record(state)
        candidate = select_candidate_sta(state)
        if candidate is None:
            return []
        observation = build_observation(
            state,
            k_max=self.k_max,
            use_forecast=self.use_forecast,
            forecaster=self.forecaster,
            candidate_sta_id=candidate,
            rssi_history=list(self.rssi_history),
            fairness_history=list(self.fairness_history),
        )
        action, _ = self.model.predict(observation, deterministic=True)
        action_id = int(np.asarray(action).item())
        if action_id <= 0 or action_id > self.k_max:
            return []
        target = action_id - 1
        row = next(index for index, sta in enumerate(state.stas) if sta.sta_id == candidate)
        sta = state.stas[row]
        if target == sta.ap_id or target >= len(state.aps) or state.rssi_dbm[row, target] < -82.0:
            return []
        return [Handover(candidate, target, reason="trained_rl")]
