"""Gymnasium observation/action/reward behavior."""

import numpy as np
from gymnasium.utils.env_checker import check_env

from fa_rlwa.config import scenario_from_mapping
from fa_rlwa.env.wifi_env import WifiAssocEnv


def env_scenario():
    return scenario_from_mapping(
        {
            "name": "env_test",
            "area_m": [10, 10],
            "duration_s": 30,
            "warmup_s": 0,
            "channels": [1, 6, 11],
            "aps": [
                {"id": 0, "pos": [0, 5], "channel": 1},
                {"id": 1, "pos": [10, 5], "channel": 6},
                {"id": 2, "pos": [5, 9], "channel": 11},
            ],
            "background_stas": {
                "count": 8,
                "placement": "uniform",
                "mobility": "static",
                "traffic": {"type": "cbr", "mbps": 1.0},
            },
            "measurement_noise": False,
        }
    )


def test_observation_shapes_match_srs() -> None:
    with_forecast = WifiAssocEnv(env_scenario(), k_max=5, use_forecast=True, seed=1)
    obs, _ = with_forecast.reset(seed=1)
    assert obs.shape == (64,)
    assert with_forecast.action_space.n == 6
    without_forecast = WifiAssocEnv(env_scenario(), k_max=5, use_forecast=False, seed=1)
    obs_no, _ = without_forecast.reset(seed=1)
    assert obs_no.shape == (44,)
    assert np.all((obs >= 0.0) & (obs <= 1.0))
    assert np.all((obs_no >= 0.0) & (obs_no <= 1.0))


def test_env_step_is_finite_and_invalid_padded_ap_is_penalized() -> None:
    env = WifiAssocEnv(env_scenario(), k_max=5, use_forecast=True, max_episode_steps=2, seed=4)
    observation, info = env.reset(seed=4)
    assert not info["action_mask"][5]  # AP 4 is padding for this three-AP scenario.
    observation, reward, terminated, truncated, info = env.step(5)
    assert observation.shape == (64,)
    assert np.isfinite(reward)
    assert not terminated
    assert info["invalid_action"]
    assert truncated is False
    _, _, _, truncated, _ = env.step(0)
    assert truncated


def test_gymnasium_env_checker() -> None:
    env = WifiAssocEnv(env_scenario(), k_max=5, use_forecast=False, max_episode_steps=5, seed=9)
    check_env(env, skip_render_check=True)
