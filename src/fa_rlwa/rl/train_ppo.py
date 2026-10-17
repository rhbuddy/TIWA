"""Train optional PPO policy using the same Gymnasium Wi-Fi association environment."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml

from fa_rlwa.config import load_scenario
from fa_rlwa.env.reward import RewardConfig
from fa_rlwa.env.wifi_env import WifiAssocEnv
from fa_rlwa.forecast.registry import load_forecaster


def _git_commit() -> str | None:
    """Return the current project revision when the source is in a Git checkout."""

    repository = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def train(
    config_path: str | Path, seed: int, timesteps: int | None = None, model_out: str | None = None
):
    """Train and save one PPO model; action invalidity remains penalized by the environment."""

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as exc:
        raise ImportError("Install the RL extra with pip install -r requirements-ml.txt") from exc
    with Path(config_path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    scenarios = [
        load_scenario(path)
        for path in cfg.get("scenario_mix", ["configs/scenarios/sc1_steady.yaml"])
    ]
    use_forecast = bool(cfg.get("use_forecast", True))
    forecaster = (
        load_forecaster(cfg["forecaster"]) if use_forecast and cfg.get("forecaster") else None
    )
    env = make_vec_env(
        lambda: WifiAssocEnv(
            scenarios,
            reward_cfg=RewardConfig(**cfg.get("reward", {})),
            forecaster=forecaster,
            k_max=int(cfg.get("k_max", 5)),
            use_forecast=use_forecast,
            max_episode_steps=int(cfg.get("episode_steps", 600)),
            seed=seed,
        ),
        n_envs=int(cfg.get("n_envs", 1)),
        seed=seed,
    )
    ppo = cfg.get("ppo", {})
    output = Path(
        model_out or f"models/rl/ppo_{'forecast' if use_forecast else 'noforecast'}_s{seed}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.get("log_dir", "results/rl/logs")) / f"ppo_s{seed}"
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=float(ppo.get("learning_rate", 3e-4)),
        gamma=float(ppo.get("gamma", 0.95)),
        n_steps=int(ppo.get("n_steps", 2048)),
        batch_size=int(ppo.get("batch_size", 64)),
        n_epochs=int(ppo.get("n_epochs", 10)),
        gae_lambda=float(ppo.get("gae_lambda", 0.95)),
        clip_range=float(ppo.get("clip_range", 0.2)),
        ent_coef=float(ppo.get("ent_coef", 0.01)),
        policy_kwargs={"net_arch": list(ppo.get("net_arch", [128, 128]))},
        seed=seed,
        verbose=1,
        tensorboard_log=str(log_dir),
    )
    total = int(timesteps or cfg.get("total_timesteps", 2_000_000))
    model.learn(total_timesteps=total, progress_bar=False)
    model.save(str(output))
    final_checkpoint = Path(str(output) if output.suffix == ".zip" else f"{output}.zip")
    metadata = {
        "algorithm": "PPO",
        "seed": int(seed),
        "total_timesteps": total,
        "scenario_mix": [scenario.name for scenario in scenarios],
        "use_forecast": use_forecast,
        "forecaster": str(cfg.get("forecaster", "EWMA")),
        "k_max": int(cfg.get("k_max", 5)),
        "reward": cfg.get("reward", {}),
        "hyperparameters": ppo,
        "checkpoint": str(final_checkpoint),
        "git_commit": _git_commit(),
    }
    metadata_path = output.with_name(f"{output.stem}_meta.yaml")
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    env.close()
    return output


def main() -> None:
    """Command-line PPO trainer."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--model-out")
    args = parser.parse_args()
    train(args.config, args.seed, args.timesteps, args.model_out)


if __name__ == "__main__":
    main()
