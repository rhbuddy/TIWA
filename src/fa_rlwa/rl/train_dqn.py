"""Train the primary DQN policy from a YAML experiment configuration."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

import yaml

from fa_rlwa.config import load_scenario
from fa_rlwa.env.reward import RewardConfig
from fa_rlwa.env.wifi_env import WifiAssocEnv
from fa_rlwa.forecast.registry import load_forecaster
from fa_rlwa.rl.callbacks import checkpoint_callback

LOGGER = logging.getLogger(__name__)


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
    """Run one DQN seed and save the best/final checkpoints."""

    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.env_util import make_vec_env
    except ImportError as exc:
        raise ImportError("Install the RL extra with pip install -r requirements-ml.txt") from exc
    with Path(config_path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    paths = cfg.get("scenario_mix", ["configs/scenarios/sc1_steady.yaml"])
    scenarios = [load_scenario(path) for path in paths]
    rewards = RewardConfig(**cfg.get("reward", {}))
    use_forecast = bool(cfg.get("use_forecast", True))
    forecaster = (
        load_forecaster(cfg["forecaster"]) if use_forecast and cfg.get("forecaster") else None
    )
    episode_steps = int(cfg.get("episode_steps", 600))
    k_max = int(cfg.get("k_max", 5))
    env = make_vec_env(
        lambda: WifiAssocEnv(
            scenarios,
            reward_cfg=rewards,
            forecaster=forecaster,
            k_max=k_max,
            use_forecast=use_forecast,
            max_episode_steps=episode_steps,
            seed=seed,
        ),
        n_envs=int(cfg.get("n_envs", 1)),
        seed=seed,
    )
    eval_env = make_vec_env(
        lambda: WifiAssocEnv(
            scenarios,
            reward_cfg=rewards,
            forecaster=forecaster,
            k_max=k_max,
            use_forecast=use_forecast,
            max_episode_steps=episode_steps,
            seed=seed + 10_000,
        ),
        n_envs=1,
        seed=seed + 10_000,
    )
    dqn = cfg.get("dqn", {})
    output = Path(
        model_out or f"models/rl/dqn_{'forecast' if use_forecast else 'noforecast'}_s{seed}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    experiment_tag = output.stem
    log_dir = Path(cfg.get("log_dir", "results/rl/logs")) / experiment_tag
    callback = checkpoint_callback(
        output.parent / "checkpoints" / experiment_tag, int(cfg.get("eval_every", 20_000))
    )
    best_dir = output.parent / "best" / experiment_tag
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(log_dir),
        eval_freq=int(cfg.get("eval_every", 20_000)),
        n_eval_episodes=int(cfg.get("eval_episodes", 10)),
        deterministic=True,
        render=False,
    )
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=float(dqn.get("learning_rate", 1e-4)),
        buffer_size=int(dqn.get("buffer_size", 100_000)),
        learning_starts=int(dqn.get("learning_starts", 5_000)),
        batch_size=int(dqn.get("batch_size", 64)),
        train_freq=int(dqn.get("train_freq", 4)),
        target_update_interval=int(dqn.get("target_update_interval", 1_000)),
        exploration_fraction=float(dqn.get("exploration_fraction", 0.3)),
        exploration_final_eps=float(dqn.get("exploration_final_eps", 0.05)),
        gamma=float(dqn.get("gamma", 0.95)),
        policy_kwargs={"net_arch": list(dqn.get("net_arch", [128, 128]))},
        seed=seed,
        verbose=1,
        tensorboard_log=str(log_dir),
    )
    total = int(timesteps or cfg.get("total_timesteps", 1_500_000))
    model.learn(total_timesteps=total, callback=[callback, eval_callback], progress_bar=False)
    final_checkpoint = Path(str(output) if output.suffix == ".zip" else f"{output}.zip")
    best_checkpoint = best_dir / "best_model.zip"
    if best_checkpoint.exists():
        final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_checkpoint, final_checkpoint)
    else:
        model.save(str(output))
    metadata = {
        "algorithm": "DQN",
        "seed": int(seed),
        "total_timesteps": total,
        "scenario_mix": [str(path) for path in paths],
        "use_forecast": use_forecast,
        "forecaster": str(cfg.get("forecaster", "EWMA")),
        "k_max": k_max,
        "reward": cfg.get("reward", {}),
        "hyperparameters": dqn,
        "checkpoint": str(final_checkpoint),
        "git_commit": _git_commit(),
    }
    metadata_path = output.with_name(f"{output.stem}_meta.yaml")
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    env.close()
    eval_env.close()
    LOGGER.info("Saved DQN checkpoint to %s", output)
    return output


def main() -> None:
    """Command-line trainer."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--model-out")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    train(args.config, args.seed, args.timesteps, args.model_out)


if __name__ == "__main__":
    main()
