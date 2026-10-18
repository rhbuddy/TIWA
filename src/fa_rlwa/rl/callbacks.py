"""Stable-Baselines3 callbacks for checkpointing and episodic logging."""

from __future__ import annotations

from pathlib import Path


def checkpoint_callback(path: str | Path, save_freq: int):
    """Build a periodic checkpoint callback (imports SB3 only on demand)."""

    try:
        from stable_baselines3.common.callbacks import CheckpointCallback
    except ImportError as exc:
        raise ImportError(
            "Install Stable-Baselines3 with pip install -r requirements-ml.txt"
        ) from exc
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return CheckpointCallback(
        save_freq=max(1, int(save_freq)), save_path=str(directory), name_prefix="dqn"
    )
