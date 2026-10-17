"""Train one configured forecaster and save a versioned model directory."""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
from pathlib import Path

import numpy as np
import yaml

from fa_rlwa.forecast.arima import ARIMAForecaster
from fa_rlwa.forecast.ewma import EWMAForecaster, MovingAverageForecaster
from fa_rlwa.forecast.lstm import LSTMForecaster
from fa_rlwa.forecast.trees import RandomForestForecaster, XGBoostForecaster


def _load_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read finite numeric training arrays from a dataset archive."""

    with np.load(path, allow_pickle=False) as archive:
        return archive["X"].astype(np.float32), archive["y"].astype(np.float32)


def _git_commit() -> str | None:
    """Return the repository revision used for training, when Git is available."""

    repository = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _build_model(name: str, config: dict, seed: int):
    """Construct a forecaster from its name and optional YAML parameters."""

    if name == "xgb":
        name = "xgboost"
    if name == "rf":
        name = "random_forest"
    if name == "ewma":
        return EWMAForecaster(float(config.get("alpha", 0.3)))
    if name == "moving_average":
        return MovingAverageForecaster(int(config.get("window", 10)))
    if name == "arima":
        return ARIMAForecaster(
            tuple(config.get("order", [1, 0, 0])),
            auto_order=bool(config.get("auto_order", True)),
            refit_interval_s=int(config.get("refit_interval_s", 60)),
            history_window_s=int(config.get("history_window_s", 300)),
        )
    if name == "random_forest":
        return RandomForestForecaster(
            int(config.get("n_estimators", 300)),
            int(config.get("max_depth", 8)),
            seed,
        )
    if name == "xgboost":
        return XGBoostForecaster(
            n_estimators=int(config.get("n_estimators", 300)),
            max_depth=int(config.get("max_depth", 6)),
            learning_rate=float(config.get("learning_rate", 0.05)),
            random_state=seed,
            subsample=float(config.get("subsample", 0.8)),
            early_stopping_rounds=int(config.get("early_stopping_rounds", 20)),
        )
    if name == "lstm":
        allowed = {
            "hidden_size",
            "layers",
            "dropout",
            "learning_rate",
            "epochs",
            "batch_size",
            "patience",
        }
        return LSTMForecaster(
            seed=seed, **{key: value for key, value in config.items() if key in allowed}
        )
    raise ValueError(f"Unsupported forecaster {name!r}")


def train(
    model_name: str,
    config_path: str | Path,
    train_data: str | Path,
    val_data: str | Path,
    output_dir: str | Path,
    seed: int = 0,
):
    """Fit the chosen model, select EWMA alpha on validation, and write metadata."""

    with Path(config_path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    normalized_name = {"xgb": "xgboost", "rf": "random_forest"}.get(model_name, model_name)
    model_cfg = cfg.get(model_name, cfg.get(normalized_name, cfg))
    x_train, y_train = _load_npz(train_data)
    x_val, y_val = _load_npz(val_data)
    model = _build_model(model_name, model_cfg, seed)
    if model_name == "ewma":
        model.fit(x_val, y_val)
    elif isinstance(model, XGBoostForecaster | LSTMForecaster):
        model.fit(x_train, y_train, validation_data=(x_val, y_val))
    else:
        model.fit(x_train, y_train)
    prediction = model.predict(x_val)
    val_mae = float(np.mean(np.abs(prediction - y_val)))
    val_rmse = float(np.sqrt(np.mean(np.square(prediction - y_val))))
    output = Path(output_dir)
    model.save(output)
    meta = {
        "model": model_name,
        "seed": int(seed),
        "git_commit": _git_commit(),
        "config": model_cfg,
        "training_data": str(Path(train_data)),
        "validation_data": str(Path(val_data)),
        "train_samples": int(len(x_train)),
        "validation_samples": int(len(x_val)),
        "validation_mae": val_mae,
        "validation_rmse": val_rmse,
        "features": ["rssi_dbm", "loss_frac", "tput_mbps", "occ_frac", "n_clients", "demand_mbps"],
        "targets": ["rssi_dbm", "loss_frac", "tput_mbps", "occ_frac"],
        "horizons_s": list(model.horizons),
        "numpy_version": np.__version__,
    }
    scaling = {
        name: np.asarray(value).tolist()
        for name in ("x_mean", "x_std", "y_mean", "y_std")
        if (value := getattr(model, name, None)) is not None
    }
    if scaling:
        meta["scaling"] = scaling
    try:
        meta["model_library_version"] = importlib.metadata.version(
            {
                "arima": "statsmodels",
                "random_forest": "scikit-learn",
                "xgboost": "xgboost",
                "lstm": "torch",
            }.get(normalized_name, "numpy")
        )
    except importlib.metadata.PackageNotFoundError:
        pass
    (output / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    return meta


def main() -> None:
    """CLI entry point for model training."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        choices=["ewma", "moving_average", "arima", "random_forest", "xgboost", "xgb", "lstm"],
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-data", default="data/forecast/v1/train.npz")
    parser.add_argument("--val-data", default="data/forecast/v1/val.npz")
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = args.out or f"models/forecast/{args.model}_v1"
    result = train(args.model, args.config, args.train_data, args.val_data, out, args.seed)
    print(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()
