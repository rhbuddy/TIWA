"""Optional two-layer LSTM AP-indicator forecaster (PyTorch)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fa_rlwa.forecast.base import (
    DEFAULT_HORIZONS,
    Forecaster,
    validate_history,
    validate_training_data,
)


class LSTMForecaster(Forecaster):
    """Two-layer, 64-unit LSTM with a 12-value forecast head."""

    def __init__(
        self,
        hidden_size: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 128,
        seed: int = 0,
        horizons: tuple[int, ...] = DEFAULT_HORIZONS,
        patience: int = 10,
    ) -> None:
        self.hidden_size = int(hidden_size)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.horizons = tuple(horizons)
        self.patience = int(patience)
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_std: np.ndarray | None = None
        self.model: Any = None

    def _new_model(self):
        """Build the small CPU-friendly PyTorch network."""

        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError("Install PyTorch with pip install -r requirements-ml.txt") from exc

        class Network(nn.Module):
            def __init__(self, hidden: int, layers: int, dropout: float) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=6,
                    hidden_size=hidden,
                    num_layers=layers,
                    dropout=dropout if layers > 1 else 0.0,
                    batch_first=True,
                )
                self.head = nn.Linear(hidden, 12)

            def forward(self, batch):
                encoded, _ = self.lstm(batch)
                return self.head(encoded[:, -1, :])

        torch.manual_seed(self.seed)
        return Network(self.hidden_size, self.layers, self.dropout)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> LSTMForecaster:
        """Train on z-scored data, using the held-out scenario split for early stopping."""

        if y is None:
            raise ValueError("LSTMForecaster.fit requires targets y")
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError("Install PyTorch with pip install -r requirements-ml.txt") from exc
        features, targets = validate_training_data(x, y)
        if validation_data is None:
            permutation = np.random.default_rng(self.seed).permutation(len(features))
            split = max(1, int(len(permutation) * 0.9))
            train_idx, val_idx = permutation[:split], permutation[split:]
            if not len(val_idx):
                val_idx = train_idx[-1:]
            train_features = features[train_idx]
            train_targets = targets[train_idx]
            val_features = features[val_idx]
            val_targets = targets[val_idx]
        else:
            val_features, val_targets = validate_training_data(*validation_data)
            train_features, train_targets = features, targets
        self.x_mean = train_features.mean(axis=(0, 1), keepdims=True)
        self.x_std = np.maximum(train_features.std(axis=(0, 1), keepdims=True), 1e-6)
        flattened_train_targets = train_targets.reshape(len(train_targets), -1)
        flattened_val_targets = val_targets.reshape(len(val_targets), -1)
        self.y_mean = flattened_train_targets.mean(axis=0, keepdims=True)
        self.y_std = np.maximum(flattened_train_targets.std(axis=0, keepdims=True), 1e-6)
        train_x = torch.tensor((train_features - self.x_mean) / self.x_std, dtype=torch.float32)
        train_y = torch.tensor(
            (flattened_train_targets - self.y_mean) / self.y_std, dtype=torch.float32
        )
        val_x = torch.tensor((val_features - self.x_mean) / self.x_std, dtype=torch.float32)
        val_y = torch.tensor(
            (flattened_val_targets - self.y_mean) / self.y_std, dtype=torch.float32
        )
        self.model = self._new_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        best_loss, best_state, patience = float("inf"), None, 0
        for _ in range(self.epochs):
            self.model.train()
            order = torch.randperm(len(train_x))
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                optimizer.zero_grad()
                loss = criterion(self.model(train_x[batch]), train_y[batch])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                val_loss = float(criterion(self.model(val_x), val_y).item())
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = {
                    key: value.detach().clone() for key, value in self.model.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
                if patience >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict(self, history: np.ndarray) -> np.ndarray:
        """Predict normalized-history batches and return values in their original units."""

        try:
            import torch
        except ImportError as exc:
            raise ImportError("Install PyTorch with pip install -r requirements-ml.txt") from exc
        if self.model is None or any(
            item is None for item in (self.x_mean, self.x_std, self.y_mean, self.y_std)
        ):
            raise RuntimeError("LSTMForecaster must be fitted or loaded before predict")
        values, single = validate_history(history)
        x_scaled = (values - self.x_mean) / self.x_std
        self.model.eval()
        with torch.no_grad():
            pred = self.model(torch.tensor(x_scaled, dtype=torch.float32)).cpu().numpy()
        output = (pred * self.y_std + self.y_mean).reshape(len(values), len(self.horizons), 4)
        return output[0] if single else output

    def save(self, path: str | Path) -> None:
        """Persist model weights, scaling statistics, and training configuration."""

        if self.model is None or any(
            item is None for item in (self.x_mean, self.x_std, self.y_mean, self.y_std)
        ):
            raise RuntimeError("Cannot save an unfitted LSTM")
        import torch

        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "x_mean": self.x_mean,
                "x_std": self.x_std,
                "y_mean": self.y_mean,
                "y_std": self.y_std,
            },
            directory / "model.pt",
        )
        (directory / "model.json").write_text(
            json.dumps(
                {
                    "model": "lstm",
                    "hidden_size": self.hidden_size,
                    "layers": self.layers,
                    "dropout": self.dropout,
                    "learning_rate": self.learning_rate,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "seed": self.seed,
                    "patience": self.patience,
                    "horizons": self.horizons,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> LSTMForecaster:
        """Load PyTorch weights and the fitted scaling metadata."""

        try:
            import torch
        except ImportError as exc:
            raise ImportError("Install PyTorch with pip install -r requirements-ml.txt") from exc
        directory = Path(path)
        metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        model = cls(
            hidden_size=int(metadata["hidden_size"]),
            layers=int(metadata["layers"]),
            dropout=float(metadata["dropout"]),
            learning_rate=float(metadata["learning_rate"]),
            epochs=int(metadata["epochs"]),
            batch_size=int(metadata["batch_size"]),
            seed=int(metadata["seed"]),
            horizons=tuple(metadata.get("horizons", DEFAULT_HORIZONS)),
            patience=int(metadata.get("patience", 10)),
        )
        checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
        model.model = model._new_model()
        model.model.load_state_dict(checkpoint["state_dict"])
        for attribute in ("x_mean", "x_std", "y_mean", "y_std"):
            setattr(model, attribute, checkpoint[attribute])
        model.model.eval()
        return model
