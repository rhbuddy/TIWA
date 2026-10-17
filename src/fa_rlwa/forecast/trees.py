"""Gradient-boosted/random-forest tabular forecasters (scikit-learn optional)."""

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


def _tree_features(windows: np.ndarray) -> np.ndarray:
    """Flatten lag windows and add the rolling summaries specified by the SRS."""

    flat = windows.reshape(windows.shape[0], -1)
    summaries: list[np.ndarray] = []
    for width in (5, 10, 30):
        recent = windows[:, -width:, :]
        summaries.extend([recent.mean(axis=1), recent.std(axis=1)])
    summaries.append(windows[:, -1, :] - windows[:, 0, :])
    return np.concatenate([flat, *summaries], axis=1).astype(np.float32)


class RandomForestForecaster(Forecaster):
    """Multi-output random-forest forecast model with one output per horizon/indicator."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 8,
        random_state: int = 0,
        horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.random_state = int(random_state)
        self.horizons = tuple(horizons)
        self.model: Any = None

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> RandomForestForecaster:
        """Fit a sklearn random forest; sklearn is an optional forecast dependency."""

        if y is None:
            raise ValueError("RandomForestForecaster.fit requires targets y")
        features, targets = validate_training_data(x, y)
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as exc:
            raise ImportError(
                "Install the forecast extra: pip install -r requirements-ml.txt"
            ) from exc
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(_tree_features(features), targets.reshape(len(targets), -1))
        return self

    def predict(self, history: np.ndarray) -> np.ndarray:
        """Predict the 5/10/30-second targets for one or more AP histories."""

        if self.model is None:
            raise RuntimeError("RandomForestForecaster must be fitted or loaded before predict")
        values, single = validate_history(history)
        prediction = self.model.predict(_tree_features(values)).reshape(
            len(values), len(self.horizons), 4
        )
        return prediction[0] if single else prediction

    def save(self, path: str | Path) -> None:
        """Persist the fitted sklearn model and its configuration."""

        if self.model is None:
            raise RuntimeError("Cannot save an unfitted random-forest model")
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("Install scikit-learn to save a random-forest model") from exc
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, directory / "model.pkl")
        (directory / "model.json").write_text(
            json.dumps(
                {
                    "model": "random_forest",
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "random_state": self.random_state,
                    "horizons": self.horizons,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> RandomForestForecaster:
        """Load a random forest from a directory created by :meth:`save`."""

        directory = Path(path)
        metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        model = cls(
            int(metadata["n_estimators"]),
            int(metadata["max_depth"]),
            int(metadata["random_state"]),
            tuple(metadata.get("horizons", DEFAULT_HORIZONS)),
        )
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("Install scikit-learn to load a random-forest model") from exc
        model.model = joblib.load(directory / "model.pkl")
        return model


# SRS calls the gradient-boosted model XGBoost; RF is the documented fallback when that
# package is unavailable. XGBoost's sklearn API has the same window/target contract.
class XGBoostForecaster(RandomForestForecaster):
    """Multi-output XGBoost forecaster, trained as 12 compact horizon/indicator models."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 0,
        horizons: tuple[int, ...] = DEFAULT_HORIZONS,
        subsample: float = 0.8,
        early_stopping_rounds: int = 20,
    ) -> None:
        super().__init__(n_estimators, max_depth, random_state, horizons)
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.models: list[list[Any]] = []

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> XGBoostForecaster:
        """Train 12 regressors; use the validation split for SRS early stopping when supplied."""

        if y is None:
            raise ValueError("XGBoostForecaster.fit requires targets y")
        features, targets = validate_training_data(x, y)
        validation_matrix = None
        validation_targets = None
        if validation_data is not None:
            val_features, validation_targets = validate_training_data(*validation_data)
            validation_matrix = _tree_features(val_features)
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("Install XGBoost with pip install -r requirements-ml.txt") from exc
        matrix = _tree_features(features)
        self.models = []
        for horizon in range(targets.shape[1]):
            row = []
            for target in range(targets.shape[2]):
                estimator = XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    subsample=self.subsample,
                    objective="reg:squarederror",
                    random_state=self.random_state,
                    n_jobs=-1,
                    early_stopping_rounds=(
                        self.early_stopping_rounds if validation_matrix is not None else None
                    ),
                )
                fit_kwargs: dict[str, Any] = {}
                if validation_matrix is not None and validation_targets is not None:
                    fit_kwargs["eval_set"] = [
                        (validation_matrix, validation_targets[:, horizon, target])
                    ]
                    fit_kwargs["verbose"] = False
                estimator.fit(matrix, targets[:, horizon, target], **fit_kwargs)
                row.append(estimator)
            self.models.append(row)
        return self

    def predict(self, history: np.ndarray) -> np.ndarray:
        """Return XGBoost predictions in ``[AP, horizon, indicator]`` order."""

        if not self.models:
            raise RuntimeError("XGBoostForecaster must be fitted or loaded before predict")
        values, single = validate_history(history)
        matrix = _tree_features(values)
        output = np.empty((len(values), len(self.horizons), 4), dtype=float)
        for horizon, row in enumerate(self.models):
            for target, model in enumerate(row):
                output[:, horizon, target] = model.predict(matrix)
        return output[0] if single else output

    def save(self, path: str | Path) -> None:
        """Save XGBoost boosters and metadata; model artifacts are git-ignored."""

        if not self.models:
            raise RuntimeError("Cannot save an unfitted XGBoost model")
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        for horizon, row in enumerate(self.models):
            for target, model in enumerate(row):
                model.save_model(directory / f"h{horizon}_i{target}.json")
        (directory / "model.json").write_text(
            json.dumps(
                {
                    "model": "xgboost",
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "learning_rate": self.learning_rate,
                    "random_state": self.random_state,
                    "subsample": self.subsample,
                    "early_stopping_rounds": self.early_stopping_rounds,
                    "horizons": self.horizons,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> XGBoostForecaster:
        """Load a saved set of XGBoost horizon/indicator models."""

        directory = Path(path)
        metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("Install XGBoost with pip install -r requirements-ml.txt") from exc
        result = cls(
            int(metadata["n_estimators"]),
            int(metadata["max_depth"]),
            float(metadata["learning_rate"]),
            int(metadata["random_state"]),
            tuple(metadata.get("horizons", DEFAULT_HORIZONS)),
            float(metadata.get("subsample", 0.8)),
            int(metadata.get("early_stopping_rounds", 20)),
        )
        result.models = []
        for h in range(len(result.horizons)):
            row = []
            for i in range(4):
                estimator = XGBRegressor()
                estimator.load_model(directory / f"h{h}_i{i}.json")
                row.append(estimator)
            result.models.append(row)
        return result
