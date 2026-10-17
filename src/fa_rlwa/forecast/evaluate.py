"""Evaluate saved forecasters by horizon, indicator, scenario split, and event window."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fa_rlwa.experiments.stats import bootstrap_mean_ci, holm_adjust, paired_wilcoxon
from fa_rlwa.forecast.registry import load_forecaster

TARGETS = ("rssi_dbm", "loss_frac", "tput_mbps", "occ_frac")


def _load_evaluation_data(
    dataset_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read features, targets, event labels, and run IDs from a forecast archive."""

    with np.load(dataset_path, allow_pickle=False) as archive:
        x = archive["X"].astype(np.float32)
        y = archive["y"].astype(np.float32)
        event = (
            archive["event_window"].astype(bool)
            if "event_window" in archive
            else np.zeros(len(x), dtype=bool)
        )
        run_ids = (
            archive["run_id"].astype(str)
            if "run_id" in archive
            else np.full(len(x), Path(dataset_path).stem, dtype="U128")
        )
    if len(x) != len(y) or len(x) != len(event) or len(x) != len(run_ids):
        raise ValueError("forecast archive arrays must contain the same number of samples")
    return x, y, event, run_ids


def _metric_rows(
    model_name: str,
    dataset_name: str,
    model: Any,
    dataset_path: str | Path,
    *,
    by_run_errors: dict[tuple[str, str, int, str], dict[str, dict[str, float]]] | None = None,
) -> list[dict[str, Any]]:
    """Calculate MAE/RMSE/MAPE for all samples and event windows."""

    x, y, event_mask, run_ids = _load_evaluation_data(dataset_path)
    prediction = np.asarray(model.predict(x), dtype=float)
    if prediction.shape != y.shape:
        raise ValueError(f"model output shape {prediction.shape} does not match targets {y.shape}")
    rows: list[dict[str, Any]] = []
    scopes = {"all": np.ones(len(x), dtype=bool), "event_window": event_mask}
    for scope, sample_mask in scopes.items():
        for horizon_index, horizon in enumerate(model.horizons):
            for target_index, target in enumerate(TARGETS):
                selected = sample_mask
                errors = prediction[selected, horizon_index, target_index] - y[
                    selected, horizon_index, target_index
                ]
                actual = y[selected, horizon_index, target_index]
                mae: float | None
                rmse: float | None
                mape: float | None
                if len(errors):
                    mae = float(np.mean(np.abs(errors)))
                    rmse = float(np.sqrt(np.mean(np.square(errors))))
                    mape = float(
                        np.mean(np.abs(errors) / np.maximum(np.abs(actual), 1e-8)) * 100.0
                    )
                else:
                    mae = rmse = mape = None
                rows.append(
                    {
                        "model": model_name,
                        "dataset": dataset_name,
                        "scope": scope,
                        "horizon_s": int(horizon),
                        "indicator": target,
                        "n_samples": int(len(errors)),
                        "mae": mae,
                        "rmse": rmse,
                        "mape_pct": mape,
                    }
                )
                if by_run_errors is not None and len(errors):
                    key = (dataset_name, scope, int(horizon), target)
                    per_run: dict[str, float] = {}
                    selected_run_ids = run_ids[selected]
                    absolute_errors = np.abs(errors)
                    for run_id in np.unique(selected_run_ids):
                        run_mask = selected_run_ids == run_id
                        per_run[str(run_id)] = float(np.mean(absolute_errors[run_mask]))
                    by_run_errors.setdefault(key, {})[model_name] = per_run
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV, using the first row for stable field ordering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    model_dir: str | Path, dataset_path: str | Path, output_path: str | Path
) -> list[dict[str, Any]]:
    """Evaluate one saved forecaster and write all/event-window metric rows."""

    model = load_forecaster(model_dir)
    rows = _metric_rows(Path(model_dir).name, Path(dataset_path).stem, model, dataset_path)
    _write_csv(Path(output_path), rows)
    return rows


def evaluate_models(
    model_dirs: Sequence[str | Path],
    datasets: Sequence[str | Path],
    output_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate model directories and write paired per-run MAE tests with Holm correction."""

    metric_rows: list[dict[str, Any]] = []
    per_run_errors: dict[tuple[str, str, int, str], dict[str, dict[str, float]]] = {}
    for dataset in datasets:
        for model_dir in model_dirs:
            model = load_forecaster(model_dir)
            metric_rows.extend(
                _metric_rows(
                    Path(model_dir).name,
                    Path(dataset).stem,
                    model,
                    dataset,
                    by_run_errors=per_run_errors,
                )
            )
    comparisons: list[dict[str, Any]] = []
    for (dataset, scope, horizon, indicator), model_results in sorted(per_run_errors.items()):
        model_names = sorted(model_results)
        for index, model_a in enumerate(model_names):
            for model_b in model_names[index + 1 :]:
                a, b = model_results[model_a], model_results[model_b]
                runs = sorted(set(a) & set(b))
                if len(runs) < 2:
                    continue
                errors_a = [a[run] for run in runs]
                errors_b = [b[run] for run in runs]
                p_value = paired_wilcoxon(errors_a, errors_b)
                ci_low, ci_high = bootstrap_mean_ci(
                    [left - right for left, right in zip(errors_a, errors_b, strict=True)]
                )
                mean_a, mean_b = float(np.mean(errors_a)), float(np.mean(errors_b))
                comparisons.append(
                    {
                        "dataset": dataset,
                        "scope": scope,
                        "horizon_s": horizon,
                        "indicator": indicator,
                        "model_a": model_a,
                        "model_b": model_b,
                        "n_pairs": len(runs),
                        "mean_mae_a": mean_a,
                        "mean_mae_b": mean_b,
                        "mae_improvement_pct_b_vs_a": (
                            (mean_a - mean_b) / abs(mean_a) * 100.0 if mean_a else 0.0
                        ),
                        "paired_difference_ci95_low": ci_low,
                        "paired_difference_ci95_high": ci_high,
                        "p_value": p_value,
                    }
                )
    if comparisons:
        adjusted = holm_adjust([float(row["p_value"]) for row in comparisons])
        for row, p_value in zip(comparisons, adjusted, strict=True):
            row["p_holm"] = float(p_value)
    output = Path(output_path)
    comparison_path = output.with_name(f"{output.stem}_paired_tests.csv")
    _write_csv(output, metric_rows)
    _write_csv(comparison_path, comparisons)
    return metric_rows, comparisons


def _resolve_models(value: list[str], model_root: Path) -> list[Path]:
    """Resolve explicit model paths or the ``all`` shorthand."""

    if value == ["all"]:
        paths = sorted(path for path in model_root.iterdir() if (path / "model.json").is_file())
    else:
        paths = [Path(item) for item in value]
    if not paths:
        raise FileNotFoundError(f"No saved forecast models found under {model_root}")
    return paths


def main() -> None:
    """CLI entry point for one-model or comparative forecast evaluation."""

    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--model", help="one saved model directory")
    choice.add_argument("--models", nargs="+", help="model directories or 'all'")
    parser.add_argument("--test-data", default="data/forecast/v1/test_id.npz")
    parser.add_argument("--test-unseen", help="optional unseen-scenario test archive")
    parser.add_argument("--model-root", default="models/forecast")
    parser.add_argument("--out", default="results/forecast/forecast_metrics.csv")
    args = parser.parse_args()
    if args.model:
        rows = evaluate(args.model, args.test_data, args.out)
        print(f"Wrote {len(rows)} metric rows to {args.out}")
        return
    model_dirs = _resolve_models(args.models, Path(args.model_root))
    datasets = [Path(args.test_data)]
    if args.test_unseen:
        datasets.append(Path(args.test_unseen))
    metric_rows, comparisons = evaluate_models(model_dirs, datasets, args.out)
    paired_path = Path(args.out).with_name(f"{Path(args.out).stem}_paired_tests.csv")
    print(f"Wrote {len(metric_rows)} metric rows to {args.out}")
    print(f"Wrote {len(comparisons)} paired tests to {paired_path}")


if __name__ == "__main__":
    main()
