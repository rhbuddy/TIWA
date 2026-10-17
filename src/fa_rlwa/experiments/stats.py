"""Seed-level bootstrap intervals and multiple-comparison correction."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a percentile bootstrap confidence interval for a sample mean."""

    sample = np.asarray(list(values), dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        raise ValueError("at least one finite observation is required")
    if not 0.0 < confidence < 1.0 or resamples < 1:
        raise ValueError("confidence must be in (0, 1) and resamples must be positive")
    if sample.size == 1:
        return float(sample[0]), float(sample[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, sample.size, size=(resamples, sample.size))
    means = sample[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


def paired_wilcoxon(left: Iterable[float], right: Iterable[float]) -> float:
    """Return a two-sided paired Wilcoxon p-value (SciPy required for analysis)."""

    x, y = np.asarray(list(left), dtype=float), np.asarray(list(right), dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("paired samples must be one-dimensional and equal length")
    if x.size == 0:
        raise ValueError("paired samples cannot be empty")
    differences = x - y
    if np.allclose(differences, 0.0):
        return 1.0
    try:
        from scipy.stats import wilcoxon
    except ImportError as exc:
        raise ImportError("Install scipy to run paired significance tests") from exc
    return float(wilcoxon(x, y, alternative="two-sided", zero_method="wilcox").pvalue)


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Apply Holm's step-down family-wise error correction to p-values."""

    values = np.asarray(list(p_values), dtype=float)
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running_max = 0.0
    count = len(values)
    for rank, original_index in enumerate(order):
        candidate = (count - rank) * values[original_index]
        running_max = max(running_max, candidate)
        adjusted[original_index] = min(1.0, running_max)
    return adjusted
