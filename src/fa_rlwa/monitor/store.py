"""In-memory indicator history with configurable moving-average views."""

from __future__ import annotations

from collections import deque

import numpy as np

from fa_rlwa.monitor.indicators import moving_average
from fa_rlwa.types import IndicatorReport


class IndicatorStore:
    """Keep a bounded, time-ordered series of per-AP indicator reports."""

    def __init__(self, max_reports: int = 300) -> None:
        if max_reports < 1:
            raise ValueError("max_reports must be positive")
        self._reports: deque[IndicatorReport] = deque(maxlen=max_reports)

    def append(self, report: IndicatorReport) -> None:
        """Append one report; report timestamps must be non-decreasing."""

        if self._reports and report.t < self._reports[-1].t:
            raise ValueError("indicator report timestamps must be non-decreasing")
        self._reports.append(report)

    @property
    def reports(self) -> list[IndicatorReport]:
        """Return a shallow list copy of the retained reports."""

        return list(self._reports)

    def latest(self) -> IndicatorReport:
        """Return the newest report or raise if the store is empty."""

        if not self._reports:
            raise ValueError("indicator store is empty")
        return self._reports[-1]

    def average(self, field: str, window: int = 10) -> np.ndarray:
        """Return a moving-average vector for a named report field."""

        return moving_average(list(self._reports), field, window)
