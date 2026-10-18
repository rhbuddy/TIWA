"""Per-run Wi-Fi metrics and report-row construction."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from fa_rlwa.controller.handover import count_ping_pongs
from fa_rlwa.types import APState, IndicatorReport


def channel_occupancy_std(report: IndicatorReport, aps: list[APState]) -> float:
    """Return standard deviation of mean occupancy across distinct channels."""

    by_channel: dict[int, list[float]] = defaultdict(list)
    for ap, occupancy in zip(aps, report.occ_frac, strict=True):
        by_channel[ap.channel].append(float(occupancy))
    values = np.asarray([np.mean(loads) for loads in by_channel.values()], dtype=float)
    return float(np.std(values)) if values.size else 0.0


def summarize_run(
    *,
    run_id: str,
    scenario: str,
    policy: str,
    seed: int,
    step_rows: Iterable[Mapping[str, Any]],
    reports: Iterable[IndicatorReport],
    aps: list[APState],
    handover_events: list[dict[str, object]],
    decision_latencies_ms: Iterable[float],
) -> dict[str, Any]:
    """Compute SRS KPIs from the post-warm-up per-cycle data."""

    rows = list(step_rows)
    metric_names = (
        "agg_tput_mbps",
        "offered_mbps",
        "jfi",
        "retx_per_frame",
        "occ_mean",
        "occ_std",
        "phy_mean_mbps",
    )
    result: dict[str, Any] = {
        "run_id": run_id,
        "scenario": scenario,
        "policy": policy,
        "seed": seed,
    }
    for name in metric_names:
        values = np.asarray([float(row[name]) for row in rows], dtype=float)
        result[name] = float(np.mean(values)) if values.size else 0.0
    result["handovers"] = len(handover_events)
    result["pingpongs"] = count_ping_pongs(handover_events, window_s=10)
    latencies = np.asarray(list(decision_latencies_ms), dtype=float)
    result["decision_ms_mean"] = float(np.mean(latencies)) if latencies.size else 0.0
    result["decision_ms_p95"] = float(np.percentile(latencies, 95)) if latencies.size else 0.0
    result["invalid_actions"] = int(sum(int(row.get("invalid_actions", 0)) for row in rows))
    report_list = list(reports)
    if report_list:
        result["occ_channel_std"] = float(
            np.mean([channel_occupancy_std(item, aps) for item in report_list])
        )
    else:
        result["occ_channel_std"] = 0.0
    return result
