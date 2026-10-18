"""Handover-log analysis utilities."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, cast


def count_ping_pongs(events: Iterable[Mapping[str, object]], window_s: int = 10) -> int:
    """Count a return to the immediately previous AP within ``window_s`` seconds."""

    if window_s < 0:
        raise ValueError("window_s cannot be negative")
    by_sta: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for event in events:
        try:
            by_sta[int(cast(Any, event["sta_id"]))].append(
                (
                    int(cast(Any, event["t"])),
                    int(cast(Any, event["from_ap"])),
                    int(cast(Any, event["to_ap"])),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    count = 0
    for transitions in by_sta.values():
        transitions.sort()
        for first, second in zip(transitions, transitions[1:], strict=False):
            t1, source, target = first
            t2, source2, target2 = second
            if t2 - t1 <= window_s and source2 == target and target2 == source:
                count += 1
    return count
