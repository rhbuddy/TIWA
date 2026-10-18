"""Position updates for static, random-waypoint, and scripted STA motion."""

from __future__ import annotations

import numpy as np

from fa_rlwa.types import STAState


def advance_position(
    sta: STAState,
    rng: np.random.Generator,
    area_m: tuple[float, float],
    speed_range_mps: tuple[float, float] = (0.5, 1.5),
) -> None:
    """Advance one STA by one simulation second, keeping it inside the floor area."""

    if sta.mobility == "static":
        return
    if sta.mobility == "random_waypoint":
        if sta.pause_s > 0:
            sta.pause_s -= 1
            return
        if sta.destination is None:
            sta.destination = (float(rng.uniform(0, area_m[0])), float(rng.uniform(0, area_m[1])))
            sta.speed_mps = float(rng.uniform(*speed_range_mps))
        x, y = sta.pos
        dx, dy = sta.destination[0] - x, sta.destination[1] - y
        distance = float(np.hypot(dx, dy))
        if distance <= max(sta.speed_mps, 0.0):
            sta.pos = sta.destination
            sta.destination = None
            sta.pause_s = int(rng.integers(0, 31))
        elif distance > 0:
            sta.pos = (x + dx * sta.speed_mps / distance, y + dy * sta.speed_mps / distance)
        return

    if sta.mobility == "scripted" and sta.path:
        if sta.path_hold_s > 0:
            sta.path_hold_s -= 1
            return
        x, y = sta.pos
        target_x, target_y, hold_s = sta.path[sta.path_index % len(sta.path)]
        dx, dy = target_x - x, target_y - y
        distance = float(np.hypot(dx, dy))
        if distance <= max(sta.speed_mps, 0.5):
            sta.pos = (target_x, target_y)
            sta.path_index = (sta.path_index + 1) % len(sta.path)
            sta.path_hold_s = hold_s
        elif distance > 0:
            speed = max(sta.speed_mps, 0.5)
            sta.pos = (x + dx * speed / distance, y + dy * speed / distance)
        return

    raise ValueError(f"Unsupported mobility model {sta.mobility!r}")
