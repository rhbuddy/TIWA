"""Committed scenario YAML files are valid and bounded by the SRS envelope."""

from pathlib import Path

import numpy as np

from fa_rlwa.config import load_scenario, scenario_from_mapping, scenario_to_mapping
from fa_rlwa.sim.network import WifiNetwork


def test_all_scenario_files_load_and_respect_scale_limits() -> None:
    paths = sorted(Path("configs/scenarios").glob("*.yaml"))
    assert len(paths) >= 8
    for path in paths:
        scenario = load_scenario(path)
        assert 1 <= len(scenario.aps) <= 10
        assert scenario.max_stas <= 100
        assert scenario.step_s == 1


def test_paper_experiment_layout_is_explicit_and_seed_reproducible() -> None:
    scenario = load_scenario("configs/scenarios/sc1_paper_aspect1.yaml")
    round_trip = scenario_from_mapping(scenario_to_mapping(scenario))
    assert [group.count for group in round_trip.groups] == [6, 24]
    assert round_trip.mac.efficiency == 0.2

    first = WifiNetwork(scenario, seed=2026)
    second = WifiNetwork(scenario, seed=2026)
    assert first.snapshot().stas
    assert [sta.pos for sta in first.snapshot().stas] == [sta.pos for sta in second.snapshot().stas]
    for sta_id, sta in enumerate(first.snapshot().stas[:6]):
        ap = scenario.aps[sta_id % len(scenario.aps)]
        assert np.hypot(sta.pos[0] - ap.pos[0], sta.pos[1] - ap.pos[1]) < 2.0
    overlap_positions = np.asarray([sta.pos for sta in first.snapshot().stas[6:]])
    assert np.all(np.linalg.norm(overlap_positions - np.asarray([6.5, 6.0]), axis=1) < 1.5)
