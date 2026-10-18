"""Network determinism, handover validation, and throughput-bound tests."""

import numpy as np
import pytest

from fa_rlwa.config import scenario_from_mapping
from fa_rlwa.sim.network import WifiNetwork
from fa_rlwa.types import Handover


def small_scenario(*, far_aps: bool = False):
    """Build a compact quiet test network with predictable AP coordinates."""

    if far_aps:
        aps = [
            {"id": 0, "pos": [0, 5], "channel": 1},
            {"id": 1, "pos": [100, 5], "channel": 6},
        ]
        area = [100, 10]
    else:
        aps = [
            {"id": 0, "pos": [0, 5], "channel": 1},
            {"id": 1, "pos": [10, 5], "channel": 6},
        ]
        area = [10, 10]
    return scenario_from_mapping(
        {
            "name": "test_network",
            "area_m": area,
            "duration_s": 40,
            "warmup_s": 0,
            "channels": [1, 6, 11],
            "aps": aps,
            "background_stas": {
                "count": 8,
                "placement": "hotspot",
                "mobility": "static",
                "traffic": {"type": "cbr", "mbps": 1.0},
            },
            "measurement_noise": False,
            "handover": {"dwell_s": 10, "outage_s": 0.2},
        }
    )


def test_fixed_seed_produces_identical_network_trace() -> None:
    scenario = small_scenario()
    left = WifiNetwork(scenario, seed=19)
    right = WifiNetwork(scenario, seed=19)
    for _ in range(20):
        left_state = left.step()
        right_state = right.step()
        np.testing.assert_array_equal(left_state.rssi_dbm, right_state.rssi_dbm)
        np.testing.assert_array_equal(left_state.sta_tput_mbps, right_state.sta_tput_mbps)
        np.testing.assert_array_equal(left_state.sta_loss_frac, right_state.sta_loss_frac)


def test_throughput_never_exceeds_per_sta_offered_load() -> None:
    network = WifiNetwork(small_scenario(), seed=3)
    for _ in range(15):
        state = network.step()
        assert np.all(state.sta_tput_mbps <= state.sta_demand_mbps + 1e-9)
        assert np.all(state.sta_tput_mbps >= 0.0)


def test_out_of_range_handover_is_rejected_and_logged() -> None:
    network = WifiNetwork(small_scenario(far_aps=True), seed=0)
    state = network.snapshot()
    sta = state.stas[0]
    assert sta.ap_id == 0
    next_state = network.step([Handover(sta.sta_id, 1)])
    assert next_state.stas[0].ap_id == 0
    assert not network.handover_events
    assert network.invalid_action_events[-1]["reason"] == "out_of_range"


def test_dwell_time_rejects_a_second_move() -> None:
    network = WifiNetwork(small_scenario(), seed=1)
    sta_id = network.snapshot().stas[0].sta_id
    first = network.step([Handover(sta_id, 1)])
    assert first.stas[0].ap_id == 1
    second = network.step([Handover(sta_id, 0)])
    assert second.stas[0].ap_id == 1
    assert network.invalid_action_events[-1]["reason"] == "dwell_time"


def test_scenario_validation_fails_early_for_bad_ap_positions() -> None:
    with pytest.raises(ValueError, match="outside area_m"):
        scenario_from_mapping(
            {
                "name": "bad",
                "area_m": [5, 5],
                "aps": [{"id": 0, "pos": [8, 8], "channel": 1}],
            }
        )
