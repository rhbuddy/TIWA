"""Reference-formula tests from SRS Section 14.7."""

import numpy as np
import pytest

from fa_rlwa.monitor.indicators import jain_fairness
from fa_rlwa.monitor.normalize import normalize_rssi
from fa_rlwa.sim.mac import airtime_demand, collision_probability
from fa_rlwa.sim.propagation import path_loss_db, phy_rate_mbps, rssi_dbm


def test_jain_fairness_reference_values() -> None:
    assert jain_fairness([2, 2, 2, 2]) == pytest.approx(1.0)
    assert jain_fairness([4, 0, 0, 0]) == pytest.approx(0.25)
    assert jain_fairness([1, 2, 3, 4]) == pytest.approx(100 / (4 * 30))


def test_path_loss_and_received_power_reference_values() -> None:
    assert path_loss_db(10, pl0_db=40, exponent=3) == pytest.approx(70.0)
    assert rssi_dbm(15, 10, pl0_db=40, exponent=3) == pytest.approx(-55.0)
    assert path_loss_db(20, pl0_db=40, exponent=3) == pytest.approx(79.03, abs=0.01)
    assert rssi_dbm(15, 20, pl0_db=40, exponent=3) == pytest.approx(-64.03, abs=0.01)


def test_phy_rate_lookup_reference_values() -> None:
    actual = phy_rate_mbps(np.asarray([-60, -68, -75, -83]), collision_probability=0.0)
    np.testing.assert_allclose(actual, [65.0, 39.0, 19.5, 0.0])


def test_airtime_and_collision_reference_values() -> None:
    assert airtime_demand(3.0, 39.0, 0.6, 0.0) == pytest.approx(3 / (39 * 0.6))
    assert collision_probability(10, tau=0.01) == pytest.approx(1 - 0.99**9)


def test_rssi_normalization_reference_value() -> None:
    assert normalize_rssi(-65.0) == pytest.approx(0.5)
