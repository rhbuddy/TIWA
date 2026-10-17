"""Committed scenario YAML files are valid and bounded by the SRS envelope."""

from pathlib import Path

from fa_rlwa.config import load_scenario


def test_all_scenario_files_load_and_respect_scale_limits() -> None:
    paths = sorted(Path("configs/scenarios").glob("*.yaml"))
    assert len(paths) >= 8
    for path in paths:
        scenario = load_scenario(path)
        assert 1 <= len(scenario.aps) <= 10
        assert scenario.max_stas <= 100
        assert scenario.step_s == 1
