"""Top-level command dispatcher for the FA-RLWA research framework."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

COMMANDS = {
    "run": "fa_rlwa.experiments.runner",
    "forecast-dataset": "fa_rlwa.forecast.dataset",
    "forecast-train": "fa_rlwa.forecast.train",
    "forecast-evaluate": "fa_rlwa.forecast.evaluate",
    "train-dqn": "fa_rlwa.rl.train_dqn",
    "train-ppo": "fa_rlwa.rl.train_ppo",
    "report": "fa_rlwa.experiments.report",
    "wifi-check": "fa_rlwa.validation.wifi_scan",
    "validate-mininet": "fa_rlwa.validation.mininet_wifi.runner",
}


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch to a focused module CLI while preserving its own argument parser."""

    parser = argparse.ArgumentParser(
        prog="fa-rlwa",
        description="Forecast-aware reinforcement learning for Wi-Fi association",
    )
    parser.add_argument("command", choices=sorted(COMMANDS))
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.parse_args(arguments)
        return
    command = arguments[0]
    if command not in COMMANDS:
        parser.error(f"invalid choice: {command!r} (choose from {', '.join(sorted(COMMANDS))})")
    sys.argv = [COMMANDS[command], *arguments[1:]]
    module = __import__(COMMANDS[command], fromlist=["main"])
    module.main()
