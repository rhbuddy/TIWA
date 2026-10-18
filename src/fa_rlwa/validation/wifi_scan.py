"""Linux Wi-Fi adapter preflight for active scan support."""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class WifiScanCheck:
    """Outcome of probing whether an interface can complete an 802.11 scan."""

    interface: str
    can_scan: bool
    visible_bss_count: int | None
    detail: str


def _brief_output(stdout: str, stderr: str) -> str:
    """Return a short command diagnostic without exposing scanned SSIDs."""

    lines = (stderr or stdout).strip().splitlines()
    return lines[0][:240] if lines else "command returned no diagnostic output"


def check_wifi_scan(interface: str, timeout_s: float = 15.0) -> WifiScanCheck:
    """Check interface existence and run an active scan using Linux ``iw``.

    A successful scan with zero BSS entries still proves the adapter accepted the
    scan request; it simply means no beacons were observed during this scan.
    The function never brings an interface up or changes its mode.
    """

    if not interface or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,15}", interface):
        return WifiScanCheck(interface, False, None, "invalid Wi-Fi interface name")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        return WifiScanCheck(
            interface, False, None, "scan timeout must be finite and greater than zero"
        )
    if not sys.platform.startswith("linux"):
        return WifiScanCheck(
            interface,
            False,
            None,
            "Wi-Fi adapter scanning is supported only on Linux via the `iw` utility",
        )
    iw = shutil.which("iw")
    if iw is None:
        return WifiScanCheck(
            interface,
            False,
            None,
            "the Linux 'iw' utility is not installed or is not available on PATH",
        )
    try:
        info = subprocess.run(
            [iw, "dev", interface, "info"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if info.returncode != 0:
            return WifiScanCheck(
                interface,
                False,
                None,
                f"interface could not be queried: {_brief_output(info.stdout, info.stderr)}",
            )
        scan = subprocess.run(
            [iw, "dev", interface, "scan"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except FileNotFoundError:
        return WifiScanCheck(interface, False, None, "the 'iw' utility disappeared from PATH")
    except PermissionError as exc:
        return WifiScanCheck(interface, False, None, f"permission denied while scanning: {exc}")
    except subprocess.TimeoutExpired:
        return WifiScanCheck(
            interface,
            False,
            None,
            f"Wi-Fi scan did not finish within {timeout_s:g} seconds",
        )
    if scan.returncode != 0:
        return WifiScanCheck(
            interface,
            False,
            None,
            f"adapter rejected the scan request: {_brief_output(scan.stdout, scan.stderr)}",
        )
    count = len(re.findall(r"(?m)^\s*BSS\s+", scan.stdout))
    detail = (
        f"active scan succeeded; observed {count} BSS entr{'y' if count == 1 else 'ies'}"
        if count
        else "active scan succeeded; no beacons were observed"
    )
    return WifiScanCheck(interface, True, count, detail)


def main() -> None:
    """CLI for a standalone Wi-Fi scan-capability check."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True, help="Linux Wi-Fi interface, e.g. wlan0")
    parser.add_argument(
        "--timeout", type=float, default=15.0, help="per-command timeout in seconds"
    )
    args = parser.parse_args()
    result = check_wifi_scan(args.interface, args.timeout)
    status = "PASS" if result.can_scan else "FAIL"
    print(f"{status}: {result.interface}: {result.detail}")
    if not result.can_scan:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
