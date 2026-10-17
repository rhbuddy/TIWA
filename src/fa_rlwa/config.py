"""Scenario configuration types, YAML loading, and early validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class APConfig:
    """Static AP parameters; positions are metres in the scenario coordinate system."""

    ap_id: int
    pos: tuple[float, float]
    channel: int
    tx_dbm: float = 20.0


@dataclass(frozen=True)
class StationGroupConfig:
    """Specification for a population of initially present STAs."""

    count: int
    placement: str = "uniform"
    mobility: str = "static"
    traffic_type: str = "cbr"
    offered_mbps: float = 1.0
    speed_mps: tuple[float, float] = (0.5, 1.5)
    scripted_path: tuple[tuple[float, float, int], ...] = ()


@dataclass(frozen=True)
class FlashCrowdConfig:
    """A synthetic burst of clients arriving around a hotspot."""

    count: int
    start_s: int
    window_s: int
    hotspot: tuple[float, float]
    spread_m: float = 3.0
    dwell_mean_s: float = 120.0
    traffic_type: str = "cbr"
    offered_mbps: float = 1.0


@dataclass(frozen=True)
class PropagationConfig:
    """Log-distance propagation and measurement noise parameters."""

    pl0_db: float = 40.0
    exponent: float = 3.0
    shadow_sigma_db: float = 4.0
    shadow_rho: float = 0.95
    rssi_noise_db: float = 2.0
    association_threshold_dbm: float = -82.0
    ap_carrier_sense_dbm: float = -82.0


@dataclass(frozen=True)
class MacConfig:
    """Contention and airtime model parameters."""

    efficiency: float = 0.6
    c_max: float = 0.9
    tau: float = 0.01
    collision_cap: float = 0.5
    retry_limit: int = 7
    tput_noise_fraction: float = 0.05


@dataclass(frozen=True)
class HandoverConfig:
    """Controller transition costs; outage is expressed in seconds."""

    mode: str = "lvap"
    outage_s: float = 0.2
    dwell_s: int = 10
    channel_change_dwell_s: int = 60
    max_moves_per_cycle: int = 2


@dataclass(frozen=True)
class ScenarioConfig:
    """Complete, validated scenario definition."""

    name: str
    area_m: tuple[float, float]
    aps: tuple[APConfig, ...]
    groups: tuple[StationGroupConfig, ...] = ()
    flash_crowd: FlashCrowdConfig | None = None
    duration_s: int = 1200
    warmup_s: int = 60
    step_s: int = 1
    channels: tuple[int, ...] = (1, 6, 11)
    propagation: PropagationConfig = field(default_factory=PropagationConfig)
    mac: MacConfig = field(default_factory=MacConfig)
    handover: HandoverConfig = field(default_factory=HandoverConfig)
    measurement_noise: bool = True

    @property
    def max_stas(self) -> int:
        """Maximum population, including the configured flash crowd."""

        return sum(group.count for group in self.groups) + (
            self.flash_crowd.count if self.flash_crowd else 0
        )


def _traffic_params(value: Any) -> tuple[str, float]:
    """Parse a traffic mapping and apply the SRS traffic-profile defaults."""

    if isinstance(value, str):
        traffic_type, requested = value, None
    elif isinstance(value, dict):
        traffic_type = str(value.get("type", "cbr"))
        requested = value.get("mbps")
    else:
        traffic_type, requested = "cbr", None
    defaults = {"cbr": 1.0, "voip": 0.1, "video": 4.0, "bulk": 8.0}
    if traffic_type not in defaults:
        raise ValueError(f"Unsupported traffic type {traffic_type!r}; use {sorted(defaults)}")
    return traffic_type, float(defaults[traffic_type] if requested is None else requested)


def _parse_group(raw: dict[str, Any]) -> StationGroupConfig:
    """Convert a station-population mapping into a typed configuration."""

    traffic_type, offered_mbps = _traffic_params(raw.get("traffic", "cbr"))
    speed = raw.get("speed_mps", [0.5, 1.5])
    path = raw.get("scripted_path", ())
    return StationGroupConfig(
        count=int(raw.get("count", 0)),
        placement=str(raw.get("placement", "uniform")),
        mobility=str(raw.get("mobility", "static")),
        traffic_type=traffic_type,
        offered_mbps=offered_mbps,
        speed_mps=(float(speed[0]), float(speed[1])),
        scripted_path=tuple((float(p[0]), float(p[1]), int(p[2])) for p in path),
    )


def validate_scenario(config: ScenarioConfig) -> None:
    """Raise ``ValueError`` with a useful message for invalid scenario settings."""

    if not config.name:
        raise ValueError("Scenario name must not be empty")
    if config.area_m[0] <= 0 or config.area_m[1] <= 0:
        raise ValueError("area_m dimensions must be positive")
    if not 1 <= len(config.aps) <= 10:
        raise ValueError("Scenario must define between 1 and 10 APs")
    ids = [ap.ap_id for ap in config.aps]
    if len(ids) != len(set(ids)) or sorted(ids) != list(range(len(ids))):
        raise ValueError("AP ids must be unique, contiguous, and zero-based")
    if len(config.channels) != len(set(config.channels)) or any(
        channel not in {1, 6, 11} for channel in config.channels
    ):
        raise ValueError("2.4 GHz channels must be unique values selected from 1, 6, 11")
    if not set(config.channels).issuperset(ap.channel for ap in config.aps):
        raise ValueError("Every AP channel must be listed in the scenario channels")
    for ap in config.aps:
        if not (0 <= ap.pos[0] <= config.area_m[0] and 0 <= ap.pos[1] <= config.area_m[1]):
            raise ValueError(f"AP {ap.ap_id} position {ap.pos} lies outside area_m {config.area_m}")
    if config.duration_s < 1 or config.warmup_s < 0 or config.step_s < 1:
        raise ValueError("duration_s and step_s must be positive; warmup_s cannot be negative")
    if config.max_stas > 100:
        raise ValueError("Scenario population cannot exceed 100 STAs")
    if config.propagation.exponent <= 0 or config.propagation.shadow_sigma_db < 0:
        raise ValueError("Propagation exponent must be positive and shadow sigma non-negative")
    if not 0 <= config.propagation.shadow_rho < 1:
        raise ValueError("shadow_rho must be in [0, 1)")
    if not 0 < config.mac.efficiency <= 1 or not 0 < config.mac.c_max <= 1:
        raise ValueError("MAC efficiency and c_max must be in (0, 1]")
    if not 0 <= config.mac.tau < 1:
        raise ValueError("MAC tau must be in [0, 1)")
    if config.handover.dwell_s < 0 or config.handover.max_moves_per_cycle < 1:
        raise ValueError("Handover dwell must be non-negative and max_moves_per_cycle positive")
    for group in config.groups:
        if group.count < 0 or group.placement not in {"uniform", "hotspot"}:
            raise ValueError(
                "Station group count must be non-negative and placement uniform/hotspot"
            )
        if group.mobility not in {"static", "random_waypoint", "scripted"}:
            raise ValueError(f"Unsupported mobility model {group.mobility!r}")
        if group.offered_mbps < 0:
            raise ValueError("offered_mbps cannot be negative")
        if group.speed_mps[0] < 0 or group.speed_mps[1] < group.speed_mps[0]:
            raise ValueError("speed_mps must be an ordered, non-negative [minimum, maximum] pair")
        if group.mobility == "scripted" and not group.scripted_path:
            raise ValueError("A scripted mobility group requires scripted_path points")
        for x, y, _ in group.scripted_path:
            if not (0 <= x <= config.area_m[0] and 0 <= y <= config.area_m[1]):
                raise ValueError("scripted_path points must lie within area_m")
    if config.flash_crowd:
        flash = config.flash_crowd
        if flash.count < 0 or flash.start_s < 0 or flash.window_s < 0:
            raise ValueError("flash crowd count, start_s, and window_s must be non-negative")
        if flash.spread_m < 0 or flash.dwell_mean_s <= 0 or flash.offered_mbps < 0:
            raise ValueError("flash crowd spread/dwell/load values are invalid")
        if not (
            0 <= flash.hotspot[0] <= config.area_m[0] and 0 <= flash.hotspot[1] <= config.area_m[1]
        ):
            raise ValueError("flash-crowd hotspot must lie within area_m")


def scenario_from_mapping(raw: dict[str, Any]) -> ScenarioConfig:
    """Build and validate a :class:`ScenarioConfig` from decoded YAML."""

    aps = tuple(
        APConfig(
            ap_id=int(ap.get("id", index)),
            pos=(float(ap["pos"][0]), float(ap["pos"][1])),
            channel=int(ap.get("channel", 1)),
            tx_dbm=float(ap.get("tx_dbm", 20.0)),
        )
        for index, ap in enumerate(raw.get("aps", []))
    )
    groups: list[StationGroupConfig] = []
    if raw.get("background_stas"):
        groups.append(_parse_group(raw["background_stas"]))
    if raw.get("mobile_stas"):
        groups.append(_parse_group(raw["mobile_stas"]))
    groups.extend(_parse_group(group) for group in raw.get("groups", []))

    flash_raw = raw.get("flash_crowd")
    flash: FlashCrowdConfig | None = None
    if flash_raw:
        traffic_type, offered_mbps = _traffic_params(flash_raw.get("traffic", "cbr"))
        flash = FlashCrowdConfig(
            count=int(flash_raw.get("count", 0)),
            start_s=int(flash_raw.get("start_s", 300)),
            window_s=int(flash_raw.get("window_s", 30)),
            hotspot=(float(flash_raw["hotspot"][0]), float(flash_raw["hotspot"][1])),
            spread_m=float(flash_raw.get("spread_m", 3.0)),
            dwell_mean_s=float(flash_raw.get("dwell_mean_s", 120.0)),
            traffic_type=traffic_type,
            offered_mbps=offered_mbps,
        )

    propagation = PropagationConfig(**raw.get("propagation", {}))
    mac = MacConfig(**raw.get("mac", {}))
    handover = HandoverConfig(**raw.get("handover", {}))
    config = ScenarioConfig(
        name=str(raw.get("name", "unnamed")),
        area_m=(float(raw.get("area_m", [15, 9])[0]), float(raw.get("area_m", [15, 9])[1])),
        aps=aps,
        groups=tuple(groups),
        flash_crowd=flash,
        duration_s=int(raw.get("duration_s", 1200)),
        warmup_s=int(raw.get("warmup_s", 60)),
        step_s=int(raw.get("step_s", 1)),
        channels=tuple(int(channel) for channel in raw.get("channels", [1, 6, 11])),
        propagation=propagation,
        mac=mac,
        handover=handover,
        measurement_noise=bool(raw.get("measurement_noise", True)),
    )
    validate_scenario(config)
    return config


def load_scenario(path: str | Path) -> ScenarioConfig:
    """Load a scenario YAML file and validate it before simulation."""

    with Path(path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"Scenario file {path} must contain a YAML mapping")
    return scenario_from_mapping(raw)


def scenario_to_mapping(config: ScenarioConfig) -> dict[str, Any]:
    """Return a JSON/YAML-friendly representation of a scenario."""

    return {
        "name": config.name,
        "area_m": list(config.area_m),
        "duration_s": config.duration_s,
        "warmup_s": config.warmup_s,
        "step_s": config.step_s,
        "channels": list(config.channels),
        "aps": [
            {"id": ap.ap_id, "pos": list(ap.pos), "channel": ap.channel, "tx_dbm": ap.tx_dbm}
            for ap in config.aps
        ],
        "groups": [
            {
                "count": group.count,
                "placement": group.placement,
                "mobility": group.mobility,
                "traffic": {"type": group.traffic_type, "mbps": group.offered_mbps},
                "speed_mps": list(group.speed_mps),
                "scripted_path": [list(point) for point in group.scripted_path],
            }
            for group in config.groups
        ],
        "flash_crowd": None
        if config.flash_crowd is None
        else {
            "count": config.flash_crowd.count,
            "start_s": config.flash_crowd.start_s,
            "window_s": config.flash_crowd.window_s,
            "hotspot": list(config.flash_crowd.hotspot),
            "spread_m": config.flash_crowd.spread_m,
            "dwell_mean_s": config.flash_crowd.dwell_mean_s,
            "traffic": {
                "type": config.flash_crowd.traffic_type,
                "mbps": config.flash_crowd.offered_mbps,
            },
        },
        "propagation": config.propagation.__dict__,
        "mac": config.mac.__dict__,
        "handover": config.handover.__dict__,
        "measurement_noise": config.measurement_noise,
    }
