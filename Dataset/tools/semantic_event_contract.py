"""Deterministic low-altitude semantic event-chain contract.

This module is the single machine-readable source for the 70 episode visual
acceptance table. Generators, converters, validators, and capture runners must
consume this contract instead of copying counts or role policy locally.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaptureBoundaryContract:
    boundary_id: str
    geometry_source: str
    anchor_entity_roles: tuple[str, ...]
    boundary_role: str
    z_policy: str


@dataclass(frozen=True)
class PadBoundaryPolicy:
    default: str
    inside_required_for: tuple[str, ...]


@dataclass(frozen=True)
class InspectContract:
    role: str
    altitude_code: str
    altitude_m: float
    min_path_length_m: float
    required_presence: str
    motion_policy: str
    corridor_policy: str
    fov_coverage_required: bool
    sensor_profile_required: bool


@dataclass(frozen=True)
class EpisodeContract:
    scenario_id: str
    uav: int
    vehicle: int
    pedestrian: int
    facility: int
    logical: int
    inspect_altitude_m: float
    required_event: str
    required_intents: tuple[str, ...]
    capture_boundary: CaptureBoundaryContract
    uav_boundary_crossing_required: bool
    pad_policy: PadBoundaryPolicy
    inspect: InspectContract
    vehicle_role: str
    pedestrian_role: str
    weather: str

    @property
    def counts(self) -> dict[str, int]:
        return {
            "uav": self.uav,
            "vehicle": self.vehicle,
            "pedestrian": self.pedestrian,
            "facility": self.facility,
            "logical": self.logical,
        }

    @property
    def inspect_code(self) -> str:
        return f"I{int(round(self.inspect_altitude_m))}"

    @property
    def pad_boundary_policy(self) -> str:
        return self.pad_policy.default


INTENT_ALIASES: dict[str, frozenset[str]] = {
    "airspace_boundary_conflict": frozenset(
        {
            "boundary conflict",
            "geofence intrusion",
            "nfz proximity alert",
            "no fly zone violation",
            "airspace boundary conflict",
            "nfz violation alert",
            "lockdown",
        }
    ),
    "avoid_or_rth": frozenset(
        {
            "avoid",
            "avoidance",
            "evade",
            "evasion",
            "rth",
            "return home",
            "return safe airspace",
            "reroute",
            "deconflict",
            "separation",
            "divert",
            "pull up",
        }
    ),
    "landing_or_terminal_resolution": frozenset(
        {
            "land",
            "lands",
            "landing",
            "touchdown",
            "safe land",
            "terminal resolution",
            "backup landing",
            "forced landing",
        }
    ),
    "pad_contention": frozenset(
        {
            "pad contention",
            "pad request",
            "pad arbitration",
            "priority landing",
            "charger unavailable",
            "charger failure",
            "station fail",
        }
    ),
    "collision_or_near_miss": frozenset(
        {
            "collision",
            "near miss",
            "nearmiss",
            "crossing conflict",
            "uav vehicle crossing",
            "pedestrian near miss",
            "vehicle conflict",
            "contact risk",
            "jaywalk conflict",
        }
    ),
    "forced_landing": frozenset({"forced descent", "forced landing", "emergency landing", "touchdown"}),
    "crowd_evacuation": frozenset({"crowd evacuation", "crowd evac", "evacuation", "evac", "crowd response", "crowd clear"}),
    "weather_degradation": frozenset({"rain", "fog", "wind", "gust", "dusk", "heat", "visibility"}),
    "digital_anomaly": frozenset(
        {
            "c2 degraded",
            "tower degraded",
            "c2 loss",
            "latency",
            "packet loss",
            "spoof",
            "jamming",
            "gcs intrusion",
            "gnss anomaly",
            "gnss drift",
        }
    ),
    "recovery_or_restore": frozenset(
        {
            "restore",
            "recovery",
            "recover",
            "correction",
            "corrected",
            "relocalize",
            "handoff",
            "validate",
            "report",
            "secure recovery",
            "backup link",
            "channel",
            "lockout",
        }
    ),
    "traffic_or_facility_incident": frozenset(
        {
            "roadwork closure",
            "hazmat leak",
            "hazmat spread",
            "signal fault",
            "queue",
            "manual flow",
            "ambulance priority",
            "av fault",
            "av failure",
            "fall",
            "responder",
            "medical handoff",
        }
    ),
}


def normalize_semantic_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def normalize_intent(value: str) -> str:
    return normalize_semantic_text(value).replace(" ", "_")


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized = normalize_semantic_text(phrase)
    if not normalized:
        return False
    return normalized in text


def canonical_intent_from_text(value: str) -> str | None:
    text = normalize_semantic_text(value)
    if not text:
        return None
    normalized = normalize_intent(text)
    if normalized in INTENT_ALIASES:
        return normalized
    for intent, aliases in INTENT_ALIASES.items():
        if intent == normalized or any(_contains_phrase(text, alias) for alias in aliases):
            return intent
    return normalized


def _intent_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) or isinstance(value, tuple):
        return tuple(str(item) for item in value if str(item))
    return ()


_ROWS: tuple[tuple[str, str, str, str, str, str], ...] = (
    ("L1-1_v1", "3/2/2/2/9", "I28", "boundary conflict > avoid/RTH > land", "normal road flow under airspace boundary", "walkers showing inhabited corridor", "clear"),
    ("L1-1_v2", "3/2/2/2/7", "I28", "alternate geofence intrusion > RTH > land", "steady traffic", "waiting/walking near visible ground reference", "clear"),
    ("L1-2_v1", "3/2/2/2/8", "I28", "altitude deviation > correction > land", "moving ground scale", "sidewalk walkers below altitude corridor", "clear"),
    ("L1-3_v1", "3/2/2/3/10", "I28", "intruder conflict > deconflict > dual landing", "uninterrupted flow under conflict", "bystanders not reacting unless risk descends", "clear"),
    ("L1-3_v2", "3/2/2/3/10", "I28", "fast intruder crossing > separation > land", "moving road context", "pedestrian flow giving low-altitude reference", "clear"),
    ("L1-4_v1", "4/2/2/4/15", "I28", "congestion > priority resequence > land", "slow background traffic", "waiting near facility/pad area", "clear"),
    ("L1-4_v2", "4/2/2/4/15", "I28", "dense congestion > hold/divert > staged landings", "queued/slow traffic", "facility-area walking/waiting", "clear"),
    ("L2-1_v1", "3/3/2/3/8", "I18", "tower degraded > backup restore > land", "traffic continues during C2 degradation", "facility pedestrians near tower", "clear"),
    ("L2-1_v2", "3/3/2/3/5", "I18", "C2 degraded > hold/reroute > restore", "moving traffic", "pedestrians provide facility context", "clear"),
    ("L2-2_v1", "3/3/2/3/6", "I18", "GNSS anomaly > drift > relocalize", "road flow used as visual relocalization background", "sidewalk reference", "clear"),
    ("L2-2_v2", "3/3/2/3/6", "I18", "longer GNSS drift > correction", "moving lane references", "non-reactive walkers", "clear"),
    ("L2-3_v1", "3/2/2/3/6", "I18", "charger unavailable > backup landing", "facility access traffic", "pedestrians near charging area", "clear"),
    ("L2-3_v2", "3/2/2/3/6", "I18", "alternate charger failure > reroute", "access-road background", "waiting/walking near charger", "clear"),
    ("L2-4_v1", "3/2/2/3/8", "I18", "pad request > arbitration > divert", "service vehicles near pad", "facility waiting context", "clear"),
    ("L2-4_v2", "3/2/2/3/8", "I18", "pad contention > priority landing", "service-road flow", "ground observers near pad", "clear"),
    ("L2-5_v1", "3/5/4/2/3", "I18", "signal fault > queue > manual flow", "queue/yield/manual-control traffic", "crosswalk wait/cross", "clear"),
    ("L3-1_v1", "3/3/3/2/14", "I18", "roadwork closure > detour > inspect", "detour/blocked/slow vehicles", "pedestrians reroute around barriers", "clear"),
    ("L3-2_v1", "3/2/2/2/6", "I28", "NFZ proximity alert > reroute", "ordinary traffic beneath NFZ", "ground population context", "clear"),
    ("L3-2_v2", "3/2/2/2/6", "I28", "alternate NFZ proximity > reroute", "moving traffic", "walkers giving occupied-zone context", "clear"),
    ("L3-3_v1", "3/2/8/4/8", "I18", "hazmat leak > isolation > evac", "ambulance/service response", "evacuation cohort", "clear"),
    ("L3-3_v2", "3/2/8/4/8", "I18", "hazmat spread > evacuation > handoff", "responder + blocked traffic", "evacuating and waiting groups", "clear"),
    ("L4-1_v1", "3/2/2/2/12", "I28", "UAV convergence > separation", "traffic below conflict area", "passive urban context", "clear"),
    ("L4-1_v2", "3/2/2/2/12", "I28", "alternate convergence > deconflict", "steady flow", "sidewalk walkers", "clear"),
    ("L4-2_v1", "3/2/2/2/6", "I18", "facade proximity > evade", "road context near facade", "building-side pedestrians", "clear"),
    ("L4-2_v2", "3/2/2/2/6", "I18", "facade near miss > recovery", "moving/stopped near building", "facade-scale pedestrians", "clear"),
    ("L4-3_v1", "3/2/8/2/4", "I10", "forced descent > crowd response", "hold/blocked by landing zone", "evade/retreat crowd", "clear"),
    ("L4-3_v2", "3/2/8/2/4", "I10", "forced landing > crowd clear", "emergency hold", "clear landing path", "clear"),
    ("L4-3_v3", "3/2/8/2/4", "I10", "forced landing variant > touchdown", "stopped/slow response", "bystanders retreat", "clear"),
    ("L4-4_v1", "3/3/2/1/4", "I18", "UAV-vehicle crossing > brake", "brake/yield/contact-risk traffic", "roadside witnesses", "clear"),
    ("L4-4_v2", "3/3/2/1/4", "I18", "crossing > emergency stop", "emergency stop/following vehicle reaction", "waiting context", "clear"),
    ("L4-5_v1", "3/2/4/1/5", "I10", "pedestrian near-miss > pull-up", "nearby slow traffic", "target + retreating pedestrians", "clear"),
    ("L4-5_v2", "3/2/4/1/5", "I10", "alternate near-miss > clear", "road context", "clear/retreat/wait states", "clear"),
    ("L4-5_v3", "3/2/4/1/5", "I10", "low-alt inspect > near-miss recovery", "moving context", "inspected group reacts", "clear"),
    ("L4-6_v1", "3/2/3/1/2", "I10", "jaywalk > vehicle brake > retreat", "braking/yielding vehicles", "jaywalker + waiting peds", "clear"),
    ("L4-6_v2", "3/2/3/1/2", "I10", "alternate jaywalk conflict", "brake/recover traffic", "retreat/wait", "clear"),
    ("L4-7_v1", "3/2/4/1/3", "I10", "fall > UAV detect > ambulance", "ambulance/yield traffic", "fallen + bystanders", "clear"),
    ("L4-7_v2", "3/2/4/1/3", "I10", "fall response > medical handoff", "responder vehicle and yielding car", "bystanders/medical wait", "clear"),
    ("L4-8_v1", "3/2/12/2/4", "I10", "crowd evacuation > safe hold", "stopped/held perimeter vehicles", "evacuating crowd", "clear"),
    ("L4-8_v2", "3/2/12/2/8", "I10", "evacuation variant > land", "perimeter hold", "evacuation to safe zone", "clear"),
    ("L4-9_v1", "3/3/3/1/2", "I18", "vehicle conflict > warning/brake", "conflict/brake/yield chain", "waiting at roadside", "clear"),
    ("L4-9_v2", "3/3/3/1/2", "I18", "alternate vehicle conflict", "lane conflict/recovery", "scale and risk background", "clear"),
    ("L4-10_v1", "3/4/3/1/3", "I18", "ambulance priority > yield", "ambulance priority + civilian yield", "crosswalk wait", "clear"),
    ("L4-10_v2", "3/4/3/1/3", "I18", "ambulance priority > clearance", "queue/yield/clearance", "waiting pedestrians", "clear"),
    ("L4-11_v1", "3/3/2/1/3", "I18", "AV fault > safe stop > report", "AV stop + follower reaction", "roadside context", "clear"),
    ("L4-11_v2", "3/3/2/1/3", "I18", "AV failure > hazard hold", "stopped AV and following traffic", "non-event background", "clear"),
    ("L5-1_v1", "3/4/6/2/6", "I22", "rain > slowdown > recovery", "rain-slow traffic", "seek shelter/walk slower", "rain"),
    ("L5-1_v2", "3/4/6/2/6", "I22", "rain variant > degraded route", "cautious traffic", "shelter/wait states", "rain"),
    ("L5-1_v3", "3/4/6/2/6", "I22", "heavy rain > safe land", "slow/queued", "sheltering and crossing delay", "rain"),
    ("L5-2_v1", "3/3/6/2/6", "I22", "fog onset > abort > land", "cautious low-visibility traffic", "slow/wait", "fog"),
    ("L5-2_v2", "3/3/6/2/6", "I22", "fog variant > recovery", "slow traffic", "reduced-visibility walking", "fog"),
    ("L5-3_v1", "3/3/4/2/6", "I22", "wind > payload swing > recovery", "normal flow", "wind-affected walking/waiting", "wind"),
    ("L5-3_v2", "3/3/4/2/6", "I22", "gust variant > land", "moving context", "cautious movement", "wind"),
    ("L5-4_v1", "3/3/4/2/5", "I22", "dusk > IR switch > validate", "low-light traffic", "low-light crossing/wait", "dusk"),
    ("L5-5_v1", "3/3/4/2/5", "I22", "heat derate > charger decision", "normal/slowed access traffic", "heat-wait/shelter", "heat"),
    ("L6-1_v1", "3/3/4/3/8", "I22", "C2 loss > failsafe/RTH", "unaffected traffic proving digital-only fault", "operator-area context", "clear"),
    ("L6-1_v2", "3/3/4/3/8", "I22", "long-route C2 loss > recovery", "steady traffic", "facility pedestrians", "clear"),
    ("L6-2_v1", "3/3/4/3/8", "I22", "latency > backup link", "traffic continues", "passive background near facility", "clear"),
    ("L6-2_v2", "3/3/4/3/8", "I22", "packet loss > recovery", "moving road context", "sidewalk context", "clear"),
    ("L6-3_v1", "3/3/4/3/8", "I22", "spoof > offset > correction", "street reference for route offset", "occupied geofence context", "clear"),
    ("L6-3_v2", "3/3/4/3/8", "I22", "spoof variant > recovery", "moving landmarks", "ground context", "clear"),
    ("L6-4_v1", "3/3/4/3/10", "I22", "jamming > safe hold > channel", "unaffected traffic", "facility/road background", "clear"),
    ("L6-4_v2", "3/3/4/3/10", "I22", "jamming variant > land", "road context", "passive scale/occupancy", "clear"),
    ("L6-5_v1", "3/3/4/3/8", "I22", "GCS intrusion > lockout", "traffic below abnormal path", "operator-area context", "clear"),
    ("L6-5_v2", "3/3/4/3/8", "I22", "intrusion variant > secure recovery", "steady context", "facility pedestrians", "clear"),
    ("X1_rain_to_c2loss_to_forced_landing", "3/3/10/3/8", "I10", "rain + C2 loss > forced landing", "emergency response/hold", "crowd evade/recover", "rain"),
    ("X2_gnss_spoof_to_geofence_violation", "3/3/4/2/8", "I22", "spoof > NFZ violation alert", "street reference and risk context", "occupied zone context", "clear"),
    ("X3_pedestrian_fall_to_emergency_response", "3/3/6/2/6", "I10", "fall > responder > handoff", "ambulance/yield chain", "fallen + bystanders", "clear"),
    ("X4_fog_to_uav_conflict", "3/3/4/2/10", "I22", "fog > UAV conflict > evasion", "low-visibility traffic", "fog-context pedestrians", "fog"),
    ("X5_comm_failure_to_pad_contention", "3/3/4/5/10", "I18", "station fail > pad contention", "facility access traffic", "waiting near facility", "clear"),
    ("X6_crowd_evacuation_to_airspace_lockdown", "3/3/10/3/12", "I10", "crowd evac > NFZ lockdown", "perimeter vehicles move/hold", "evacuation to safe zone", "clear/light smoke"),
)


def _parse_counts(value: str) -> tuple[int, int, int, int, int]:
    parts = value.split("/")
    if len(parts) != 5:
        raise ValueError(f"Invalid U/V/P/F/L count contract: {value}")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def _parse_inspect(value: str) -> float:
    text = value.strip().upper()
    if not text.startswith("I"):
        raise ValueError(f"Invalid inspect altitude code: {value}")
    return float(text[1:])


def _default_capture_boundary_contract(scenario_id: str) -> CaptureBoundaryContract:
    return CaptureBoundaryContract(
        boundary_id=f"capture_boundary.{scenario_id}.primary",
        geometry_source="event_entity",
        anchor_entity_roles=("capture_boundary", "hazard_zone", "airspace_constraint"),
        boundary_role="capture_boundary",
        z_policy="ground_to_event_altitude",
    )


def _default_pad_boundary_policy(scenario_id: str) -> PadBoundaryPolicy:
    if scenario_id.startswith(("L2-4_", "X5_")):
        return PadBoundaryPolicy(
            default="outside_allowed",
            inside_required_for=("pad_contention", "priority_landing_arbitration"),
        )
    return PadBoundaryPolicy(default="outside_allowed", inside_required_for=())


def _default_inspect_contract(inspect_altitude_m: float) -> InspectContract:
    return InspectContract(
        role="U_inspect",
        altitude_code=f"I{int(round(inspect_altitude_m))}",
        altitude_m=inspect_altitude_m,
        min_path_length_m=80.0,
        required_presence="episode_full_duration",
        motion_policy="fixed_altitude_closed_loop",
        corridor_policy="inspect_capture_corridor",
        fov_coverage_required=True,
        sensor_profile_required=True,
    )


def required_intents_for_event_text(required_event: str) -> tuple[str, ...]:
    stages = [stage.strip() for stage in str(required_event or "").split(">") if stage.strip()]
    intents: list[str] = []
    for stage in stages:
        explicit = canonical_intent_from_text(stage)
        if explicit and explicit not in intents:
            intents.append(explicit)
    return tuple(intents)


def _build_contracts() -> dict[str, EpisodeContract]:
    contracts: dict[str, EpisodeContract] = {}
    for scenario_id, counts, inspect, required_event, vehicle_role, pedestrian_role, weather in _ROWS:
        uav, vehicle, pedestrian, facility, logical = _parse_counts(counts)
        inspect_altitude_m = _parse_inspect(inspect)
        contracts[scenario_id] = EpisodeContract(
            scenario_id=scenario_id,
            uav=uav,
            vehicle=vehicle,
            pedestrian=pedestrian,
            facility=facility,
            logical=logical,
            inspect_altitude_m=inspect_altitude_m,
            required_event=required_event,
            required_intents=required_intents_for_event_text(required_event),
            capture_boundary=_default_capture_boundary_contract(scenario_id),
            uav_boundary_crossing_required=True,
            pad_policy=_default_pad_boundary_policy(scenario_id),
            inspect=_default_inspect_contract(inspect_altitude_m),
            vehicle_role=vehicle_role,
            pedestrian_role=pedestrian_role,
            weather=weather,
        )
    return contracts


EPISODE_CONTRACTS: dict[str, EpisodeContract] = _build_contracts()


def normalize_scenario_id(value: str | Path) -> str:
    text = str(value)
    name = Path(text).name
    if "__seed" in name:
        name = name.split("__seed", 1)[0]
    if name in EPISODE_CONTRACTS:
        return name
    if text in EPISODE_CONTRACTS:
        return text
    return name


def get_contract(value: str | Path) -> EpisodeContract:
    scenario_id = normalize_scenario_id(value)
    try:
        return EPISODE_CONTRACTS[scenario_id]
    except KeyError as exc:
        raise KeyError(f"No low-altitude semantic event-chain contract for {value!s}") from exc


def contract_payload(contract: EpisodeContract) -> dict[str, Any]:
    return {
        "schema": "low_altitude_event_chain_contract_v1",
        "scenario_id": contract.scenario_id,
        "exact_counts": contract.counts,
        "capture_boundary": {
            "boundary_id": contract.capture_boundary.boundary_id,
            "geometry_source": contract.capture_boundary.geometry_source,
            "anchor_entity_roles": list(contract.capture_boundary.anchor_entity_roles),
            "boundary_role": contract.capture_boundary.boundary_role,
            "z_policy": contract.capture_boundary.z_policy,
        },
        "inspect": {
            "role": contract.inspect.role,
            "altitude_code": contract.inspect.altitude_code,
            "altitude_m": contract.inspect.altitude_m,
            "min_path_length_m": contract.inspect.min_path_length_m,
            "required_presence": contract.inspect.required_presence,
            "motion_policy": contract.inspect.motion_policy,
            "corridor_policy": contract.inspect.corridor_policy,
            "fov_coverage_required": contract.inspect.fov_coverage_required,
            "sensor_profile_required": contract.inspect.sensor_profile_required,
        },
        "pad_boundary_policy": {
            "default": contract.pad_policy.default,
            "inside_required_for": list(contract.pad_policy.inside_required_for),
        },
        "uav_boundary_crossing_required": contract.uav_boundary_crossing_required,
        "inspect_fov_coverage_required": contract.inspect.fov_coverage_required,
        "required_event": contract.required_event,
        "required_intents": list(contract.required_intents),
        "background_semantics": {
            "vehicle_role": contract.vehicle_role,
            "pedestrian_role": contract.pedestrian_role,
            "policy": "semantic_context_not_decoration",
        },
        "weather": contract.weather,
        "determinism": {
            "fallback": "forbidden",
            "guessing": "forbidden",
            "compatibility_paths": "forbidden",
        },
    }


def all_contracts() -> tuple[EpisodeContract, ...]:
    return tuple(EPISODE_CONTRACTS[key] for key in EPISODE_CONTRACTS)


def event_text_haystack(event: dict[str, Any]) -> str:
    payload = dict(event.get("payload") or {})
    chain_id = str(event.get("chain_id") or "")
    is_takeoff = chain_id.startswith("lifecycle_takeoff")
    semantic_chain_id = "" if is_takeoff else chain_id
    semantic_event_id = "" if is_takeoff else str(event.get("event_id") or "")
    semantic_source_event_id = "" if is_takeoff else str(event.get("source_event_id") or "")
    semantic_payload_event_id = "" if is_takeoff else str(payload.get("event_id") or "")
    parts = [
        semantic_event_id,
        semantic_source_event_id,
        semantic_chain_id,
        semantic_payload_event_id,
        payload.get("title"),
        payload.get("category"),
        payload.get("phase"),
        payload.get("source_kind"),
    ]
    return normalize_semantic_text(" ".join(str(part or "") for part in parts))


def canonical_event_intents(event: dict[str, Any]) -> set[str]:
    payload = dict(event.get("payload") or {})
    explicit = _intent_values(event.get("intent"))
    if explicit:
        return {canonical_intent_from_text(item) or normalize_intent(item) for item in explicit}
    payload_intent = _intent_values(payload.get("intent"))
    if payload_intent:
        return {canonical_intent_from_text(item) or normalize_intent(item) for item in payload_intent}

    haystack = event_text_haystack(event)
    matched: set[str] = set()
    for intent, aliases in INTENT_ALIASES.items():
        if normalize_intent(intent) in haystack.replace(" ", "_") or any(normalize_semantic_text(alias) in haystack for alias in aliases):
            matched.add(intent)
    return matched


def required_intent_sequence_matches(
    required_intents: tuple[str, ...],
    events: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    if not required_intents:
        return True, []
    matches: list[str] = []
    index = 0
    for required in required_intents:
        required_intent = normalize_intent(required)
        found = False
        while index < len(events):
            if required_intent in canonical_event_intents(events[index]):
                matches.append(
                    str(
                        events[index].get("topic")
                        or events[index].get("source_event_id")
                        or events[index].get("event_id")
                        or ""
                    )
                )
                found = True
                index += 1
                break
            index += 1
        if not found:
            return False, matches
    return True, matches
