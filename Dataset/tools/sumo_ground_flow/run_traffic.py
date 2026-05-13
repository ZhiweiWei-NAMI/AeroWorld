"""Run Donghu SUMO traffic and export truth-frame compatible samples."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Sequence

from .coordinates import ensure_sumo_tools_path
from .incident_plan import (
    DEFAULT_SCENARIOS_ROOT,
    DEFAULT_SUMO_NET_XML,
    build_incident_plan,
    incident_records,
    load_incident_plan,
    vehicle_class_for_edge,
    write_incident_plan,
)
from .planner import SumoEdge, SumoGroundFlowPlanner


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT / "Dataset" / "sumo_outputs" / "donghu_traffic_270s"
DEFAULT_SUMO_BIN = Path("E:/sumo-1.8.0/bin/sumo.exe")
DEFAULT_DURATION_S = 270.0
DEFAULT_STEP_LENGTH_S = 0.1
DEFAULT_SAMPLE_PERIOD_S = 0.5
DEFAULT_MAX_VEHICLES = 200
SIGNAL_HAZARD = 8
SIGNAL_EMERGENCY = 10


def _json_default(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object is not JSON serializable: {type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _truth_yaw_from_sumo_angle(angle_deg: float) -> float:
    yaw = 90.0 - float(angle_deg)
    while yaw > 180.0:
        yaw -= 360.0
    while yaw <= -180.0:
        yaw += 360.0
    return yaw


def _round_list(values: Sequence[float], digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]


def _edge_allows_class(edge: SumoEdge, vehicle_class: str) -> bool:
    if edge.allow:
        return vehicle_class in edge.allow
    forbidden_type = any(token in edge.edge_type for token in ("footway", "pedestrian", "path", "steps", "rail"))
    return vehicle_class not in edge.disallow and not forbidden_type


class RouteFactory:
    def __init__(self, planner: SumoGroundFlowPlanner, rng: random.Random) -> None:
        self.planner = planner
        self.rng = rng
        self.edges_by_class: dict[str, list[SumoEdge]] = {
            "passenger": [edge for edge in planner.edges.values() if _edge_allows_class(edge, "passenger")],
            "delivery": [edge for edge in planner.edges.values() if _edge_allows_class(edge, "delivery")],
        }

    def path_from_edge(self, edge_id: str, vehicle_class: str, *, min_edges: int = 4, max_edges: int = 12) -> list[str]:
        if edge_id not in self.planner.edges:
            return self.random_path(vehicle_class, min_edges=min_edges, max_edges=max_edges)
        path = [edge_id]
        current = edge_id
        wanted = self.rng.randint(min_edges, max_edges)
        for _ in range(wanted - 1):
            candidates = [
                edge_id
                for edge_id in self.planner.adjacency.get(current, [])
                if edge_id in self.planner.edges and _edge_allows_class(self.planner.edges[edge_id], vehicle_class)
            ]
            if not candidates:
                break
            current = self.rng.choice(candidates)
            if current in path[-3:]:
                break
            path.append(current)
        return path

    def random_path(self, vehicle_class: str, *, min_edges: int = 4, max_edges: int = 14) -> list[str]:
        edges = self.edges_by_class.get(vehicle_class) or self.edges_by_class["passenger"]
        for _ in range(80):
            start = self.rng.choice(edges)
            if start.length_m < 20.0:
                continue
            path = self.path_from_edge(start.edge_id, vehicle_class, min_edges=min_edges, max_edges=max_edges)
            if len(path) >= 2:
                return path
        return [self.rng.choice(edges).edge_id]


@dataclass
class RestoreAction:
    kind: str
    target_id: str
    end_s: float
    payload: Any


class TrafficRunner:
    def __init__(
        self,
        *,
        planner: SumoGroundFlowPlanner,
        incident_plan: dict[str, Any],
        output_dir: Path,
        sumo_bin: Path,
        duration_s: float,
        step_length_s: float,
        sample_period_s: float,
        max_vehicles: int,
        seed: int,
    ) -> None:
        self.planner = planner
        self.incident_plan = incident_plan
        self.output_dir = output_dir
        self.sumo_bin = sumo_bin
        self.duration_s = float(duration_s)
        self.step_length_s = float(step_length_s)
        self.sample_period_s = float(sample_period_s)
        self.max_vehicles = int(max_vehicles)
        self.rng = random.Random(seed)
        self.route_factory = RouteFactory(planner, self.rng)
        self.background_ids: set[str] = set()
        self.controlled_ids: set[str] = set()
        self.applied_incidents: set[str] = set()
        self.restores: list[RestoreAction] = []
        self.route_counter = 0
        self.vehicle_counter = 0
        self.active_incident_by_id = {str(item["incident_id"]): item for item in incident_records(incident_plan)}

    def run(self) -> dict[str, Any]:
        ensure_sumo_tools_path()
        import traci  # type: ignore

        cmd = [
            str(self.sumo_bin),
            "-n",
            str(self.planner.net_xml),
            "--step-length",
            str(self.step_length_s),
            "--no-step-log",
            "true",
            "--no-warnings",
            "true",
            "--time-to-teleport",
            "-1",
            "--collision.action",
            "warn",
        ]
        traci.start(cmd)
        try:
            self._ensure_vehicle_types(traci)
            frames_path = self.output_dir / "sumo_traffic_frames.jsonl"
            frames_path.parent.mkdir(parents=True, exist_ok=True)
            sample_every_steps = max(1, int(round(self.sample_period_s / self.step_length_s)))
            total_steps = int(round(self.duration_s / self.step_length_s))
            sample_count = 0
            with frames_path.open("w", encoding="utf-8") as handle:
                self._write_frame(handle, traci, tick=0)
                sample_count += 1
                for step in range(1, total_steps + 1):
                    sim_time_before_step = round((step - 1) * self.step_length_s, 9)
                    self._apply_due_incidents(traci, sim_time_before_step)
                    self._manage_background_demand(traci, sim_time_before_step)
                    traci.simulationStep(round(step * self.step_length_s, 9))
                    self._restore_due_actions(traci, float(traci.simulation.getTime()))
                    self._trim_vehicle_cap(traci)
                    if step % sample_every_steps == 0:
                        self._write_frame(handle, traci, tick=step)
                        sample_count += 1
            return self._manifest(sample_count=sample_count)
        finally:
            traci.close(False)

    def _ensure_vehicle_types(self, traci: Any) -> None:
        for source, type_id, vehicle_class, color in (
            ("DEFAULT_VEHTYPE", "aero_passenger", "passenger", (40, 160, 255, 255)),
            ("DEFAULT_VEHTYPE", "aero_delivery", "delivery", (255, 180, 40, 255)),
            ("DEFAULT_VEHTYPE", "aero_emergency", "emergency", (255, 40, 40, 255)),
        ):
            try:
                traci.vehicletype.copy(source, type_id)
            except Exception:
                pass
            try:
                traci.vehicletype.setVehicleClass(type_id, vehicle_class)
                traci.vehicletype.setColor(type_id, color)
            except Exception:
                pass

    def _new_route(self, traci: Any, edges: list[str]) -> str:
        route_id = f"sumo_route_{self.route_counter:06d}"
        self.route_counter += 1
        traci.route.add(route_id, edges)
        return route_id

    def _vehicle_type_for_class(self, vehicle_class: str, *, emergency: bool = False) -> str:
        if emergency:
            return "aero_emergency"
        if vehicle_class == "delivery":
            return "aero_delivery"
        return "aero_passenger"

    def _spawn_vehicle(
        self,
        traci: Any,
        *,
        vehicle_id: str,
        route_edges: list[str],
        vehicle_class: str,
        depart_pos_m: float,
        depart_speed_mps: float,
        emergency: bool = False,
        controlled: bool = False,
    ) -> bool:
        try:
            route_id = self._new_route(traci, route_edges)
            traci.vehicle.addFull(
                vehicle_id,
                route_id,
                typeID=self._vehicle_type_for_class(vehicle_class, emergency=emergency),
                depart="now",
                departLane="best",
                departPos=f"{max(0.0, float(depart_pos_m)):.3f}",
                departSpeed=f"{max(0.0, float(depart_speed_mps)):.3f}",
            )
            if controlled:
                self.controlled_ids.add(vehicle_id)
            else:
                self.background_ids.add(vehicle_id)
            return True
        except Exception:
            return False

    def _manage_background_demand(self, traci: Any, sim_time_s: float) -> None:
        ids = set(traci.vehicle.getIDList())
        loaded_ids = set(traci.simulation.getLoadedIDList())
        arrived_ids = set(traci.simulation.getArrivedIDList())
        self.background_ids -= arrived_ids
        target = self._active_vehicle_target(sim_time_s)
        controlled_active = {vehicle_id for vehicle_id in ids if vehicle_id in self.controlled_ids}
        target_background = max(0, target - len(controlled_active))
        active_background = {vehicle_id for vehicle_id in ids if vehicle_id in self.background_ids}
        pending_background = {vehicle_id for vehicle_id in loaded_ids if vehicle_id in self.background_ids}
        current_background = len(active_background) + len(pending_background)
        if sim_time_s >= 180.0 and current_background > target_background + 2:
            removable = sorted(active_background)
            remove_count = min(20, max(0, current_background - target_background))
            for vehicle_id in removable[:remove_count]:
                try:
                    traci.vehicle.remove(vehicle_id)
                    self.background_ids.discard(vehicle_id)
                except Exception:
                    pass
            return
        additions = min(5, max(0, target_background - current_background))
        for _ in range(additions):
            vehicle_id = f"veh_bg_{self.vehicle_counter:06d}"
            self.vehicle_counter += 1
            route_edges = self.route_factory.random_path("passenger", min_edges=6, max_edges=18)
            if not self._spawn_vehicle(
                traci,
                vehicle_id=vehicle_id,
                route_edges=route_edges,
                vehicle_class="passenger",
                depart_pos_m=0.0,
                depart_speed_mps=8.0,
            ):
                continue

    def _active_vehicle_target(self, sim_time_s: float) -> int:
        if sim_time_s < 0.0:
            return 0
        if sim_time_s < 90.0:
            return int(round(self.max_vehicles * sim_time_s / 90.0))
        if sim_time_s < 180.0:
            return self.max_vehicles
        if sim_time_s <= self.duration_s:
            return max(0, int(round(self.max_vehicles * (1.0 - (sim_time_s - 180.0) / 90.0))))
        return 0

    def _apply_due_incidents(self, traci: Any, sim_time_s: float) -> None:
        for incident in incident_records(self.incident_plan):
            incident_id = str(incident["incident_id"])
            if incident_id in self.applied_incidents:
                continue
            if sim_time_s + 1e-9 < float(incident["start_s"]):
                continue
            self._apply_incident(traci, incident)
            self.applied_incidents.add(incident_id)

    def _apply_incident(self, traci: Any, incident: dict[str, Any]) -> None:
        accident_class = str(incident.get("accident_class") or "")
        if accident_class == "traffic_light_all_red_fault":
            self._apply_all_red_fault(traci, incident)
        elif accident_class in {"lane_closure_roadwork", "hazmat_isolation_zone"}:
            self._apply_lane_or_edge_slowdown(traci, incident)
        elif accident_class == "weather_speed_degradation":
            self._apply_weather_slowdown(traci, incident)
        elif accident_class == "emergency_vehicle_priority":
            self._apply_emergency_priority(traci, incident)
        elif accident_class == "medical_response_dispatch":
            self._apply_medical_dispatch(traci, incident)
        else:
            self._apply_stop_incident(traci, incident)

    def _anchor(self, incident: dict[str, Any]) -> dict[str, Any]:
        return dict(incident.get("anchor") or {})

    def _incident_vehicle_ids(self, incident: dict[str, Any], *, count: int = 1) -> list[str]:
        raw_ids = [str(item) for item in incident.get("affected_vehicle_ids") or [] if str(item)]
        if not raw_ids:
            raw_ids = [f"{incident['incident_id']}.vehicle_{index}" for index in range(count)]
        while len(raw_ids) < count:
            raw_ids.append(f"{incident['incident_id']}.vehicle_{len(raw_ids)}")
        return [item.replace(" ", "_") for item in raw_ids[:count]]

    def _spawn_controlled_near_anchor(
        self,
        traci: Any,
        incident: dict[str, Any],
        vehicle_id: str,
        *,
        approach_m: float = 45.0,
        speed_mps: float = 8.0,
        emergency: bool = False,
    ) -> bool:
        anchor = self._anchor(incident)
        edge_id = str(anchor["sumo_edge_id"])
        vehicle_class = str(anchor.get("vehicle_class") or "passenger")
        route_edges = self.route_factory.path_from_edge(edge_id, vehicle_class, min_edges=3, max_edges=10)
        depart_pos = max(0.0, float(anchor.get("lane_position_m") or 0.0) - approach_m)
        return self._spawn_vehicle(
            traci,
            vehicle_id=vehicle_id,
            route_edges=route_edges,
            vehicle_class=vehicle_class,
            depart_pos_m=depart_pos,
            depart_speed_mps=speed_mps,
            emergency=emergency,
            controlled=True,
        )

    def _clamped_stop_pos(self, anchor: dict[str, Any], *, offset_m: float = 0.0) -> float:
        edge_id = str(anchor.get("sumo_edge_id") or "")
        raw = float(anchor.get("lane_position_m") or 1.0) + float(offset_m)
        edge = self.planner.edges.get(edge_id)
        if edge is None:
            return max(1.0, raw)
        return max(1.0, min(float(edge.length_m) - 1.0, raw))

    def _apply_all_red_fault(self, traci: Any, incident: dict[str, Any]) -> None:
        traffic_light = dict(incident.get("traffic_light") or {})
        tls_id = str(traffic_light.get("traffic_light_id") or "")
        if not tls_id:
            return
        try:
            original_state = traci.trafficlight.getRedYellowGreenState(tls_id)
            original_phase = traci.trafficlight.getPhase(tls_id)
            traci.trafficlight.setRedYellowGreenState(tls_id, "r" * len(original_state))
            self.restores.append(
                RestoreAction(
                    kind="traffic_light",
                    target_id=tls_id,
                    end_s=float(incident["end_s"]),
                    payload={"state": original_state, "phase": original_phase},
                )
            )
        except Exception:
            return

    def _apply_lane_or_edge_slowdown(self, traci: Any, incident: dict[str, Any]) -> None:
        anchor = self._anchor(incident)
        lane_id = str(anchor.get("sumo_lane_id") or "")
        edge_id = str(anchor.get("sumo_edge_id") or "")
        try:
            original = traci.lane.getMaxSpeed(lane_id)
            traci.lane.setMaxSpeed(lane_id, 0.4)
            self.restores.append(RestoreAction("lane_speed", lane_id, float(incident["end_s"]), original))
        except Exception:
            pass
        for vehicle_id in self._incident_vehicle_ids(incident, count=1):
            if self._spawn_controlled_near_anchor(traci, incident, vehicle_id, approach_m=35.0, speed_mps=5.0):
                try:
                    traci.vehicle.slowDown(vehicle_id, 1.0, 10.0)
                    traci.vehicle.setStop(
                        vehicle_id,
                        edge_id,
                        self._clamped_stop_pos(anchor),
                        int(anchor.get("lane_index") or 0),
                        duration=8.0,
                    )
                except Exception:
                    pass

    def _apply_weather_slowdown(self, traci: Any, incident: dict[str, Any]) -> None:
        for vehicle_id in traci.vehicle.getIDList():
            if vehicle_id in self.controlled_ids:
                continue
            try:
                traci.vehicle.slowDown(vehicle_id, 3.5, 20.0)
                traci.vehicle.setMaxSpeed(vehicle_id, 4.5)
                self.restores.append(RestoreAction("vehicle_max_speed", vehicle_id, float(incident["end_s"]), 13.89))
            except Exception:
                pass

    def _apply_emergency_priority(self, traci: Any, incident: dict[str, Any]) -> None:
        ids = self._incident_vehicle_ids(incident, count=2)
        ambulance_id = ids[0]
        yield_id = ids[1]
        if self._spawn_controlled_near_anchor(traci, incident, ambulance_id, approach_m=70.0, speed_mps=12.0, emergency=True):
            try:
                traci.vehicle.setSignals(ambulance_id, SIGNAL_EMERGENCY)
            except Exception:
                pass
        if self._spawn_controlled_near_anchor(traci, incident, yield_id, approach_m=20.0, speed_mps=4.0):
            try:
                traci.vehicle.slowDown(yield_id, 1.2, 8.0)
                traci.vehicle.setSignals(yield_id, SIGNAL_HAZARD)
            except Exception:
                pass

    def _apply_medical_dispatch(self, traci: Any, incident: dict[str, Any]) -> None:
        vehicle_id = self._incident_vehicle_ids(incident, count=1)[0]
        if self._spawn_controlled_near_anchor(traci, incident, vehicle_id, approach_m=80.0, speed_mps=11.0, emergency=True):
            try:
                traci.vehicle.setSignals(vehicle_id, SIGNAL_EMERGENCY)
            except Exception:
                pass

    def _apply_stop_incident(self, traci: Any, incident: dict[str, Any]) -> None:
        anchor = self._anchor(incident)
        edge_id = str(anchor.get("sumo_edge_id") or "")
        lane_index = int(anchor.get("lane_index") or 0)
        ids = self._incident_vehicle_ids(incident, count=2 if incident.get("accident_class") == "vehicle_intersection_conflict" else 1)
        for offset, vehicle_id in enumerate(ids):
            approach = 55.0 + offset * 18.0
            if not self._spawn_controlled_near_anchor(traci, incident, vehicle_id, approach_m=approach, speed_mps=8.0 - offset):
                continue
            try:
                traci.vehicle.slowDown(vehicle_id, 0.5, 5.0)
                traci.vehicle.setSignals(vehicle_id, SIGNAL_HAZARD)
                traci.vehicle.setStop(
                    vehicle_id,
                    edge_id,
                    self._clamped_stop_pos(anchor, offset_m=offset * 2.0),
                    lane_index,
                    duration=12.0,
                )
            except Exception:
                pass

    def _restore_due_actions(self, traci: Any, sim_time_s: float) -> None:
        pending: list[RestoreAction] = []
        for restore in self.restores:
            if sim_time_s + 1e-9 < restore.end_s:
                pending.append(restore)
                continue
            try:
                if restore.kind == "traffic_light":
                    traci.trafficlight.setRedYellowGreenState(restore.target_id, restore.payload["state"])
                    traci.trafficlight.setPhase(restore.target_id, int(restore.payload["phase"]))
                elif restore.kind == "lane_speed":
                    traci.lane.setMaxSpeed(restore.target_id, float(restore.payload))
                elif restore.kind == "vehicle_max_speed" and restore.target_id in traci.vehicle.getIDList():
                    traci.vehicle.setMaxSpeed(restore.target_id, float(restore.payload))
            except Exception:
                pass
        self.restores = pending

    def _trim_vehicle_cap(self, traci: Any) -> None:
        ids = set(traci.vehicle.getIDList())
        excess = len(ids) - self.max_vehicles
        if excess <= 0:
            return
        removable = sorted(vehicle_id for vehicle_id in ids if vehicle_id in self.background_ids)
        for vehicle_id in removable[:excess]:
            try:
                traci.vehicle.remove(vehicle_id)
                self.background_ids.discard(vehicle_id)
            except Exception:
                pass

    def _write_frame(self, handle: Any, traci: Any, *, tick: int) -> None:
        sim_time_s = round(float(traci.simulation.getTime()), 6)
        tick_10hz = int(round(sim_time_s / self.step_length_s))
        frame = {
            "schema_name": "sumo_traffic_frame",
            "schema_version": "v1",
            "map_id": "donghu_road_topo",
            "tick": tick_10hz,
            "sample_tick": int(tick),
            "tick_hz": int(round(1.0 / self.step_length_s)),
            "sample_period_s": round(self.sample_period_s, 6),
            "sim_time_s": sim_time_s,
            "vehicles": self._vehicle_records(traci),
            "traffic_lights": self._traffic_light_records(traci),
            "active_incidents": self._active_incidents(sim_time_s),
        }
        handle.write(json.dumps(frame, ensure_ascii=False, default=_json_default, sort_keys=True) + "\n")

    def _vehicle_records(self, traci: Any) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for vehicle_id in sorted(traci.vehicle.getIDList()):
            try:
                sx, sy = traci.vehicle.getPosition(vehicle_id)
                tx, ty = self.planner.coordinate_mapper.sumo_xy_to_truth_xy(float(sx), float(sy))
                speed = float(traci.vehicle.getSpeed(vehicle_id))
                accel = float(traci.vehicle.getAcceleration(vehicle_id))
                angle = float(traci.vehicle.getAngle(vehicle_id))
                records.append(
                    {
                        "vehicle_id": vehicle_id,
                        "vehicle_type": traci.vehicle.getTypeID(vehicle_id),
                        "route_id": traci.vehicle.getRouteID(vehicle_id),
                        "sumo_edge_id": traci.vehicle.getRoadID(vehicle_id),
                        "sumo_lane_id": traci.vehicle.getLaneID(vehicle_id),
                        "lane_position_m": round(float(traci.vehicle.getLanePosition(vehicle_id)), 6),
                        "sumo_xy_m": _round_list([sx, sy]),
                        "truth_position_enu_m": _round_list([tx, ty, 0.0]),
                        "truth_yaw_deg": round(_truth_yaw_from_sumo_angle(angle), 6),
                        "sumo_angle_deg": round(angle, 6),
                        "speed_mps": round(speed, 6),
                        "accel_mps2": round(accel, 6),
                        "signals": int(traci.vehicle.getSignals(vehicle_id)),
                        "dimensions_m": {
                            "length": round(float(traci.vehicle.getLength(vehicle_id)), 6),
                            "width": round(float(traci.vehicle.getWidth(vehicle_id)), 6),
                            "height": round(float(traci.vehicle.getHeight(vehicle_id)), 6),
                        },
                        "source": "sumo_traci",
                        "control_role": "incident_controlled" if vehicle_id in self.controlled_ids else "background",
                    }
                )
            except Exception:
                continue
        return records

    def _traffic_light_records(self, traci: Any) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for tls_id in sorted(traci.trafficlight.getIDList()):
            try:
                controlled_links = traci.trafficlight.getControlledLinks(tls_id)
                records.append(
                    {
                        "tls_id": tls_id,
                        "program_id": traci.trafficlight.getProgram(tls_id),
                        "phase_index": int(traci.trafficlight.getPhase(tls_id)),
                        "state": traci.trafficlight.getRedYellowGreenState(tls_id),
                        "next_switch_s": round(float(traci.trafficlight.getNextSwitch(tls_id)), 6),
                        "controlled_links": json.loads(json.dumps(controlled_links, default=_json_default)),
                    }
                )
            except Exception:
                continue
        return records

    def _active_incidents(self, sim_time_s: float) -> list[dict[str, Any]]:
        active: list[dict[str, Any]] = []
        for incident in incident_records(self.incident_plan):
            if float(incident["start_s"]) <= sim_time_s <= float(incident["end_s"]):
                active.append(
                    {
                        "incident_id": incident["incident_id"],
                        "episode_scenario_id": incident["episode_scenario_id"],
                        "episode_event_id": incident["episode_event_id"],
                        "intent": incident["intent"],
                        "accident_class": incident["accident_class"],
                        "start_s": incident["start_s"],
                        "end_s": incident["end_s"],
                        "anchor": incident["anchor"],
                        "injection_method": incident["injection_method"],
                        "affected_vehicle_ids": incident.get("affected_vehicle_ids") or [],
                    }
                )
        return active

    def _manifest(self, *, sample_count: int) -> dict[str, Any]:
        return {
            "schema_name": "sumo_traffic_manifest",
            "schema_version": "v1",
            "map_id": "donghu_road_topo",
            "net_xml": str(self.planner.net_xml),
            "coordinate_mapper": self.planner.coordinate_mapper.fit_summary,
            "duration_s": self.duration_s,
            "step_length_s": self.step_length_s,
            "tick_hz": int(round(1.0 / self.step_length_s)),
            "sample_period_s": self.sample_period_s,
            "sample_every_ticks": int(round(self.sample_period_s / self.step_length_s)),
            "sample_count": int(sample_count),
            "max_vehicles": self.max_vehicles,
            "demand_profile": {
                "0_90_s": "ramp active vehicles from 0 to max_vehicles",
                "90_180_s": "hold max_vehicles",
                "180_270_s": "ramp active vehicles down to 0",
            },
            "incident_plan": {
                "incident_count": int(self.incident_plan.get("incident_count") or 0),
                "schema_name": self.incident_plan.get("schema_name"),
                "schema_version": self.incident_plan.get("schema_version"),
            },
            "outputs": {
                "manifest": str(self.output_dir / "sumo_traffic_manifest.json"),
                "incident_plan": str(self.output_dir / "sumo_incident_plan.json"),
                "frames": str(self.output_dir / "sumo_traffic_frames.jsonl"),
            },
        }


def run_traffic(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    incident_plan_path: Path | None = None,
    scenarios_root: Path = DEFAULT_SCENARIOS_ROOT,
    net_xml: Path = DEFAULT_SUMO_NET_XML,
    sumo_bin: Path = DEFAULT_SUMO_BIN,
    duration_s: float = DEFAULT_DURATION_S,
    step_length_s: float = DEFAULT_STEP_LENGTH_S,
    sample_period_s: float = DEFAULT_SAMPLE_PERIOD_S,
    max_vehicles: int = DEFAULT_MAX_VEHICLES,
    seed: int = 7,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if overwrite:
        for name in ("sumo_traffic_manifest.json", "sumo_incident_plan.json", "sumo_traffic_frames.jsonl"):
            path = output_dir / name
            if path.exists():
                path.unlink()
    planner = SumoGroundFlowPlanner(net_xml)
    if incident_plan_path is not None:
        incident_plan = load_incident_plan(incident_plan_path)
    else:
        incident_plan = build_incident_plan(
            scenarios_root=scenarios_root,
            net_xml=net_xml,
            planner=planner,
            duration_s=duration_s,
        )
    write_incident_plan(incident_plan, output_dir / "sumo_incident_plan.json")
    runner = TrafficRunner(
        planner=planner,
        incident_plan=incident_plan,
        output_dir=output_dir,
        sumo_bin=sumo_bin,
        duration_s=duration_s,
        step_length_s=step_length_s,
        sample_period_s=sample_period_s,
        max_vehicles=max_vehicles,
        seed=seed,
    )
    manifest = runner.run()
    _write_json(output_dir / "sumo_traffic_manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Donghu SUMO traffic and export 0.5s samples.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--incident-plan", type=Path, default=None)
    parser.add_argument("--scenarios-root", type=Path, default=DEFAULT_SCENARIOS_ROOT)
    parser.add_argument("--net-xml", type=Path, default=DEFAULT_SUMO_NET_XML)
    parser.add_argument("--sumo-bin", type=Path, default=DEFAULT_SUMO_BIN)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--step-length-s", type=float, default=DEFAULT_STEP_LENGTH_S)
    parser.add_argument("--sample-period-s", type=float, default=DEFAULT_SAMPLE_PERIOD_S)
    parser.add_argument("--max-vehicles", type=int, default=DEFAULT_MAX_VEHICLES)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_traffic(
        output_dir=args.output_dir,
        incident_plan_path=args.incident_plan,
        scenarios_root=args.scenarios_root,
        net_xml=args.net_xml,
        sumo_bin=args.sumo_bin,
        duration_s=args.duration_s,
        step_length_s=args.step_length_s,
        sample_period_s=args.sample_period_s,
        max_vehicles=args.max_vehicles,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
