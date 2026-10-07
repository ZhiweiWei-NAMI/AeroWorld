"""Authoritative SUMO lane, speed-limit, and signal context.

The road network owns lane speeds, lane lengths, and traffic-light link
indices.  Per-frame SUMO truth owns the current signal program state.  This
module joins those two sources without consulting semantic event labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

from shapely.geometry import LineString, Point


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROAD_DERIVED_NET_XML = (
    PROJECT_ROOT
    / "Plugins"
    / "SumoImporter"
    / "Maps"
    / "donghu_road_topo"
    / "source"
    / "road_derived"
    / "road_derived.net.xml"
)
NO_CONTROLLING_SIGNAL = "none"
NO_CONTROLLING_SIGNAL_STATE = "none"
STOP_LINE_TRANSITION_TOLERANCE_M = 2.0


class RoadSignalContextError(ValueError):
    """Raised when SUMO road truth cannot be joined deterministically."""


@dataclass(frozen=True)
class LaneAuthority:
    lane_id: str
    edge_id: str
    speed_mps: float
    length_m: float


@dataclass(frozen=True)
class SignalConnection:
    from_lane_id: str
    to_edge_id: str
    to_lane_id: str
    via_lane_id: str
    tls_id: str
    link_index: int


@dataclass(frozen=True)
class CrosswalkAuthority:
    """A compiled SUMO crossing edge and its physical lane footprint.

    SUMO shape points are lane centerline coordinates in metres. Width is
    the full lane width. The footprint includes its boundary, uses the lane
    endpoints as flat ends, and uses mitred joins between shape segments.
    The road-derived coordinate route is identity XY into truth ENU metres.
    """

    crosswalk_id: str
    lane_id: str
    shape_xy_m: tuple[tuple[float, float], ...]
    width_m: float
    source_ref: str

    def contains_xy(self, position: Sequence[float]) -> bool:
        footprint = LineString(self.shape_xy_m).buffer(
            self.width_m / 2.0, cap_style=2, join_style=2,
        )
        return bool(footprint.covers(Point(float(position[0]), float(position[1]))))


@dataclass(frozen=True)
class RoadSignalContext:
    net_xml: Path
    lanes: Mapping[str, LaneAuthority]
    signal_connections_by_lane: Mapping[str, tuple[SignalConnection, ...]]
    crossing_count: int
    traffic_light_controller_count: int
    signalized_junction_count: int
    crosswalks: Mapping[str, CrosswalkAuthority]

    @classmethod
    def load(cls, net_xml: Path = DEFAULT_ROAD_DERIVED_NET_XML) -> "RoadSignalContext":
        return _load_road_signal_context(str(Path(net_xml).resolve()))

    def require_lane(self, lane_id: str) -> LaneAuthority:
        lane = self.lanes.get(lane_id)
        if lane is None:
            raise RoadSignalContextError(
                f"SUMO lane {lane_id!r} is absent from {self.net_xml}"
            )
        return lane

    def enrich_sumo_vehicle(
        self,
        sumo_vehicle: Mapping[str, Any],
        traffic_light_states: Mapping[str, Mapping[str, Any]],
        *,
        previous_sumo_vehicle: Mapping[str, Any] | None = None,
        previous_traffic_light_states: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return the four regulation-contract fields for one vehicle tick."""

        lane_id = _nonempty_string(sumo_vehicle.get("sumo_lane_id"))
        if lane_id is None:
            raise RoadSignalContextError("active SUMO vehicle lacks sumo_lane_id")
        lane = self.require_lane(lane_id)
        connections = self.signal_connections_by_lane.get(lane_id, ())
        controlling_id, controlling_state = _current_signal_context(
            connections,
            traffic_light_states,
        )
        controlling_connection = connections[0] if connections else None
        crossed_stop_line = False

        if previous_sumo_vehicle is not None:
            previous_lane_id = _nonempty_string(
                previous_sumo_vehicle.get("sumo_lane_id")
            )
            if previous_lane_id is not None and previous_lane_id != lane_id:
                previous_lane = self.require_lane(previous_lane_id)
                transition = _transition_connection(
                    self.signal_connections_by_lane.get(previous_lane_id, ()),
                    lane,
                )
                previous_position = _finite_number(
                    previous_sumo_vehicle.get("lane_position_m")
                )
                approached_stop_line = (
                    previous_position is not None
                    and previous_position
                    >= previous_lane.length_m - STOP_LINE_TRANSITION_TOLERANCE_M
                )
                if transition is not None and approached_stop_line:
                    crossed_stop_line = True
                    controlling_connection = transition
                    controlling_id = transition.tls_id
                    controlling_state = _signal_state_for_connection(
                        transition,
                        previous_traffic_light_states or traffic_light_states,
                    )

        return {
            "lane_id": lane.lane_id,
            "lane_ontology_class_id": "world:RoadLane",
            "lane_instance_source_ref": f"{self.net_xml}#lane={lane.lane_id}",
            "allowed_speed_mps": lane.speed_mps,
            "speed_limit_regulation_id": _road_instance_id(
                "speed_limit_regulation", lane.lane_id
            ),
            "speed_limit_regulation_ontology_class_id": (
                "world:SpeedLimitRegulation"
            ),
            "speed_limit_regulation_source_ref": (
                f"{self.net_xml}#lane={lane.lane_id}&attribute=speed"
            ),
            "following_distance_rule_id": _road_instance_id(
                "following_distance_rule", lane.lane_id
            ),
            "following_distance_rule_ontology_class_id": (
                "world:FollowingDistanceRule"
            ),
            "following_distance_rule_source_ref": (
                f"{self.net_xml}#lane={lane.lane_id}"
            ),
            "controlling_signal_id": controlling_id,
            "controlling_signal_ontology_class_id": (
                "world:TrafficSignal"
                if controlling_connection is not None
                else None
            ),
            "controlling_signal_state": controlling_state,
            "stop_line_id": (
                _stop_line_id(controlling_connection)
                if controlling_connection is not None
                else None
            ),
            "stop_line_ontology_class_id": (
                "world:StopLine" if controlling_connection is not None else None
            ),
            "stop_line_source_ref": (
                _connection_source_ref(self.net_xml, controlling_connection)
                if controlling_connection is not None
                else None
            ),
            "right_of_way_id": (
                _road_instance_id(
                    "right_of_way",
                    controlling_connection.from_lane_id,
                    controlling_connection.tls_id,
                )
                if controlling_connection is not None
                else None
            ),
            "right_of_way_ontology_class_id": (
                "world:RightOfWay" if controlling_connection is not None else None
            ),
            "right_of_way_source_ref": (
                _connection_source_ref(self.net_xml, controlling_connection)
                if controlling_connection is not None
                else None
            ),
            "crossed_stop_line": crossed_stop_line,
        }

    def static_sumo_vehicle_fields(
        self, sumo_vehicle: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Return roster-time road authority without inventing a signal phase."""

        lane_id = _nonempty_string(sumo_vehicle.get("sumo_lane_id"))
        if lane_id is None:
            raise RoadSignalContextError("SUMO roster vehicle lacks sumo_lane_id")
        lane = self.require_lane(lane_id)
        connections = self.signal_connections_by_lane.get(lane_id, ())
        return {
            "lane_id": lane.lane_id,
            "lane_ontology_class_id": "world:RoadLane",
            "lane_instance_source_ref": f"{self.net_xml}#lane={lane.lane_id}",
            "allowed_speed_mps": lane.speed_mps,
            "speed_limit_regulation_id": _road_instance_id(
                "speed_limit_regulation", lane.lane_id
            ),
            "speed_limit_regulation_ontology_class_id": (
                "world:SpeedLimitRegulation"
            ),
            "speed_limit_regulation_source_ref": (
                f"{self.net_xml}#lane={lane.lane_id}&attribute=speed"
            ),
            "following_distance_rule_id": _road_instance_id(
                "following_distance_rule", lane.lane_id
            ),
            "following_distance_rule_ontology_class_id": (
                "world:FollowingDistanceRule"
            ),
            "following_distance_rule_source_ref": (
                f"{self.net_xml}#lane={lane.lane_id}"
            ),
            "controlling_signal_id": (
                connections[0].tls_id if connections else NO_CONTROLLING_SIGNAL
            ),
            "controlling_signal_ontology_class_id": (
                "world:TrafficSignal" if connections else None
            ),
            "controlling_signal_state": "not_sampled_at_roster_level",
            "stop_line_id": _stop_line_id(connections[0]) if connections else None,
            "stop_line_ontology_class_id": "world:StopLine" if connections else None,
            "stop_line_source_ref": (
                _connection_source_ref(self.net_xml, connections[0])
                if connections
                else None
            ),
            "right_of_way_id": (
                _road_instance_id(
                    "right_of_way",
                    connections[0].from_lane_id,
                    connections[0].tls_id,
                )
                if connections
                else None
            ),
            "right_of_way_ontology_class_id": (
                "world:RightOfWay" if connections else None
            ),
            "right_of_way_source_ref": (
                _connection_source_ref(self.net_xml, connections[0])
                if connections
                else None
            ),
            "crossed_stop_line": False,
        }


def _road_instance_id(kind: str, *source_ids: str) -> str:
    if not source_ids or any(not source_id for source_id in source_ids):
        raise RoadSignalContextError(f"{kind} requires non-empty SUMO authority ids")
    return f"sumo_{kind}:" + "|".join(source_ids)


def _stop_line_id(connection: SignalConnection) -> str:
    return _road_instance_id(
        "stop_line",
        connection.from_lane_id,
        connection.tls_id,
        str(connection.link_index),
    )


def _connection_source_ref(net_xml: Path, connection: SignalConnection) -> str:
    return (
        f"{net_xml}#connection={connection.from_lane_id}->{connection.to_lane_id}"
        f"&tls={connection.tls_id}&link_index={connection.link_index}"
    )


@lru_cache(maxsize=4)
def _load_road_signal_context(path_text: str) -> RoadSignalContext:
    net_xml = Path(path_text)
    if not net_xml.is_file():
        raise RoadSignalContextError(f"SUMO road network is missing: {net_xml}")
    root = ET.parse(net_xml).getroot()
    lanes: dict[str, LaneAuthority] = {}
    crosswalks: dict[str, CrosswalkAuthority] = {}
    for edge in root.findall("edge"):
        edge_id = _nonempty_string(edge.get("id"))
        if edge_id is None:
            continue
        for lane in edge.findall("lane"):
            lane_id = _nonempty_string(lane.get("id"))
            speed = _finite_number(lane.get("speed"))
            length = _finite_number(lane.get("length"))
            if lane_id is None or speed is None or length is None:
                raise RoadSignalContextError(
                    f"{net_xml}: edge {edge_id!r} contains an incomplete lane"
                )
            if speed < 0.0 or length < 0.0:
                raise RoadSignalContextError(
                    f"{net_xml}: lane {lane_id!r} has negative speed/length"
                )
            if lane_id in lanes:
                raise RoadSignalContextError(
                    f"{net_xml}: duplicate lane id {lane_id!r}"
                )
            lanes[lane_id] = LaneAuthority(lane_id, edge_id, speed, length)

        if edge.get("function") == "crossing":
            crossing_lanes = edge.findall("lane")
            if len(crossing_lanes) != 1 or edge_id in crosswalks:
                raise RoadSignalContextError(
                    f"{net_xml}: crossing {edge_id!r} needs one unique lane"
                )
            crossing_lane = crossing_lanes[0]
            width = _finite_number(crossing_lane.get("width"))
            lane_id = _nonempty_string(crossing_lane.get("id"))
            shape_text = _nonempty_string(crossing_lane.get("shape"))
            if width is None or width <= 0.0 or lane_id is None or shape_text is None:
                raise RoadSignalContextError(
                    f"{net_xml}: crossing {edge_id!r} lacks explicit width/shape"
                )
            points = []
            for token in shape_text.split():
                parts = token.split(",")
                coordinates = tuple(_finite_number(part) for part in parts)
                if len(coordinates) not in (2, 3) or any(value is None for value in coordinates):
                    raise RoadSignalContextError(
                        f"{net_xml}: crossing {edge_id!r} has a malformed shape"
                    )
                points.append((float(coordinates[0]), float(coordinates[1])))
            if len(points) < 2 or any(a == b for a, b in zip(points, points[1:])):
                raise RoadSignalContextError(
                    f"{net_xml}: crossing {edge_id!r} has a degenerate centerline"
                )
            crosswalks[edge_id] = CrosswalkAuthority(
                edge_id, lane_id, tuple(points), float(width),
                f"{net_xml}#edge={edge_id}&lane={lane_id}&shape,width",
            )

    connections: dict[str, list[SignalConnection]] = {}
    for item in root.findall("connection"):
        tls_id = _nonempty_string(item.get("tl"))
        from_edge = _nonempty_string(item.get("from"))
        to_edge = _nonempty_string(item.get("to"))
        from_lane_index = _nonempty_string(item.get("fromLane"))
        to_lane_index = _nonempty_string(item.get("toLane"))
        link_index_text = _nonempty_string(item.get("linkIndex"))
        if tls_id is None:
            continue
        if None in (from_edge, to_edge, from_lane_index, to_lane_index, link_index_text):
            raise RoadSignalContextError(
                f"{net_xml}: TLS connection {tls_id!r} is incomplete"
            )
        try:
            link_index = int(str(link_index_text))
        except ValueError as exc:
            raise RoadSignalContextError(
                f"{net_xml}: invalid linkIndex {link_index_text!r}"
            ) from exc
        from_lane_id = f"{from_edge}_{from_lane_index}"
        to_lane_id = f"{to_edge}_{to_lane_index}"
        if from_lane_id not in lanes or to_lane_id not in lanes:
            raise RoadSignalContextError(
                f"{net_xml}: TLS connection references an unknown lane: "
                f"{from_lane_id!r}->{to_lane_id!r}"
            )
        connection = SignalConnection(
            from_lane_id=from_lane_id,
            to_edge_id=str(to_edge),
            to_lane_id=to_lane_id,
            via_lane_id=str(item.get("via") or ""),
            tls_id=tls_id,
            link_index=link_index,
        )
        connections.setdefault(from_lane_id, []).append(connection)

    inherited_speed_by_internal_lane: dict[str, list[float]] = {}
    for items in connections.values():
        for item in items:
            if item.via_lane_id:
                inherited_speed_by_internal_lane.setdefault(
                    item.via_lane_id, []
                ).append(lanes[item.from_lane_id].speed_mps)
    for lane_id, inherited_speeds in inherited_speed_by_internal_lane.items():
        lane = lanes.get(lane_id)
        if lane is not None and lane.speed_mps == 0.0:
            lanes[lane_id] = LaneAuthority(
                lane_id=lane.lane_id,
                edge_id=lane.edge_id,
                speed_mps=min(inherited_speeds),
                length_m=lane.length_m,
            )

    normalized_connections: dict[str, tuple[SignalConnection, ...]] = {}
    for lane_id, items in sorted(connections.items()):
        controller_ids = {item.tls_id for item in items}
        if len(controller_ids) != 1:
            raise RoadSignalContextError(
                f"{net_xml}: lane {lane_id!r} maps to multiple TLS controllers: "
                f"{sorted(controller_ids)}"
            )
        normalized_connections[lane_id] = tuple(
            sorted(items, key=lambda item: (item.link_index, item.to_lane_id))
        )

    return RoadSignalContext(
        net_xml=net_xml,
        lanes=lanes,
        signal_connections_by_lane=normalized_connections,
        crossing_count=len(crosswalks),
        traffic_light_controller_count=len(root.findall("tlLogic")),
        signalized_junction_count=sum(
            junction.get("type") == "traffic_light"
            for junction in root.findall("junction")
        ),
        crosswalks=crosswalks,
    )


def _transition_connection(
    connections: Sequence[SignalConnection],
    current_lane: LaneAuthority,
) -> SignalConnection | None:
    matches = [
        item
        for item in connections
        if item.to_lane_id == current_lane.lane_id
        or item.to_edge_id == current_lane.edge_id
        or item.via_lane_id == current_lane.lane_id
    ]
    if not matches:
        return None
    exact = [item for item in matches if item.to_lane_id == current_lane.lane_id]
    selected = exact or matches
    if len({item.link_index for item in selected}) != 1:
        raise RoadSignalContextError(
            "lane transition maps to multiple traffic-light link indices: "
            f"{connections[0].from_lane_id!r}->{current_lane.lane_id!r}"
        )
    return selected[0]


def _current_signal_context(
    connections: Sequence[SignalConnection],
    traffic_light_states: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str]:
    if not connections:
        return NO_CONTROLLING_SIGNAL, NO_CONTROLLING_SIGNAL_STATE
    tls_id = connections[0].tls_id
    normalized_states = {
        _signal_state_for_connection(connection, traffic_light_states)
        for connection in connections
    }
    return tls_id, normalized_states.pop() if len(normalized_states) == 1 else "mixed"


def _signal_state_for_connection(
    connection: SignalConnection,
    traffic_light_states: Mapping[str, Mapping[str, Any]],
) -> str:
    controller = traffic_light_states.get(connection.tls_id)
    if not isinstance(controller, Mapping):
        raise RoadSignalContextError(
            f"traffic-light state lacks controller {connection.tls_id!r}"
        )
    state = controller.get("state")
    if not isinstance(state, str) or connection.link_index >= len(state):
        raise RoadSignalContextError(
            f"controller {connection.tls_id!r} state cannot resolve link "
            f"{connection.link_index}: {state!r}"
        )
    controlled_links = controller.get("controlled_links")
    if not isinstance(controlled_links, list) or connection.link_index >= len(
        controlled_links
    ):
        raise RoadSignalContextError(
            f"controller {connection.tls_id!r} lacks controlled_links for link "
            f"{connection.link_index}"
        )
    link_group = controlled_links[connection.link_index]
    if not isinstance(link_group, list) or not any(
        isinstance(link, list)
        and len(link) >= 3
        and link[0] == connection.from_lane_id
        and link[1] == connection.to_lane_id
        and link[2] == connection.via_lane_id
        for link in link_group
    ):
        raise RoadSignalContextError(
            f"controller {connection.tls_id!r} controlled_links disagrees with "
            f"net.xml link {connection.link_index}"
        )
    return _normalize_signal_character(state[connection.link_index])


def _normalize_signal_character(value: str) -> str:
    if value in {"r", "R"}:
        return "red"
    if value in {"y", "Y"}:
        return "yellow"
    if value in {"g", "G"}:
        return "green"
    if value == "u":
        return "red_yellow"
    if value in {"o", "O"}:
        return "off"
    raise RoadSignalContextError(f"unsupported SUMO traffic-light state {value!r}")


def _nonempty_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


__all__ = [
    "DEFAULT_ROAD_DERIVED_NET_XML",
    "NO_CONTROLLING_SIGNAL",
    "NO_CONTROLLING_SIGNAL_STATE",
    "RoadSignalContext",
    "RoadSignalContextError",
]
