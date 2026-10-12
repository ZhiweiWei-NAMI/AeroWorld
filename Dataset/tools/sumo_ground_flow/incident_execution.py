"""Measured lane-control evidence, separate from scheduled incident provenance."""
from __future__ import annotations

import math
from typing import Any


class LaneIncidentExecution:
    """Track actual writes and readback for explicitly anchored lane closures.

    A vehicle having an authored trajectory does not implement a lane control.
    Neither a plan's time window nor an attempted write certifies its effect.
    """

    def __init__(self):
        self.events: list[dict[str, Any]] = []
        self.controls: dict[str, dict[str, Any]] = {}

    def apply(self, traci, incident: dict[str, Any], *, maximum_speed_mps: float) -> float:
        incident_id = incident['incident_id']
        lane_id = incident['anchor']['sumo_lane_id']
        edge_id = incident['anchor']['sumo_edge_id']
        if not isinstance(lane_id, str) or not lane_id or not isinstance(edge_id, str) or not edge_id:
            raise ValueError('lane closure requires explicit lane and edge identities')
        if incident_id in self.controls:
            raise ValueError('duplicate lane-incident application')
        if not math.isfinite(maximum_speed_mps) or maximum_speed_mps < 0:
            raise ValueError('invalid lane maximum speed')
        before = float(traci.lane.getMaxSpeed(lane_id))
        traci.lane.setMaxSpeed(lane_id, maximum_speed_mps)
        after = float(traci.lane.getMaxSpeed(lane_id))
        if not math.isfinite(after) or after != maximum_speed_mps:
            raise RuntimeError(f'lane control readback differs: {after} != {maximum_speed_mps}')
        row = {'incident_id': incident_id, 'episode_event_id': incident['episode_event_id'],
               'accident_class': incident['accident_class'], 'lane_id': lane_id, 'road_segment_id': edge_id,
               'sim_time_s': float(traci.simulation.getTime()), 'operation': 'lane.setMaxSpeed',
               'before_maximum_speed_mps': before, 'after_maximum_speed_mps': after,
               'status': 'applied_and_readback_verified', 'scheduled_start_s': incident['start_s'],
               'scheduled_end_s': incident['end_s']}
        self.events.append(row)
        self.controls[incident_id] = {**row, 'restored': False}
        return before

    def observe(self, traci) -> list[dict[str, Any]]:
        rows = []
        for control in self.controls.values():
            measured = float(traci.lane.getMaxSpeed(control['lane_id']))
            if not math.isfinite(measured):
                raise RuntimeError('nonfinite lane state')
            expected = control['before_maximum_speed_mps'] if control['restored'] else control['after_maximum_speed_mps']
            if measured != expected:
                raise RuntimeError('lane state changed without a recorded control/recovery event')
            rows.append({'incident_id': control['incident_id'], 'lane_id': control['lane_id'],
                         'road_segment_id': control['road_segment_id'],
                         'maximum_speed_mps': measured, 'closure_control_active': not control['restored'],
                         'sim_time_s': float(traci.simulation.getTime())})
        return rows

    def record_restoration(self, traci, lane_id: str, expected_speed_mps: float) -> None:
        actual = float(traci.lane.getMaxSpeed(lane_id))
        if actual != expected_speed_mps:
            raise RuntimeError('lane restoration readback differs from original state')
        matched = [row for row in self.controls.values() if row['lane_id'] == lane_id and not row['restored']]
        if len(matched) != 1:
            raise RuntimeError('lane restoration does not uniquely identify an active control')
        control = matched[0]
        control['restored'] = True
        self.events.append({'incident_id': control['incident_id'], 'episode_event_id': control['episode_event_id'],
                            'lane_id': lane_id, 'road_segment_id': control['road_segment_id'],
                            'sim_time_s': float(traci.simulation.getTime()),
                            'operation': 'lane.setMaxSpeed', 'after_maximum_speed_mps': actual,
                            'status': 'restored_and_readback_verified'})
