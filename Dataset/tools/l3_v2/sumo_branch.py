"""L3 road controls executed by the existing SUMO runner, with measured readback."""
from __future__ import annotations
import copy
import os
from pathlib import Path
from .common import ROOT, CAPTURE, load, rows, write, write_rows, digest, file_digest


def execute_sumo_branch(spec, out: Path) -> Path:
    from Dataset.tools.sumo_ground_flow.run_traffic import run_traffic
    from Dataset.tools.sumo_ground_flow.planner import SumoGroundFlowPlanner
    from Dataset.tools.sumo_ground_flow.incident_plan import DEFAULT_SUMO_NET_XML
    source = ROOT / 'aw_data/sumo_outputs/episode_vehicle_authority' / spec.episode_id
    original = load(source / 'sumo_incident_plan.json')
    plan = copy.deepcopy(original)
    for incident in plan['incidents']:
        incident['start_s'] = incident['episode_local_start_s']
        incident['end_s'] = incident['episode_local_end_s']
        if incident['incident_id'] == spec.sumo_intervention['incident_id']:
            release = spec.sumo_intervention['release_tick']
            if release is not None:
                if not incident['start_s'] < release / 10 < incident['end_s']:
                    raise ValueError('early opening must shorten the actual closure interval')
                incident['end_s'] = release / 10
                incident['episode_local_end_s'] = release / 10
    key = digest({'episode': spec.episode_id, 'plan': plan,
                  'runner': file_digest(ROOT / 'Dataset/tools/sumo_ground_flow/run_traffic.py'),
                  'execution': file_digest(ROOT / 'Dataset/tools/sumo_ground_flow/incident_execution.py'),
                  'vehicle_plan': file_digest(ROOT / 'Dataset/episodes' / spec.episode_id / 'sumo_explicit_vehicle_plan.json')}).split(':')[1]
    work = ROOT / 'design/L3_repair/sumo' / key
    frames = work / spec.episode_id / 'sumo_traffic_frames.jsonl'
    if not (work / 'complete.json').is_file():
        work.mkdir(parents=True, exist_ok=True)
        write(work / 'incident_plan.json', plan)
        os.environ['SUMO_HOME'] = '/mnt/data2/weizhiwei/uav_compute/sumo-1.8.0'
        os.environ['PATH'] = os.environ['SUMO_HOME'] + '/bin:' + os.environ['PATH']
        # Bind the frozen demand records explicitly to the current measured UE
        # lane geometry. The frozen plans predate this executable field; they are
        # never overwritten and no missing-provenance fallback is used by SUMO.
        source_plan = ROOT / 'Dataset/episodes' / spec.episode_id / 'sumo_explicit_vehicle_plan.json'
        demand = load(source_plan)
        planner = SumoGroundFlowPlanner(DEFAULT_SUMO_NET_XML)
        demand['traffic_geometry'] = planner.traffic_geometry_provenance
        demand['l3_reconstruction'] = {'source_plan': str(source_plan.relative_to(ROOT)),
            'source_digest': file_digest(source_plan),
            'geometry_binding': 'current_native_planner; vehicle demand and controls unchanged'}
        input_episode = work / 'inputs' / spec.episode_id
        write(input_episode / 'sumo_explicit_vehicle_plan.json', demand)
        write(input_episode / 'global_entity_roster.json',
              load(ROOT / 'Dataset/episodes' / spec.episode_id / 'global_entity_roster.json'))
        run_traffic(output_dir=work, incident_plan_path=work / 'incident_plan.json',
                    semantic_plan_root=work / 'inputs', duration_s=90., warmup_s=90.,
                    step_length_s=.1, sample_period_s=.1, seed=7, max_vehicles=200,
                    semantic_episode_scope=[spec.episode_id])
        evidence = load(frames.parent / 'lane_incident_execution.json')
        if not evidence['events'] or any(e['status'] not in {'applied_and_readback_verified', 'restored_and_readback_verified'} for e in evidence['events']):
            raise RuntimeError('SUMO road branch has no verified lane control evidence')
        write(work / 'complete.json', {'frames_digest': file_digest(frames), 'lane_controls': evidence})
    elif load(work / 'complete.json')['frames_digest'] != file_digest(frames):
        raise ValueError('cached SUMO source changed')
    write(out / 'raw/sumo_source.json', {'frames': str(frames.relative_to(ROOT)),
          'digest': file_digest(frames), 'mechanism': spec.sumo_intervention,
          'lane_execution': str((frames.parent / 'lane_incident_execution.json').relative_to(ROOT))})
    return frames


class FrameOverlay:
    """Project measured SUMO vehicles using the repository's native UE geometry."""
    def __init__(self, episode_id: str, frames: Path, scene_ids: set[str]):
        from Dataset.tools.sumo_ground_flow.truth_integration import SumoTrafficDataset
        from Dataset.tools.render_ready_vehicle_lanes import VehicleLaneProjector
        from Dataset.tools.sumo_ground_flow.road_signal_context import RoadSignalContext
        self.dataset = SumoTrafficDataset(frames.parent)
        manifest = load(CAPTURE / episode_id / 'episode_manifest.json')
        self.segment = self.dataset.segment_for_episode(episode_id, manifest)
        self.profile = episode_id.split('__')[0]
        self.roster = {e['sumo_vehicle_id']: e for e in load(CAPTURE / episode_id / 'global_entity_roster.json')['entities']
                       if 'sumo_vehicle_id' in e and e['entity_id'] not in scene_ids}
        self.projector = VehicleLaneProjector(ROOT, 'donghu_road_topo')
        self.context = RoadSignalContext.load()
        from Dataset.tools.convert_to_render_ready import source_boundary_polygon
        script = load(ROOT / 'Dataset/scenarios/L3_dynamic_constraints/interaction' / self.profile / 'event_script.json')
        self.capture_contract = {'capture_boundary_polygon_enu_m': source_boundary_polygon(script)}
        self.previous = {}
        self.previous_lights = None
        self.previous_tick = None

    def apply(self, frame: dict) -> dict:
        from Dataset.tools.convert_to_render_ready import (sumo_vehicle_truth_entity,
            projected_sumo_vehicle_position, point_in_runtime_boundary_for_contract, runtime_visibility_payload)
        tick = int(frame['tick'])
        sample = self.dataset.sample(segment=self.segment, episode_sim_time_s=tick / 10,
                       selected_vehicle_ids=self.roster, scenario_id=self.profile)
        owned = {e['entity_id'] for e in self.roster.values()}
        frame['entities'] = [e for e in frame['entities'] if e['entity_id'] not in owned]
        for vehicle in sample['vehicles']:
            previous = self.previous.get(vehicle['vehicle_id']) if self.previous_tick == tick-1 else None
            vehicle.update(self.context.enrich_sumo_vehicle(vehicle, sample['traffic_lights'],
                 previous_sumo_vehicle=previous, previous_traffic_light_states=self.previous_lights if previous else None))
            entry = self.roster[vehicle['vehicle_id']]
            position = projected_sumo_vehicle_position(vehicle, self.projector)
            inside = point_in_runtime_boundary_for_contract(position, self.capture_contract)
            if not inside and entry['sumo_vehicle'].get('source_presence_required') is not True:
                continue
            entity = sumo_vehicle_truth_entity(vehicle=vehicle, roster_entry=entry,
                vehicle_lane_projector=self.projector, tick=tick, site_id=entry['site_id'],
                roi_id=entry['roi_id'], capture_boundary_id='l3_road_control', observation_distance_m=0.)
            entity['runtime_visibility'] = runtime_visibility_payload(position, self.capture_contract)
            frame['entities'].append(entity)
            self.previous[vehicle['vehicle_id']] = copy.deepcopy(vehicle)
        self.previous_tick, self.previous_lights = tick, sample['traffic_lights']
        frame['sumo_active_incidents'] = sample['active_incidents']
        frame['sumo_traffic_light_states'] = sample['traffic_lights']
        frame['sumo_segment'] = self.segment.as_dict()
        rendered = [e for e in frame['entities'] if e['entity_id'] in owned]
        outside = sum(not e['runtime_visibility']['runtime_visible'] for e in rendered)
        frame['sumo_semantics'] = {
            'enabled': True, 'segment': self.segment.as_dict(), 'scenario_id': self.profile,
            'absolute_time_s': sample['absolute_time_s'], 'source_prev_time_s': sample['source_prev_time_s'],
            'source_next_time_s': sample['source_next_time_s'], 'source_alpha': sample['source_alpha'],
            'source_vehicle_count': sample['source_vehicle_count'], 'selected_vehicle_count': len(self.roster),
            'active_selected_vehicle_count': len(rendered), 'traffic_light_count': len(sample['traffic_lights']),
            'traffic_light_states': copy.deepcopy(sample['traffic_lights']),
            'active_incidents': copy.deepcopy(sample['active_incidents']),
            'active_incident_count': len(sample['active_incidents']),
            'runtime_boundary_visible_selected_vehicle_count': len(rendered)-outside,
            'semantic_lifecycle_outside_runtime_boundary_count': outside,
            'retained_outside_boundary_sumo_vehicle_count': outside, 'max_observation_distance_m': 60.0,
            'source_frames': str(self.dataset.frames_path.relative_to(ROOT))}
        assert frame['sumo_semantics']['active_incidents'] == frame['sumo_active_incidents']
        assert frame['sumo_semantics']['traffic_light_states'] == frame['sumo_traffic_light_states']
        return frame
