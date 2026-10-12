"""Deliver the retained L5 weather branches."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Dataset.tools.l5_arm_common import load_config, RUNTIME_FAMILIES
from Dataset.tools.l5_arm_engine import execute
from Dataset.tools.l5_arm_pipeline import execute_config

FINAL_BRANCH = 'grp_l5_env_branch_v1'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def configs():
    return sorted((ROOT/'interventions').glob(f'L5-*/seed*/{FINAL_BRANCH}/arm_*.yaml'))


def response_evidence(arm_dir, config, engine, audit):
    start, end = config['window']['start_tick'], config['window']['end_tick']
    primary = config['primary_entity_id']
    states = [r for r in engine.trajectory_rows if start<=r['tick']<=end and r['entity_id']==primary]
    transitions = []
    previous = None
    for row in states:
        state = {'activity_type':row['activity_type'], **{k:row[k] for k in RUNTIME_FAMILIES if k in row}}
        if state != previous:
            transitions.append({'tick':row['tick'], 'state':state,
                                'evidence_ref':f'raw/ue_trajectories.jsonl#tick={row["tick"]}&entity={primary}'})
        previous = state
    actions = [{'tick':a['tick'], 'action':a['action'], 'result':a['result'],
                'evidence_ref':f'raw/actions.jsonl#sequence={a["sequence"]}'}
               for a in audit if start<=a['tick']<=end]
    write_json(arm_dir/'objective/business_response_evidence.json', {
        'schema':'l5_measured_business_response_v1', 'entity_id':primary,
        'window':config['window'], 'state_transitions':transitions, 'dispatched_actions':actions,
        'interpretation':'实际派发与后续状态证据；不新增本体事件、不将脚本标签解释为物理测量。',
        'observed_end_state':states[-1],
        'physical_battery_measurements':'not_generated'})


def deliver_config(config_path, *, export=False):
    from Dataset.tools.arm_ue_export import write_engine_inputs, export_arm
    config_path = Path(config_path).absolute()
    config = load_config(config_path)
    arm_dir = ROOT/config['output']
    if not arm_dir.exists():
        execute_config(config_path)
        return
    engine, interpreter, audit, _, _ = execute(config)
    manifest = json.loads((arm_dir/'manifest.json').read_text())
    # Verify the original subject trajectories before adding all scene actors.
    current = {(r['tick'],r['entity_id']):r for r in engine.trajectory_rows}
    for line in (arm_dir/'raw/trajectories.jsonl').read_text().splitlines():
        old = json.loads(line); new = current[(old['tick'],old['entity_id'])]
        if math.dist(old['pos_enu'],new['pos_enu'])>2e-6:
            raise ValueError(f'{config_path}: existing position differs at {old["tick"]}')
        for key in ('activity_type',*RUNTIME_FAMILIES):
            if old.get(key)!=new.get(key):
                raise ValueError(f'{config_path}: existing {key} differs at {old["tick"]}')
    script_path = ROOT/config['sources']['script']
    write_engine_inputs(arm_dir,engine,interpreter,config['window'],script=json.loads(script_path.read_text()))
    response_evidence(arm_dir,config,engine,audit)
    if export:
        validation = export_arm(arm_dir, ROOT/'aw_data/render_ready_episodes_capture_filtered'/config['episode_id'],
                   ROOT/config['sources']['scene'], script_path)
        manifest['ue_package_ref'] = str((arm_dir/'ue/scenario_package.json').relative_to(ROOT))
        manifest['ue_validation_ref'] = str((arm_dir/'ue/validation.json').relative_to(ROOT))
        manifest['ue_capture_status'] = 'input_package_validated_capture_not_executed'
        manifest['certification']['local_candidate_asset'] = 'generated'
        manifest['certification']['release'] = 'local_ue_inputs_exported_capture_not_executed'
    manifest['provenance']['parameter_sources']['branch_environment'] = config['sources']['scene']
    manifest['provenance']['parameter_sources']['branch_definition'] = config['sources']['script']
    manifest['artifacts'] = sorted(str(p.relative_to(arm_dir)) for p in (arm_dir/'raw').glob('ue_*') if p.is_file())
    manifest['artifacts'].append('objective/business_response_evidence.json')
    write_json(arm_dir/'manifest.json',manifest)
    print(json.dumps({'arm_key':manifest['arm_key'], 'window':config['window'],
        'scene_actor_count':len(engine.entities),'ue_exported':export}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--export',action='store_true');parser.add_argument('--config',type=Path)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    for p in ([args.config] if args.config else configs()):
        if args.resume:
            cfg=load_config(p);out=ROOT/cfg['output']
            if (out/'ue/validation.json').is_file() and (out/'manifest.json').is_file():
                m=json.loads((out/'manifest.json').read_text())
                if m.get('ue_capture_status')=='input_package_validated_capture_not_executed':
                    continue
        deliver_config(p,export=args.export)

if __name__=='__main__':main()
