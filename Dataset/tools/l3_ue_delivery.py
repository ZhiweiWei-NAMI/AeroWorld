"""Run one bound L3 configuration and export its measured UE capture inputs."""
from __future__ import annotations

from pathlib import Path
import yaml

from Dataset.tools.arm_ue_export import export_arm
from Dataset.tools.convert_to_render_ready import indexed_realizations, _event_id_from_trace
from Dataset.tools.validate_arm_ue import validate_arm
from Dataset.tools.l3_v2.arm_runtime import ArmSpec, run_arm
from Dataset.tools.l3_v2.common import ROOT, load, rows, write, write_rows


def entry_source(config_ref: str) -> str:
    return ('"""Fixed L3 simulation and UE export entry."""\n'
            'from pathlib import Path\nimport sys\n'
            'ROOT = Path(__file__).resolve().parents[5]\nsys.path.insert(0, str(ROOT))\n'
            'from Dataset.tools.l3_ue_delivery import load_spec, deliver_config\n'
            f'CONFIG = {config_ref!r}\nSPEC = load_spec(ROOT / CONFIG)\n'
            'if __name__ == "__main__":\n    print(deliver_config(ROOT / CONFIG), flush=True)\n')


def load_spec(config_path: Path) -> ArmSpec:
    config_path = Path(config_path)
    spec = ArmSpec(**yaml.safe_load(config_path.read_text()))
    expected = ROOT / 'interventions' / spec.group_key / (spec.arm_id + '.yaml')
    if config_path.resolve() != expected.resolve():
        raise ValueError('L3 configuration identity differs from its bound path')
    return spec


def export_existing(spec: ArmSpec, config_path: Path) -> dict:
    arm = ROOT / spec.output
    manifest = load(arm / 'manifest.json')
    source = ROOT / manifest['materialized_source']
    export_arm(
        arm, ROOT / 'aw_data/render_ready_episodes_capture_filtered' / spec.episode_id,
        ROOT / manifest['sources']['scene']['path'], ROOT / manifest['sources']['script']['path'],
        frame_source=source / 'render_background_frames.jsonl' if spec.sumo_intervention is not None else None,
    )
    # A dispatch frame is a valid observation, but cannot stand in for an
    # unobserved action result. Keep scheduled labels and their explicit status.
    realized = indexed_realizations(rows(arm / 'ue/event_realization.jsonl'), spec.profile)
    labels = rows(arm / 'ue/dynamic_labels.jsonl')
    for label in labels:
        event = realized[_event_id_from_trace(label, spec.profile)]
        label['result_tick'] = event['result_tick'] if event['result_observed'] else None
        label['evidence_tick'] = event['evidence_tick'] if event['result_observed'] else None
    write_rows(arm / 'ue/dynamic_labels.jsonl', labels)
    validation = validate_arm(arm)
    write(arm / 'ue/validation.json', validation)
    write(arm / 'ue_validation.json', validation)
    manifest['configuration'] = {'path': str(config_path.relative_to(ROOT))}
    entrypoint = ROOT / 'scripts/arms' / spec.group_key / (spec.arm_id + '.py')
    manifest['entrypoint'] = {'path': str(entrypoint.relative_to(ROOT))}
    manifest['ue_delivery'] = {'scenario_package': 'ue/scenario_package.json',
                               'validation': 'ue/validation.json', 'capture_executed': False}
    manifest['artifacts'] = sorted(str(p.relative_to(arm)) for p in arm.rglob('*')
                                   if p.is_file() and p != arm / 'manifest.json')
    write(arm / 'manifest.json', manifest)
    return validation


def deliver_config(config_path: Path) -> dict:
    spec = load_spec(config_path)
    result = run_arm(spec)
    result['ue'] = export_existing(spec, config_path)
    return result
