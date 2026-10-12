"""Check current ARM configuration structure and filesystem identity."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields, MISSING
import json
from pathlib import Path
import sys

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AUTHORITIES = {
    'L1': 'rules/arm_config.schema.json',
    'L2': 'rules/l2_window.schema.json',
    'L3': 'Dataset.tools.l3_v2.arm_runtime.ArmSpec',
    'L4': 'rules/l4_arm_config.schema.json',
    'L5': 'rules/l5_arm_config.schema.json',
    'L6': 'Dataset.tools.l6_v2.arm_runtime.ArmSpec',
    'X': 'Dataset.tools.x_arm_pipeline',
}


def _spec_values(cls, values):
    declared = {field.name for field in fields(cls)}
    required = {field.name for field in fields(cls)
                if field.default is MISSING and field.default_factory is MISSING}
    if not isinstance(values, dict):
        raise ValueError('specification is not a mapping')
    missing, extra = required - values.keys(), values.keys() - declared
    if missing or extra:
        raise ValueError(f'specification fields: missing={sorted(missing)}, extra={sorted(extra)}')
    return cls(**values)


def validate_directory(interventions_root, profile=None):
    root = Path(interventions_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    expected = json.loads((ROOT / 'design/ue_arm_delivery/expected_coverage.json').read_text())
    family_for = {row['profile']: family for family, rows in expected['families'].items() for row in rows}
    paths = list(root.glob('*/seed*/*/*.yaml'))
    for profile_id, family in family_for.items():
        if family == 'L6':
            paths.extend(root.glob(f'{profile_id}/seed*/*/*.json'))
    paths.sort()
    if profile is not None:
        if profile not in family_for:
            raise ValueError(f'Unknown current profile: {profile}')
        paths = [path for path in paths if path.relative_to(root).parts[0] == profile]
    schemas = {family: Draft202012Validator(json.loads((ROOT / ref).read_text()))
               for family, ref in AUTHORITIES.items() if ref.endswith('.json')}
    violations, groups, usage, details = [], set(), Counter(), {}
    for path in paths:
        profile_id, seed, group_id, name = path.relative_to(root).parts
        group = '/'.join((profile_id, seed, group_id))
        detail = details.setdefault(group, {'configs': 0, 'violations': 0})
        detail['configs'] += 1
        try:
            family = family_for[profile_id]
            suffix = '.json' if family == 'L6' else '.yaml'
            if path.suffix != suffix:
                raise ValueError(f'{family} configuration must use {suffix}')
            doc = yaml.safe_load(path.read_text())
            if not isinstance(doc, dict):
                raise ValueError('configuration is not a mapping')
            if family in schemas:
                errors = sorted(schemas[family].iter_errors(doc), key=lambda error: str(list(error.absolute_path)))
                if errors:
                    raise ValueError('; '.join(error.message for error in errors))
                identity = (doc['group_key'], doc['arm_id'], doc['episode_id'])
                window = doc['window']
            elif family == 'L3':
                from Dataset.tools.l3_v2.arm_runtime import ArmSpec
                spec = _spec_values(ArmSpec, doc)
                identity = (spec.group_key, spec.arm_id, spec.episode_id)
                window = spec.window()
            elif family == 'L6':
                from Dataset.tools.l6_v2.arm_runtime import ArmSpec
                if (set(doc) != {'schema', 'version', 'spec'}
                        or doc['schema'] != 'aero_l6_explicit_arm_v3'
                        or type(doc['version']) is not int or doc['version'] != 1):
                    raise ValueError('L6 requires the current explicit v3 configuration')
                spec = _spec_values(ArmSpec, doc['spec'])
                identity = (spec.group_key, spec.arm_id, spec.episode_id)
                window = spec.window()
            elif family == 'X':
                required = {'profile', 'seed', 'episode_id', 'group', 'arm_id', 'mode', 'window',
                            'sources', 'removed_event_ids', 'intervention_actions', 'definition',
                            'gates', 'scope', 'hypothesis'}
                if set(doc) != required:
                    raise ValueError(f'X fields: missing={sorted(required-doc.keys())}, extra={sorted(doc.keys()-required)}')
                if doc['profile'] != profile_id or doc['seed'] != seed:
                    raise ValueError('X profile/seed differs from path')
                identity = (doc['group'], doc['arm_id'], doc['episode_id'])
                window = doc['window']
            else:
                raise ValueError(f'Unknown family: {family}')
            if identity != (group, path.stem, f'{profile_id}__{seed}'):
                raise ValueError('configuration identity differs from path')
            start, fork, end = (window[key] for key in ('start_tick', 'fork_tick', 'end_tick'))
            if any(type(tick) is not int for tick in (start, fork, end)) or not 0 <= start < fork < end <= 900:
                raise ValueError('invalid closed window or fork')
            if window['phase'] != 'after_state_sample_before_actions':
                raise ValueError('unsupported branch sampling phase')
            groups.add(group)
            usage[family] += 1
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
            violations.append({'path': str(path), 'check': 'format', 'message': str(error)})
            detail['violations'] += 1
    return {'configs_scanned': len(paths), 'groups': len(groups),
            'schemas_used': dict(usage), 'authorities': AUTHORITIES,
            'violations': violations, 'empty': not paths,
            'by_check': {'format': len(violations)} if violations else {}, 'groups_detail': details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interventions-root', type=Path, default=ROOT / 'interventions')
    parser.add_argument('--profile')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--schema', action='store_true', help='print current format authorities')
    args = parser.parse_args()
    if args.schema:
        print(json.dumps(AUTHORITIES, indent=2))
        return 0
    report = validate_directory(args.interventions_root, args.profile)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(bool(report['violations']) or report['empty'])


if __name__ == '__main__':
    raise SystemExit(main())
