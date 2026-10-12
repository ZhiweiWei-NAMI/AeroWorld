"""Run every ARM of the L5 environment-branch group through its fixed entry.

Refuses to overwrite an existing ARM directory.  Prints one JSON object per
completed ARM so a caller can capture the whole batch deterministically.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GROUP = 'grp_l5_env_branch_v1'


def groups():
    for group in sorted((ROOT / 'scripts/arms').glob(f'L5-*/*/{GROUP}')):
        yield group.relative_to(ROOT / 'scripts/arms')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', action='append', default=[])
    parser.add_argument('--only-arm')
    args = parser.parse_args()
    wanted = args.group or [str(g) for g in groups()]
    failures = []
    for g in wanted:
        entry_dir = ROOT / 'scripts/arms' / g
        for entry in sorted(entry_dir.glob('arm_*.py')):
            if args.only_arm and entry.stem != args.only_arm:
                continue
            out = ROOT / 'arms' / g / entry.stem
            if out.exists():
                failures.append({'entry': str(entry.relative_to(ROOT)), 'error': 'output exists'})
                print(json.dumps({'entry': str(entry.relative_to(ROOT)), 'status': 'skipped_existing'}),
                      flush=True)
                continue
            proc = subprocess.run([sys.executable, str(entry)], cwd=ROOT,
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                failures.append({'entry': str(entry.relative_to(ROOT)),
                                 'returncode': proc.returncode,
                                 'stderr_tail': proc.stderr.strip().splitlines()[-3:]})
                print(json.dumps(failures[-1], ensure_ascii=False), flush=True)
            else:
                print(json.dumps({'entry': str(entry.relative_to(ROOT)), 'status': 'ok',
                                  'summary': proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ''},
                                 ensure_ascii=False), flush=True)
    print(json.dumps({'failures': failures, 'failure_count': len(failures)}), flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
