#!/usr/bin/env python3
"""Execute audited CI argv from a validated plan; never evaluate shell fragments."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from plan import validate


def commands(plan, role, toolchain='stable'):
    selected = ['--workspace'] if plan['mode'] == 'full' else [x for p in plan['packages'] for x in ('-p', p)]
    if not selected and role != 'python':
        raise ValueError('no packages selected for ' + role)
    cargo = ['cargo', '+' + toolchain]
    release = ['--release'] if role == 'macos' else []
    result = []
    if role == 'fmt':
        return [cargo + ['fmt', '--check']]
    if role in ('linux', 'macos'):
        result += [cargo + [operation, '--locked'] + selected + release for operation in ('build', 'test')]
        if role == 'linux' and plan['mps']:
            result += [cargo + ['test', '--locked', '-p', 'aleph-mps', '--no-default-features']]
        if role == 'macos' and not plan['jobs']['clippy']:
            result += [cargo + ['clippy', '--locked'] + selected + ['--all-targets', '--', '-D', 'warnings']]
        if role == 'macos' and plan['metal']:
            result += [cargo + ['build', '--locked', '-p', 'aleph-metal', '--features', 'metal', '--release'],
                       cargo + ['test', '--locked', '-p', 'aleph-metal', '--features', 'metal', '--release'],
                       cargo + ['clippy', '--locked', '-p', 'aleph-metal', '--features', 'metal', '--all-targets', '--', '-D', 'warnings'],
                       cargo + ['check', '--locked', '-p', 'aleph-cli', '--features', 'metal', '--all-targets'],
                       cargo + ['check', '--locked', '-p', 'aleph-py', '--features', 'python,metal', '--all-targets']]
        if role == 'linux' and toolchain == 'stable' and plan['cuda']:
            result += [cargo + ['build', '--locked', '-p', 'aleph-cuda', '--features', feature] for feature in ('cuda', 'nccl')]
        return result
    if role == 'clippy':
        result = [cargo + ['clippy', '--locked'] + selected + ['--all-targets', '--', '-D', 'warnings']]
        if plan['cuda']:
            for feature in ('cuda', 'nccl'):
                result += [cargo + ['clippy', '--locked', '-p', 'aleph-cuda', '--features', feature, '--all-targets', '--', '-D', 'warnings']]
            result += [cargo + ['check', '--locked', '-p', 'aleph-cli', '--features', 'cuda', '--all-targets'],
                       cargo + ['check', '--locked', '-p', 'aleph-py', '--features', 'python,cuda', '--all-targets']]
            for suffix in (['--test', 'public_execution', 'pure_planning_contracts'],
                           ['--lib', 'scratch_budget_has_no_padding_or_oversized_single_tile'],
                           ['--lib', 'cooperative_graph_ownership_is_checked_before_device_access'],
                           ['--lib', 'sv::paged::tests']):
                result += [cargo + ['test', '--locked', '-p', 'aleph-cuda', '--features', 'cuda'] + suffix]
        return result
    if role == 'python':
        return [['maturin', 'build', '--release', '--locked', '-m', 'crates/aleph-py/Cargo.toml', '--features', plan['python_features'], '--out', 'dist']]
    raise ValueError('unknown role')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('role', choices=('fmt', 'clippy', 'linux', 'macos', 'python'))
    ap.add_argument('--toolchain', choices=('stable', 'beta'), default='stable')
    ap.add_argument('--plan', default='ci-plan.json')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    plan = validate(json.loads(Path(args.plan).read_text()))
    if not plan['jobs'].get(args.role, False):
        raise ValueError('attempt to execute an unselected role')
    if os.environ.get('GITHUB_SHA') and plan['tested_sha'] != os.environ['GITHUB_SHA']:
        raise ValueError('plan does not match tested revision')
    for command in commands(plan, args.role, args.toolchain):
        print(json.dumps(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
