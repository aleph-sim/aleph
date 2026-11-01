#!/usr/bin/env python3
"""Conservative affected-check plan. No network API diff limits or shell evaluation."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess

VERSION = 1
JOBS = ('docs', 'fmt', 'clippy', 'linux', 'macos', 'python', 'hardware')
SHA = re.compile(r'[0-9a-f]{40}')
PACKAGE = re.compile(r'aleph-[a-z0-9-]+')
ROOT_DOCS = {'README.md', 'BACKLOG.md', 'ROADMAP.md', 'CREATE ISSUES.md',
             'OPTIMIZATION GUIDE.md', 'OPTIMIZATION CYCLE.md', 'LICENSE'}
DOC_EXT = {'.md', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.pdf'}
REPORT_EXT = {'.json', '.jsonl', '.csv', '.txt', '.log', '.gz', '.zip'}


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.PIPE)


def is_doc(path):
    p = PurePosixPath(path)
    if p.is_absolute() or '..' in p.parts:
        return False
    if path in ROOT_DOCS:
        return True
    if p.parts[0] != 'docs' or any(x in p.parts for x in ('fixtures', 'testdata', 'generated')):
        return False
    return p.suffix.lower() in DOC_EXT or (
        path.startswith('docs/perf/data/') and
        (p.suffix.lower() in REPORT_EXT or p.name == 'SHA256SUMS'))


def workspace(root):
    """Metadata includes inactive optional/target and dev/build edges: a safe superset."""
    data = json.loads(subprocess.check_output(
        ['cargo', 'metadata', '--locked', '--no-deps', '--format-version', '1'], cwd=root))
    members = set(data['workspace_members'])
    packages = {p['name']: p for p in data['packages'] if p['id'] in members}
    if not packages or not all(PACKAGE.fullmatch(n) for n in packages):
        raise ValueError('unknown workspace package names')
    roots = {str(Path(p['manifest_path']).parent.resolve().relative_to(root.resolve())): n
             for n, p in packages.items()}
    reverse = {n: set() for n in packages}
    for consumer, p in packages.items():
        for d in p['dependencies']:
            if d['name'] in packages and d.get('path'):
                reverse[d['name']].add(consumer)
    return roots, reverse


def closure(seeds, reverse):
    selected = set(seeds)
    pending = list(seeds)
    while pending:
        for consumer in reverse[pending.pop()] - selected:
            selected.add(consumer)
            pending.append(consumer)
    return sorted(selected)


def classify(paths, roots, reverse, force=False):
    reasons = []
    all_packages = sorted(reverse)
    mode = 'affected'
    seeds = set()
    hardware = False
    script_python = False
    if not paths and not force:
        force = True
        reasons.append('Empty diff is ambiguous: full validation')
    for path in paths:
        if is_doc(path):
            continue
        if path.startswith('hw/'):
            hardware = True
            continue
        # The Python and public-workflow suites import these scripts/fixtures.
        if path.startswith(('scripts/python/', 'scripts/benchmarks/')) or path == 'scripts/qec_threshold_plot.py':
            script_python = True
            seeds.update(('aleph-py', 'aleph-cli', 'aleph-cuda', 'aleph-metal'))
            continue
        matched = next((roots[prefix] for prefix in sorted(roots, key=len, reverse=True) if path.startswith(prefix + '/')), None)
        config = PurePosixPath(path).name in ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'pyproject.toml', 'setup.cfg', 'setup.py', 'build.rs') or '.cargo' in PurePosixPath(path).parts
        if matched and not config:
            seeds.add(matched)
            continue
        force = True
        reasons.append('Full fallback for policy/configuration/unmapped input: ' + path)
    if paths and all(is_doc(p) for p in paths) and not force:
        mode, packages = 'docs', []
    elif force:
        mode, packages = 'full', all_packages
        hardware = True
    else:
        packages = closure(seeds, reverse)
        reasons.append('Reverse normal/dev/build/optional/target consumer closure: ' + ', '.join(packages))
    metal_only = mode == 'affected' and seeds == {'aleph-metal'} and not hardware and not script_python
    cuda_only = mode == 'affected' and seeds == {'aleph-cuda'} and not hardware and not script_python
    frontends = bool({'aleph-cli', 'aleph-py'} & set(packages))
    # A frontend edit may change cfg(cuda/metal) code without changing a GPU crate.
    cuda = 'aleph-cuda' in packages or (frontends and not metal_only)
    metal = 'aleph-metal' in packages or (frontends and not cuda_only)
    # Any affected QEC consumer path may touch its Rust hardware golden model.
    hardware = hardware or 'aleph-qec' in packages
    jobs = {'docs': True, 'fmt': bool(packages), 'clippy': bool(packages),
            'linux': bool(packages) and not metal_only,
            'macos': bool(packages), 'python': 'aleph-py' in packages or script_python,
            'hardware': hardware}
    if metal_only:
        jobs['clippy'] = False  # Metal and consumer Clippy run on macOS instead.
    return {'schema_version': VERSION, 'mode': mode, 'paths': sorted(paths),
            'packages': packages, 'seeds': sorted(seeds), 'jobs': jobs,
            'cuda': cuda, 'metal': metal, 'mps': 'aleph-mps' in packages,
            'python_os': 'macos-latest' if metal_only else 'ubuntu-latest',
            'python_features': 'python,metal' if metal_only else ('python,cuda' if cuda else 'python'),
            'reasons': reasons or (['Approved prose/report files only; no code suites selected'] if mode == 'docs' else ['Full validation requested by event, override or non-regular diff'])}


def changed_paths(root, event, name):
    tested = git(root, 'rev-parse', 'HEAD').decode().strip()
    if name == 'pull_request':
        head = event['pull_request']['head']['sha']
        base = event['pull_request']['base']['sha']
        if not all(SHA.fullmatch(s) for s in (head, base)):
            raise ValueError('invalid event revisions')
        start = git(root, 'merge-base', base, head).decode().strip()
    elif name == 'push':
        head, base = event['after'], event['before']
        if not all(SHA.fullmatch(s) and s != '0' * 40 for s in (head, base)):
            raise ValueError('unavailable push range')
        start = base
    else:
        return [], tested, tested, tested, True
    raw = git(root, 'diff', '--name-only', '-z', '--no-renames', start, head, '--')
    typed = git(root, 'diff', '--raw', '-z', '--no-renames', start, head, '--').split(b'\0')
    non_regular = any(mode.lstrip(b':') not in (b'000000', b'100644')
                      for header in typed[0:-1:2] for mode in header.split()[:2])
    return sorted(set(p.decode('utf-8') for p in raw.split(b'\0') if p)), head, base, tested, non_regular


def validate(plan):
    if plan.get('schema_version') != VERSION or plan.get('mode') not in ('docs', 'affected', 'full'):
        raise ValueError('invalid plan version/mode')
    if set(plan.get('jobs', {})) != set(JOBS) or any(type(v) is not bool for v in plan['jobs'].values()):
        raise ValueError('invalid job selection')
    if not plan['jobs']['docs'] or not all(SHA.fullmatch(plan.get(k, '')) for k in ('head_sha', 'base_sha', 'tested_sha')):
        raise ValueError('missing identities/docs gate')
    if not isinstance(plan.get('packages'), list) or any(not PACKAGE.fullmatch(p) for p in plan['packages']):
        raise ValueError('invalid package selection')
    if plan['mode'] == 'docs' and (plan['packages'] or any(v for k, v in plan['jobs'].items() if k != 'docs') or not plan['paths'] or not all(is_doc(p) for p in plan['paths'])):
        raise ValueError('invalid docs plan')
    for k in ('cuda', 'metal', 'mps'):
        if type(plan.get(k)) is not bool:
            raise ValueError('invalid feature flag')
    if plan.get('python_os') not in ('ubuntu-latest', 'macos-latest') or plan.get('python_features') not in ('python', 'python,cuda', 'python,metal'):
        raise ValueError('invalid Python configuration')
    paths = plan.get('paths')
    if not isinstance(paths, list) or any(not isinstance(p, str) or not p or '\x00' in p or PurePosixPath(p).is_absolute() or '..' in PurePosixPath(p).parts for p in paths):
        raise ValueError('invalid changed paths')
    if len(set(plan['packages'])) != len(plan['packages']):
        raise ValueError('duplicate packages')
    if plan['mode'] == 'full' and not all(plan['jobs'].values()):
        raise ValueError('full plan omitted a CI group')
    if plan['jobs']['python'] != ('aleph-py' in plan['packages']):
        raise ValueError('Python consumer coverage mismatch')
    if plan['jobs']['fmt'] != bool(plan['packages']) or plan['jobs']['macos'] != bool(plan['packages']):
        raise ValueError('Rust coverage mismatch')
    if plan['mode'] == 'docs' and any(plan[k] for k in ('cuda', 'metal', 'mps')):
        raise ValueError('docs plan requested backend features')
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', default='ci-plan.json')
    args = ap.parse_args()
    root = Path.cwd().resolve()
    event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
    name = os.environ['GITHUB_EVENT_NAME']
    fallback = None
    try:
        paths, head, base, tested, force = changed_paths(root, event, name)
    except (KeyError, ValueError, UnicodeError, subprocess.CalledProcessError) as error:
        paths, force = [], True
        tested = git(root, 'rev-parse', 'HEAD').decode().strip()
        head = event.get('pull_request', {}).get('head', {}).get('sha', tested)
        base = event.get('pull_request', {}).get('base', {}).get('sha', tested)
        fallback = 'Unavailable complete diff; full validation: ' + str(error)
    labels = {x['name'] for x in event.get('pull_request', {}).get('labels', [])}
    force = force or 'ci:full' in labels or name in ('workflow_dispatch', 'schedule', 'merge_group')
    # Main code pushes retain broad integration, while prose pushes remain cheap.
    force = force or (name == 'push' and not all(is_doc(p) for p in paths))
    # Never inspect a symlink as inert prose or code; unknown ownership is full.
    force = force or any((root / p).is_symlink() for p in paths)
    if paths and all(is_doc(p) for p in paths) and not force:
        plan = classify(paths, {}, {})  # no Rust/Cargo initialization for docs
    else:
        roots, reverse = workspace(root)
        plan = classify(paths, roots, reverse, force)
    if fallback:
        plan['reasons'].append(fallback)
    plan.update(head_sha=head, base_sha=base, tested_sha=tested)
    validate(plan)
    Path(args.output).write_text(json.dumps(plan, indent=2) + '\n')
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            for key, value in plan['jobs'].items():
                f.write(f'{key}={str(value).lower()}\n')
            f.write('python_os=' + plan['python_os'] + '\n')
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as f:
            f.write('## CI plan: ' + plan['mode'] + '\n\n')
            f.write('Head `' + head + '`; tested `' + tested + '`\n\n')
            for key, value in plan['jobs'].items():
                f.write('- ' + key + ': ' + ('selected' if value else 'not selected (not a test pass)') + '\n')
            for reason in plan['reasons']:
                f.write('\n' + reason.replace('\n', ' ') + '\n')
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
