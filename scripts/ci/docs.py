#!/usr/bin/env python3
"""Check changed documentation without builds, network links or executing examples."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

from plan import git, validate


def links(text):
    # Inline repository file links only; fenced example source is not a link index.
    text = re.sub(r'```.*?```', '', text, flags=re.S)
    for raw in re.findall(r'!?\[[^\]]*\]\((<[^>]+>|[^)\s]+)(?:\s+"[^"]*")?\)', text):
        target = raw.strip('<>')
        url = urlsplit(target)
        if not url.scheme and not url.netloc and url.path:
            yield unquote(url.path)


def check(root, plan):
    errors = []
    paths = plan['paths']
    deleted = {p for p in paths if not (root / p).exists()}
    for name in paths:
        p = root / name
        if not p.exists() or not (name.endswith('.md') or name.startswith('docs/')):
            continue
        resolved = p.resolve()
        if p.is_symlink() or root.resolve() not in resolved.parents:
            errors.append(name + ': symlink/outside checkout')
            continue
        try:
            suffix = p.suffix.lower()
            if suffix in ('.md', '.txt', '.log', '.csv', '.json', '.jsonl', '.svg') or p.name == 'SHA256SUMS':
                text = p.read_text(encoding='utf-8')
                if '\x00' in text:
                    raise ValueError('NUL byte in text document')
            if suffix == '.json':
                json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON: ' + x)))
            elif suffix == '.jsonl':
                for line in text.splitlines():
                    if line.strip():
                        json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON: ' + x)))
            elif suffix == '.csv':
                list(csv.reader(io.StringIO(text), strict=True))
            elif suffix == '.svg':
                ET.fromstring(text)
            elif suffix == '.md':
                try:
                    before = git(root, 'show', plan['base_sha'] + ':' + name).decode('utf-8')
                except Exception:
                    before = ''  # new doc: all links must resolve
                old = set(links(before))
                for target in set(links(text)) - old:
                    local = (p.parent / target)
                    if root.resolve() not in local.resolve().parents and local.resolve() != root.resolve():
                        errors.append(name + ': local link escapes checkout ' + target)
                    elif not local.exists():
                        errors.append(name + ': new missing local link ' + target)
        except (ValueError, UnicodeError, csv.Error, ET.ParseError) as error:
            errors.append(name + ': ' + str(error))
    # A deleted asset can break a Markdown file that itself did not change.
    if deleted:
        for p in root.rglob('*.md'):
            if any(x in p.parts for x in ('.git', 'target', '.venv', 'node_modules')):
                continue
            for target in links(p.read_text(encoding='utf-8')):
                local = (p.parent / target).resolve()
                if any(local == (root / d).resolve() for d in deleted):
                    errors.append(str(p.relative_to(root)) + ': link to deleted asset ' + target)
    # Check manifests adjacent to edited report data, even if not edited themselves.
    manifests = set()
    for name in paths:
        p = root / name
        if name.startswith('docs/perf/data/'):
            manifests.update(parent / 'SHA256SUMS' for parent in (p.parent, *p.parent.parents)
                             if parent != root and root in parent.parents and (parent / 'SHA256SUMS').is_file())
    for p in manifests:
        for line in p.read_text().splitlines():
            match = re.fullmatch(r'([0-9a-f]{64})\s+\*?(.+)', line)
            if not match:
                errors.append(str(p) + ': invalid SHA256SUMS row')
                continue
            target = p.parent / match[2]
            if root.resolve() not in target.resolve().parents or not target.is_file():
                errors.append(str(p) + ': missing/unsafe checksum target ' + match[2])
            elif hashlib.sha256(target.read_bytes()).hexdigest() != match[1]:
                errors.append(str(p) + ': stale checksum for ' + match[2])
    if errors:
        raise ValueError('\n'.join(errors))
    return len(paths)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', default='ci-plan.json')
    args = ap.parse_args()
    plan = validate(json.loads(Path(args.plan).read_text()))
    print('Documentation checks passed for', check(Path.cwd(), plan), 'changed paths')
