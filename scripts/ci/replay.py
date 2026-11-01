#!/usr/bin/env python3
"""Shadow routing on historical and focused changes; does not execute backend tests."""
import argparse
import json
from pathlib import Path
import time

import docs
import plan


def replay(root):
    roots, reverse = plan.workspace(root)
    base = '6114dab7e18c4deac0fda3039060e00f7661b653'
    head = 'f9781d05b3d5087b3265971435849ec8eeafdb12'
    raw = plan.git(root, 'diff', '--name-only', '-z', '--no-renames', base, head, '--')
    report_paths = [p.decode() for p in raw.split(b'\0') if p]
    cases = {'report-596-complete-diff': report_paths,
             'metal-shader': ['crates/aleph-metal/src/shaders/sv.metal'],
             'cuda-kernel': ['crates/aleph-cuda/src/sv/backend.rs'],
             'cpu-kernel': ['crates/aleph-sv/src/kernels/aos.rs'],
             'qec-model': ['crates/aleph-qec/src/dem.rs'],
             'shared-ir': ['crates/aleph-ir/src/instruction.rs'],
             'python-frontend': ['crates/aleph-py/src/run.rs'],
             'hardware': ['hw/bp_relay_decoder.sv'],
             'policy': ['scripts/ci/plan.py'],
             'mixed-docs-code': ['README.md', 'crates/aleph-sv/src/lib.rs']}
    result = []
    for name, paths in cases.items():
        p = plan.classify(paths, roots, reverse)
        p.update(head_sha=head, base_sha=base, tested_sha=plan.git(root,'rev-parse','HEAD').decode().strip())
        plan.validate(p)
        if name == 'report-596-complete-diff':
            if p['mode'] != 'docs':
                raise ValueError('real documentation PR was not classified docs-only')
            start=time.perf_counter()
            docs.check(root,p)
            p['docs_check_seconds']=time.perf_counter()-start
        p['case']=name
        p['evidence_scope']='historical file-list/current graph replay; no physical GPU run or hosted queue measurement'
        result.append(p)
    return result


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    args.output.write_text(json.dumps(replay(Path.cwd()),indent=2)+'\n')
