#!/usr/bin/env python3
"""Fail closed when planning or any selected CI group is not actually successful."""
import json
import os
from pathlib import Path

from plan import JOBS, validate


def check(plan, needs, head, tested):
    validate(plan)
    if plan['head_sha'] != head or plan['tested_sha'] != tested:
        raise ValueError('stale plan identity')
    if needs.get('plan', {}).get('result') != 'success':
        raise ValueError('planning did not succeed')
    for job in JOBS:
        expected = 'success' if plan['jobs'][job] else 'skipped'
        if needs.get(job, {}).get('result') != expected:
            raise ValueError(job + ': expected ' + expected + ', got ' + str(needs.get(job)))
    return True


if __name__ == '__main__':
    check(json.loads(Path('ci-plan.json').read_text()), json.loads(os.environ['CI_NEEDS']),
          os.environ['CI_HEAD_SHA'], os.environ['GITHUB_SHA'])
    print('Every selected check succeeded; unselected suites were not executed.')
