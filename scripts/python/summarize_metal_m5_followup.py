"""Summarize #520 receipts, preserving supply changes and pre/post drift."""
import argparse
import datetime
import json
import pathlib
import re


def instant(value):
    return datetime.datetime.fromisoformat(value)


def supply(text):
    if "'AC Power'" in text:
        return 'ac'
    if "'Battery Power'" in text:
        return 'battery'
    return 'unknown'


def summarize(root):
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('telemetry_errors'):
        raise ValueError('telemetry failed')
    telemetry=[json.loads(line) for line in (root/'host-power.jsonl').read_text().splitlines()]
    labels=['sv_before','readout','batching','cpu_before','gpu_missing','cpu_after','sv_after']
    if [x['label'] for x in manifest['commands']]!=labels or any(x.get('returncode')!=0 for x in manifest['commands']):
        raise ValueError('incomplete campaign')
    steps={}
    expected=dict(sv_before=2,readout=6,batching=4,cpu_before=1,gpu_missing=1,cpu_after=1,sv_after=2)
    for command in manifest['commands']:
        label=command['label']
        rows=[x for x in telemetry if instant(command['started_utc'])<=instant(x['utc'])<=instant(command['finished_utc'])]
        modes={supply(x['power']) for x in rows}|{supply(command['power_before']),supply(command['power_after'])}
        state=next(iter(modes)) if len(modes)==1 and 'unknown' not in modes else 'mixed_or_unknown'
        cells={}
        for p in (root/'criterion'/label).glob('**/new/estimates.json'):
            sample=json.loads(p.with_name('sample.json').read_text())
            if len(sample['iters'])!=10 or len(sample['times'])!=10:
                raise ValueError('missing 10 actual samples: '+str(p))
            key='/'.join(p.relative_to(root/'criterion'/label).parts[:-2])
            cells[key]=dict(estimates=json.loads(p.read_text()),sample=sample)
        if len(cells)!=expected[label]:
            raise ValueError(f'{label}: {len(cells)} measured cells, expected {expected[label]}')
        charges=[int(m[1]) for row in rows if (m:=re.search(r'(\d+)%;',row['power']))]
        steps[label]=dict(power_profile=state,battery_percent_range=[min(charges),max(charges)] if charges else None,
                          elapsed_seconds=(instant(command['finished_utc'])-instant(command['started_utc'])).total_seconds(),
                          cells=cells)
    def estimate(label,key,stat='median'):
        return steps[label]['cells'][key]['estimates'][stat]['point_estimate']/1e9
    gpu_key='mps_vs_cpu_large_n/gpu/n20_chi512'
    cpu_key='mps_vs_cpu_large_n/cpu/n20_chi512'
    gpu=estimate('gpu_missing',gpu_key)
    ratios={label:gpu/estimate(label,cpu_key) for label in ['cpu_before','cpu_after']}
    profiles=[steps[x]['power_profile'] for x in ['cpu_before','gpu_missing','cpu_after']]
    comparable=len(set(profiles))==1 and profiles[0] in ['ac','battery']
    drift={}
    for mode in ['unfused','fused']:
        key=f'sv_fused_random/{mode}/18'
        change=estimate('sv_after',key)/estimate('sv_before',key)-1.0
        drift[mode]=dict(relative_median_change=change,material_over_10_percent=abs(change)>0.10,
                         same_power_profile=steps['sv_after']['power_profile']==steps['sv_before']['power_profile'] and steps['sv_before']['power_profile'] in ['ac','battery'])
    cpu_change=estimate('cpu_after',cpu_key)/estimate('cpu_before',cpu_key)-1.0
    return dict(steps=steps,mps_n20_chi512=dict(gpu_median_seconds=gpu,
                cpu_before_median_seconds=estimate('cpu_before',cpu_key),cpu_after_median_seconds=estimate('cpu_after',cpu_key),
                gpu_over_cpu_ratios=ratios,ratio_estimator='ratio of median point estimates',
                gpu_mean_seconds=estimate('gpu_missing',gpu_key,'mean'),
                cpu_before_mean_seconds=estimate('cpu_before',cpu_key,'mean'),
                cpu_after_mean_seconds=estimate('cpu_after',cpu_key,'mean'),
                mean_gpu_over_cpu_ratios={label:estimate('gpu_missing',gpu_key,'mean')/estimate(label,cpu_key,'mean') for label in ['cpu_before','cpu_after']},
                same_stable_supply=comparable,power_profiles=profiles,
                cpu_relative_median_change=cpu_change,cpu_material_drift=abs(cpu_change)>0.10),
                sv_observed_drift=drift,
                limitation='Drift is observational, not a thermal/chip-generation attribution. Stable supply does not prove thermal, display or workload isolation.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign',type=pathlib.Path)
    parser.add_argument('output',type=pathlib.Path)
    args=parser.parse_args()
    result=summarize(args.campaign)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['mps_n20_chi512'],indent=2))
