"""Paired actual CLI/Python CUDA A/B; bounded states, no service changes.

The public routes construct a backend/plan per call. Criterion separately measures
reusable-plan execution. Install the candidate wheel and build the CUDA CLI first.
"""
import argparse
import gc
import json
import math
import pathlib
import os
import random
import subprocess
import tempfile
import time
import numpy as np
import aleph


def workload(family,n):
    lines=['OPENQASM 3.0; include "stdgates.inc";',f'qubit[{n}] q;']
    def gate(name,*qs,angle=None):
        op=name if angle is None else f'{name}({angle:.17g})'
        if name=='cp':
            # The current QASM parser accepts p/cx, not cp or ctrl modifiers.
            # Exact controlled-phase decomposition, shared by both public routes.
            control,target=qs
            gate('p',control,angle=angle/2);gate('p',target,angle=angle/2)
            gate('cx',control,target);gate('p',target,angle=-angle/2);gate('cx',control,target)
            return
        lines.append(op+' '+','.join(f'q[{q}]' for q in qs)+';')
    def qft(width,inverse=False):
        for i in (reversed(range(width)) if inverse else range(width)):
            if not inverse:gate('h',i)
            for j in range(i+1,width):gate('cp',j,i,angle=(-1 if inverse else 1)*math.pi/2**(j-i))
            if inverse:gate('h',i)
    rng=random.Random(552)
    if family=='qft':qft(n)
    elif family=='qpe':
        gate('x',n-1)
        for q in range(n-1):gate('h',q)
        for q in range(n-1):gate('cp',q,n-1,angle=(.37*2**q)%(2*math.pi))
        qft(n-1,True)
    elif family=='qaoa':
        for q in range(n):gate('h',q)
        for layer in range(4):
            for q in range(n-1):
                gate('cx',q,q+1);gate('rz',q+1,angle=.17+.03*layer);gate('cx',q,q+1)
            for q in range(n):gate('rx',q,angle=.31+.02*layer)
    else:
        for layer in range(6):
            for q in range(n):
                gate('ry',q,angle=rng.random() if family=='brickwall' else .13+.07*(q+layer))
                gate('rz',q,angle=.23+.03*(q+layer))
            edges=range(n-1) if family=='vqe' else range(layer%2,n-1,2)
            for q in edges:gate('cx',q,q+1)
    return '\n'.join(lines)


def gpu():
    return subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,utilization.gpu,memory.free','--format=csv,noheader,nounits'],text=True).strip()


def idle():
    for _ in range(10):
        value=gpu()
        if int(value.split(',')[-2])<=5:return value
        time.sleep(2)
    raise RuntimeError('GPU busy; campaign stopped without changing services')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--qubits',type=int,default=20)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--shots',type=int,default=32)
    parser.add_argument('--families',nargs='+',choices=['qft','qpe','brickwall','vqe','qaoa'],default=['qft','qpe','brickwall','vqe','qaoa'])
    args=parser.parse_args()
    if not 8<=args.qubits<=22 or not 0<=args.shots<=1024 or not 1<=args.repeats<=10:
        parser.error('bounded campaign: n 8..22, shots 0..1024, repeats 1..10')
    families=args.families
    # Verify every timed workload family on a small exact CPU FP64 oracle first.
    for family in families:
        c=aleph.Circuit.from_qasm(workload(family,8))
        expected=aleph.run(c,backend='sv',shots=0).statevector()
        for backend,tol in [('cuda',1e-10),('cuda-f32',1e-5)]:
            for optimize in [False,True]:
                r=aleph.run(c,backend=backend,shots=0,cuda_optimize=optimize)
                np.testing.assert_allclose(r.statevector(),expected,atol=tol,rtol=0)
    with tempfile.TemporaryDirectory(prefix='aleph-552-bench-') as tmp,open(args.output,'w') as out:
        for family in families:
            source=workload(family,args.qubits)
            path=pathlib.Path(tmp)/f'{family}.qasm';path.write_text(source)
            c=aleph.Circuit.from_qasm(source)
            for backend in ['cuda','cuda-f32']:
                # One public-call warmup in each arm; no persistent backend/plan.
                for optimize in [False,True]:
                    aleph.run(c,backend=backend,shots=args.shots,seed=552,cuda_optimize=optimize)
                for entry in ['python','cli']:
                    for pair in range(args.repeats):
                        for optimize in ([False,True] if pair%2==0 else [True,False]):
                            before=idle()
                            start=time.perf_counter()
                            if entry=='python':
                                r=aleph.run(c,backend=backend,shots=args.shots,seed=552,cuda_optimize=optimize)
                                caller_ms=(time.perf_counter()-start)*1000
                                stats=r.execution_stats()
                                export_start=time.perf_counter();amplitudes=r.statevector()
                                export_ms=(time.perf_counter()-export_start)*1000
                                assert np.isfinite(amplitudes).all()
                                assert abs(np.vdot(amplitudes,amplitudes).real-1)<1e-3
                                assert sum(r.counts().values())==args.shots
                                del r,amplitudes;gc.collect()
                            else:
                                cmd=[args.binary,'run',str(path),'--backend',backend,'--shots',str(args.shots),'--seed','552','--cuda-diagnostics']
                                if not optimize:cmd+=['--cuda-raw']
                                p=subprocess.run(cmd,text=True,capture_output=True,check=True)
                                caller_ms=(time.perf_counter()-start)*1000;export_ms=0
                                stats=json.loads(next(l.split('=',1)[1] for l in p.stderr.splitlines() if l.startswith('CUDA_EXECUTION=')))
                            assert stats['optimized']==float(optimize)
                            assert stats['tf32_effective']==0 and stats['tf32_launches_process_delta']==0
                            row=dict(family=family,backend=backend,entry=entry,qubits=args.qubits,shots=args.shots,pair=pair,optimize=optimize,
                                caller_ms=caller_ms,export_ms=export_ms,stats=stats,gpu_before=before,gpu_after=gpu(),seed=552,host_load=os.getloadavg(),
                                warm_state='same-process/new-backend' if entry=='python' else 'fresh-process',executed=True)
                            out.write(json.dumps(row)+'\n');out.flush()
    print('All requested family oracles and paired public-route measurements executed successfully')


if __name__=='__main__':main()
