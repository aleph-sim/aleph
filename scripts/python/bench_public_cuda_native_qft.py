"""Bounded native c.cp QFT public-route check; complements the parser decomposition.
Run with the candidate CUDA wheel. No service changes. Output is raw paired data.
"""
import argparse
import gc
import json
import math
import time
import numpy as np
import aleph
from bench_public_cuda_execution import idle,gpu


def qft(n):
    c=aleph.Circuit(n)
    for q in range(n):
        c.h(q)
        for j in range(q+1,n):c.cp(math.pi/2**(j-q),j,q)
    return c


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    small=qft(8);expected=aleph.run(small,backend='sv',shots=0).statevector()
    for backend,tol in [('cuda',1e-10),('cuda-f32',1e-5)]:
        for optimize in [False,True]:
            r=aleph.run(small,backend=backend,shots=0,cuda_optimize=optimize)
            np.testing.assert_allclose(r.statevector(),expected,atol=tol,rtol=0)
    c=qft(22)
    with open(a.output,'w') as output:
        for backend in ['cuda','cuda-f32']:
            for optimize in [False,True]:aleph.run(c,backend=backend,shots=32,seed=552,cuda_optimize=optimize)
            for pair in range(3):
                for optimize in ([False,True] if pair%2==0 else [True,False]):
                    before=idle();start=time.perf_counter()
                    r=aleph.run(c,backend=backend,shots=32,seed=552,cuda_optimize=optimize)
                    caller_ms=(time.perf_counter()-start)*1000
                    stats=r.execution_stats();start=time.perf_counter();v=r.statevector()
                    export_ms=(time.perf_counter()-start)*1000
                    assert np.isfinite(v).all()
                    output.write(json.dumps(dict(family='native-qft',backend=backend,entry='python',qubits=22,shots=32,pair=pair,optimize=optimize,caller_ms=caller_ms,export_ms=export_ms,stats=stats,gpu_before=before,gpu_after=gpu(),executed=True))+'\n');output.flush()
                    del r,v;gc.collect()
    print('Native controlled-phase QFT oracle and 12 paired public calls executed')


if __name__=='__main__':main()
