#!/usr/bin/env python3
"""Steady-state Sparse Blossom profile; usage BINARY NEW_DIRECTORY.
Attach perf only after Criterion enters its profiling phase. Not acceptance timing.
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time


def main():
    binary,directory=sys.argv[1:];root=pathlib.Path(directory);root.mkdir()
    cpu=3;sibling=23
    actual=pathlib.Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text().strip()
    assert actual=='3,23',actual
    env=dict(os.environ,RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    argv=['taskset','-c',str(cpu),binary,'--bench','sparse/d11','--profile-time','20','--warm-up-time','1','--noplot']
    with (root/'criterion.log').open('w') as log:
        child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
        for _ in range(300):
            if 'profiling for' in (root/'criterion.log').read_text().lower():break
            if child.poll() is not None:raise RuntimeError('benchmark exited before profiling')
            time.sleep(.1)
        else:
            child.terminate();raise RuntimeError('no steady-state marker')
        perf=['perf','record','-e','cycles:u','-F','499','--call-graph','dwarf,4096','-p',str(child.pid),'-o',str(root/'perf.data'),'--','sleep','10']
        result=subprocess.run(perf,capture_output=True,text=True)
        (root/'perf.log').write_text(result.stdout+result.stderr)
        code=child.wait();assert code==0 and result.returncode==0
    report=subprocess.run(['perf','report','--stdio','--no-children','-i',str(root/'perf.data')],capture_output=True,text=True,check=True)
    (root/'report.txt').write_text(report.stdout+report.stderr)
    (root/'manifest.json').write_text(json.dumps(dict(binary_sha256=hashlib.sha256(pathlib.Path(binary).read_bytes()).hexdigest(),cpu=cpu,sibling=sibling,argv=argv,perf_argv=perf,scope='profile attached after Criterion steady-state marker; not elapsed-time comparison'),indent=2)+'\n')
    print('PROFILE_PASS')


if __name__=='__main__':main()
