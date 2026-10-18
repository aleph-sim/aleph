#!/usr/bin/env python3
"""P8-10 matched counter profiles, separate from timing acceptance.
Usage: profile_cooperative_bp.py BINARY CAMPAIGN_DIRECTORY NEW_OUTPUT_DIRECTORY
Requires Nsight Compute `ncu` on PATH. Services/clocks are unchanged. Each profiled
kernel is the first timed call after two unprofiled warmups and cached CPU checks.
"""
import hashlib
import json
import pathlib
import subprocess
import sys


def main():
    binary,campaign,directory=sys.argv[1:]
    root=pathlib.Path(directory);root.mkdir()
    metrics='gpu__time_duration.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,dram__bytes_read.sum,dram__bytes_write.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__thread_inst_executed_per_inst_executed.ratio,launch__registers_per_thread'
    receipts=[]
    for model,n,p in [('surface7',1,.03),('surface7',4096,.001),('surface11',4096,.03),('gross',1,.03),('gross',4096,.03)]:
        for mode in ['shot','warp','coop']:
            oracle=pathlib.Path(campaign)/'oracles'/f'{model}-n{n}-p{p}.json'
            assert oracle.exists()
            argv=['ncu','--csv','--kernel-name','bp_decode','--launch-skip','2','--launch-count','1','--clock-control','none','--cache-control','all','--metrics',metrics,binary,model,str(n),str(p),'1',mode,str(oracle)]
            name=f'{model}-n{n}-p{p}-{mode}'
            result=subprocess.run(argv,capture_output=True,text=True)
            (root/(name+'.csv')).write_text(result.stdout);(root/(name+'.log')).write_text(result.stderr)
            receipts.append(dict(label=name,argv=argv,exit_code=result.returncode))
            (root/'manifest.json').write_text(json.dumps(dict(binary_sha256=hashlib.sha256(pathlib.Path(binary).read_bytes()).hexdigest(),profiles=receipts),indent=2)+'\n')
            assert result.returncode==0,(name,result.stdout,result.stderr)


if __name__=='__main__':
    main()
