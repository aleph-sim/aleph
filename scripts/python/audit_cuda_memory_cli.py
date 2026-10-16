"""Bounded real-CUDA CLI output/memory-policy checks on the reference ~20 GiB card.
Run with --binary /path/to/aleph (built with --features cuda). No intentional OOM.
"""
import subprocess,pathlib,json,argparse,tempfile,re,math
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary',required=True,type=pathlib.Path)
args=parser.parse_args()
binary=args.binary.resolve()
workspace=tempfile.TemporaryDirectory(prefix='aleph-memory-cli-')
root=pathlib.Path(workspace.name)
results=[]
for n in (30,31):
 f=root/f'large{n}.qasm';f.write_text(f'OPENQASM 3.0; include "stdgates.inc"; qubit[{n}] q; h q[0];')
 for backend in ('cuda','cuda-f32'):
  p=subprocess.run([str(binary),'run',str(f),'--backend',backend,'--shots','16'],capture_output=True,text=True)
  assert p.returncode!=0 and 'memory plan needs' in p.stderr and 'sampling' in p.stderr,(p.stdout,p.stderr)
  assert 'norm only' not in p.stdout and '‖' not in p.stdout
  results.append(dict(qubits=n,backend=backend,rejected=True,error=p.stderr.strip()))
f=root/'bell.qasm';f.write_text('OPENQASM 3.0; include "stdgates.inc"; qubit[2] q; h q[0]; cx q[0],q[1];')
for backend in ('cuda','cuda-f32'):
 for flags in (['--shots','16'],['--statevector'],['--expectation','ZZ']):
  p=subprocess.run([str(binary),'run',str(f),'--backend',backend,*flags,'--seed','554'],capture_output=True,text=True)
  assert p.returncode==0,(p.stdout,p.stderr)
  if flags[0]=='--shots':
   counts=re.findall(r'\|([01]{2})⟩\s+(\d+)\s+\(',p.stdout)
   assert counts and all(b in ('00','11') for b,_ in counts)
   assert sum(int(v) for _,v in counts)==16
  elif flags[0]=='--statevector':
   rows=re.findall(r'\|([01]{2})⟩\s+(\S+)\s+(\S+)i',p.stdout)
   assert len(rows)==4
   for bits,real,imag in rows:
    expected=1/math.sqrt(2) if bits in ('00','11') else 0
    assert abs(float(real)-expected)<1e-6 and abs(float(imag))<1e-6
  else:
   assert 'ZZ' in p.stdout and abs(float(p.stdout.splitlines()[-1].split()[-1])-1)<1e-6
  results.append(dict(backend=backend,flags=flags,output=p.stdout.strip()))
print(json.dumps(results,indent=2))
