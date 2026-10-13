"""Paired public CLI raw/optimized wall times with executed-plan diagnostics.

Uses the same release binary for both modes; raw replays the former CLI driver.
This is a pipeline A/B, not an old-revision binary comparison. Compilation must
finish before measurements. Optional Metal must execute, never silently skip.
"""
import argparse
import hashlib
import json
import math
import pathlib
import statistics
import subprocess
import tempfile
import time

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True)
parser.add_argument("--backend", choices=["sv", "metal"], default="sv")
parser.add_argument("--precision", choices=["f64", "f32", "auto"], default="f64")
parser.add_argument("--qubits", type=int, default=20)
parser.add_argument("--output", required=True)
args = parser.parse_args()


def qft(n):
    gates = []
    for high in reversed(range(n)):
        gates.append(f"h q[{high}];")
        for low in reversed(range(high)):
            angle = math.pi / (2 ** (high-low)) / 2
            gates.extend([f"p({angle}) q[{high}];", f"cx q[{high}],q[{low}];",
                          f"p({-angle}) q[{low}];", f"cx q[{high}],q[{low}];",
                          f"p({angle}) q[{low}];"])
    for q in range(n//2):
        gates.append(f"swap q[{q}],q[{n-1-q}];")
    return gates


def brickwall(n):
    gates = []
    for layer in range(12):
        for q in range(n):
            angle = math.sin((layer+1)*(q+1))
            gates.extend([f"rz({angle}) q[{q}];", f"rx({angle*0.7+0.13}) q[{q}];"])
        for q in range(layer % 2, n-1, 2):
            gates.append(f"cx q[{q}],q[{q+1}];")
    return gates


rows = []
with tempfile.TemporaryDirectory(prefix="aleph-cli-p8-01-") as temp:
    for name, n, gates in [("tiny_bell", 2, ["h q[0];", "cx q[0],q[1];"]),
                            ("qft", args.qubits, qft(args.qubits)),
                            ("brickwall_d12", args.qubits, brickwall(args.qubits))]:
        source = 'OPENQASM 3.0; include "stdgates.inc";\n'+f'qubit[{n}] q;\n'+'\n'.join(gates)+'\n'
        path = pathlib.Path(temp) / (name+'.qasm')
        path.write_text(source)
        raw_output = None
        stable_outputs = {}
        for mode in ["raw", "optimized", "optimized", "raw", "raw", "optimized"]:
            cmd = [args.binary, 'run', str(path), '--backend', args.backend,
                   '--precision', args.precision, '--shots', '32', '--seed', '7',
                   '--execution-stats']
            if mode == 'raw':
                cmd.append('--raw')
            start = time.perf_counter()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            wall_ms = (time.perf_counter()-start)*1000
            if mode == 'raw':
                raw_output = result.stdout
            if mode in stable_outputs and result.stdout != stable_outputs[mode]:
                raise RuntimeError(f"{name}: same-mode seeded run is not reproducible")
            stable_outputs[mode] = result.stdout
            # Floating-point fusion may change alias-bin rounding and hence
            # seed-to-outcome mapping; correctness is checked by amplitude oracles.
            for line in result.stdout.splitlines()[1:]:
                if not line.strip().startswith('|'):
                    raise RuntimeError(f"unexpected counts output: {line}")
            total = sum(int(line.split()[1]) for line in result.stdout.splitlines()[1:])
            if total != 32:
                raise RuntimeError("invalid sample count")
            stats = {}
            for part in result.stderr.strip().split(';')[1:]:
                key, value = part.strip().split('=', 1)
                stats[key] = float(value) if key == 'planning_us' else int(value)
            row = dict(workload=name, qubits=n, backend=args.backend, precision=args.precision,
                       mode=mode, wall_ms=wall_ms, seeded_histogram_matches_raw=(result.stdout == raw_output), source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                       stdout_sha256=hashlib.sha256(result.stdout.encode()).hexdigest(), **stats)
            rows.append(row)
            pathlib.Path(args.output).write_text(json.dumps(rows, indent=2)+'\n')
            print(json.dumps(row), flush=True)
pathlib.Path(args.output).write_text(json.dumps(rows, indent=2)+'\n')
for name in ["tiny_bell", "qft", "brickwall_d12"]:
    for mode in ["raw", "optimized"]:
        values = [r['wall_ms'] for r in rows if r['workload']==name and r['mode']==mode]
        print(name, mode, 'median_ms', statistics.median(values), 'range', min(values), max(values))
