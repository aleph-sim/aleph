# P8-08 — lazy GPU discovery in public frontends

Issue: [#558](https://github.com/aleph-sim/aleph/issues/558).

## Change and contract

Previously CLI constructed both availability probes before resolving an explicit/auto
request. Its Metal probe constructed `MetalSvBackend`; CUDA CLI/Python constructed
`CudaSvBackendF32` (including NVRTC compilation). The selected GPU backend was then
constructed again. Python kept Metal explicit-only and continues to do so.

`select_with_probes` resolves explicit requests without analysis/probes. Auto analyzes
once and evaluates whether hypothetical device availability changes the existing pure
selection rule. Only a relevant device is queried. Clifford, small CPU, and large shallow
MPS circuits skip GPU discovery. CLI's amplitude-driven stabilizer downgrade is preserved.
The old `resolve_backend` signature remains available for callers with an existing environment.

Metal discovery calls `Device::system_default` without a queue or shaders. CUDA checks the
optional driver library and device count without a context, pool, NVRTC, or backend. Absent
libraries/devices/query failure leave auto on its existing CPU/MPS/stabilizer choice.
Discovery does not guarantee backend readiness: a selected GPU's initialization/kernel
compilation failure remains an actionable execution error, not a silent CPU fallback.
CUDA SV construction checks optional driver/NVRTC libraries before cudarc's panicking loaders.
No module cache is added; the selected backend is constructed once per public run.

## Evidence and measurement

Local macOS arm64, release build. `probe_startup` invokes the public CPU CLI path on a
2-qubit Bell fixture, 32 shots, seed 7. `legacy` replays the exact former eager Metal
constructor before the same current CLI path; this isolates removed setup work. It is
**not an old-revision binary comparison**, a whole-process timer, or a gate-throughput claim.
Each row is a separate process; 20 calls, first call separated from the median of the
remaining 19. Interleaved order: legacy/lazy/lazy/legacy/legacy/lazy.

| Mode | First call (ms) | Warm median (ms) | SV construction attempts |
|------|----------------:|-----------------:|-------------------------:|
| legacy | 119.833 | 0.664 | 20 |
| lazy | 0.078 | 0.0248 | 0 |
| lazy | 0.065 | 0.0184 | 0 |
| legacy | 42.843 | 0.612 | 20 |
| legacy | 44.842 | 0.684 | 20 |
| lazy | 0.0605 | 0.0173 | 0 |

The reliable regression invariant is construction count: explicit CPU calls do no GPU
setup. The local machine was busy during development, so these illustrative setup timings
are not a calibrated idle-machine speedup or confidence interval. First calls include
framework/driver/shader-cache initialization effects and vary substantially.

The built abi3 Python/Metal wheel was run with bundled Python 3.12 and NumPy 2.3.5.
`scripts/python/bench_probe_startup.py` reports import, first-call, and repeated-call time:

| Request | Calls | First run (ms) | Warm median (ms) | SV construction attempts |
|---------|------:|---------------:|-----------------:|-------------------------:|
| sv | 20 | 0.229 | 0.00504 | 0 |
| auto (Clifford) | 20 | 0.0393 | 0.00275 | 0 |
| metal | 3 | 250.960 | 1.507 | 3 |

These are candidate smoke/counter measurements, **not a Python before/after speedup**:
Python did not auto-probe Metal before this change. Linux/CUDA Python is the platform
where the removed eager compiler setup must be measured before/after.

## Validation

- Shared policy: 36 backend unit tests passed, including all explicit kinds and a
  lazy-versus-pure selection matrix across circuit shapes, availability and precision.
- CLI: 35 unit tests, 40 command integration tests and the new public CPU/Clifford
  initialization-counter regression passed. The counter test also passed with Metal enabled.
- Real Metal: device discovery left construction count unchanged; backend construction
  incremented once; deliberately invalid shader source failed compilation while device
  discovery remained true. Sandboxed execution hid the GPU; the real smoke ran outside it.
- Python: maturin abi3 Python/Metal wheel built; 27 binding tests passed; public counter
  measurements confirm zero CPU/auto and one-per-call explicit Metal constructions.
- Targeted default/Metal/Python clippy, formatting and diff checks passed during development.

**Pending:** Linux CUDA build/hardware smoke and actual baseline/candidate cold/warm Python
measurements. Copying the baseline snapshot/patch to the shared CUDA host was rejected by
automatic approval review pending explicit user authorization. No source was transferred,
no shared GPU services were paused, and no CUDA speedup is claimed by this report.

## Reproduce

```bash
cargo test -p aleph-backend --lib
cargo test -p aleph-cli --lib --test cli --test gpu_probe
cargo test -p aleph-cli --features metal --test gpu_probe
cargo test -p aleph-metal --features metal --test device_probe -- --nocapture
cargo run --release -p aleph-cli --features metal --example probe_startup -- legacy 20
cargo run --release -p aleph-cli --features metal --example probe_startup -- lazy 20
python scripts/python/bench_probe_startup.py --backend sv --repeats 20
python scripts/python/bench_probe_startup.py --backend auto --repeats 20
python scripts/python/bench_probe_startup.py --backend metal --repeats 3
```

On an authorized Linux CUDA host, use `--features cuda` for the CLI example/regression and
`cargo test -p aleph-cuda --features cuda --test device_probe -- --nocapture`. Compare actual
baseline/candidate wheels in fresh interleaved processes; hardware smoke must report execution
rather than silently treating missing devices as performance evidence.
