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

## Actual Linux/CUDA before/after

Measured 2026-10-13 on Linux 6.8.0-146-generic, NVIDIA RTX 4000 SFF Ada
Generation (20 GiB), driver 580.178.04, Rust 1.96.0, Python 3.12.3 and NumPy 2.5.0.
Baseline `c7be899f5b22ac528d727f12433da80934f98a96`; candidate
`333e5b525665cb0f237b08ba1d070035bfc26ff5`. Both use locked release CUDA builds.
CLI binaries and maturin wheels were saved separately, with **separate Cargo target
directories**. An initial shared-target attempt reused stale artifacts; its timings were
discarded. The final wheel checks require the baseline to lack the new counter and the
candidate to expose its expected count, in addition to checking distinct artifact hashes.

Each backend uses fresh processes in baseline/candidate/candidate/baseline/baseline/candidate
order: three samples per revision. Same 2-qubit Bell circuit, 32 shots and seed 7.
CLI times are subprocess wall times. Python cold times are whole subprocess wall times
with **one** public call (including interpreter startup, import, execution and exit).
Python warm times are the median of calls 2–20 within each interpreter; the table reports
the median of those three medians. No timing assertions gate tests.

| Public request | Cold process baseline → candidate (ms) | Warm Python baseline → candidate (ms) | Candidate constructions per 20 calls |
|----------------|---------------------------------------:|--------------------------------------:|------------------------------------:|
| CLI statevector | 194.561 → 1.271 | — | — |
| CLI Clifford auto | 192.281 → 1.255 | — | — |
| CLI cuda-f32 | 357.078 → 196.071 | — | — |
| Python sv | 212.764 → 24.286 | 159.880 → 0.00203 | 0 |
| Python Clifford auto | 210.163 → 24.276 | 160.016 → 0.00201 | 0 |
| Python cuda-f32 | 373.970 → 217.536 | 319.756 → 159.047 | 20 |

Warm Python median ranges across the three samples: baseline sv 158.461–160.042 ms,
candidate sv 0.00198–0.00203 ms; baseline auto 159.185–160.116 ms, candidate auto
0.00188–0.00202 ms; baseline cuda-f32 319.326–319.761 ms, candidate cuda-f32
157.960–159.531 ms. Import medians were approximately 0.37–0.41 ms for both wheels
in the 20-call measurements. Candidate GPU first calls still cost approximately 189 ms;
there is no module cache, and each GPU run still compiles its selected backend.

The historical wheel has no initialization counter, so baseline counts are recorded as
null, not inferred measurements. Candidate counters confirm zero discarded CPU/Clifford
constructions and exactly one selected GPU construction per call. CLI baseline/candidate
stdout matched for every measured request. Python Bell counts were valid and deterministic
within each process. These results concern frontend setup on tiny circuits, not GPU gate
throughput or large-circuit execution.

The host remained shared: no services or watchdog were paused. GPU memory was 1339 MiB
before/after; utilization was 0% before and 3% after, CPU load averages 1.92/1.25/0.57
before and 1.41/1.22/0.61 after. Only three samples per revision were taken. Treat times
as observations on this host, not isolated-card calibration or statistical confidence.
Raw per-process measurements, including import/first-call/warm values and counters:
[data/p8-08-cuda-startup.json](data/p8-08-cuda-startup.json) and
[data/p8-08-cuda-cold-python.json](data/p8-08-cuda-cold-python.json).

## Validation

- Shared policy: 36 backend unit tests passed, including all explicit kinds and a
  lazy-versus-pure selection matrix across circuit shapes, availability and precision.
- CLI: 35 unit tests, 40 command integration tests and the new public CPU/Clifford
  initialization-counter regression passed. The counter test also passed with Metal and CUDA enabled.
- Real Metal and CUDA: discovery left construction count unchanged; backend construction
  incremented once; deliberately invalid shader/CUDA source failed compilation while
  discovery remained true. Sandboxed macOS execution hid the GPU; Metal smoke ran outside it.
  CUDA smoke executed on the device rather than taking the absent-device skip path.
- Python: maturin abi3 Metal/CUDA wheels built; all 27 binding tests passed on both hosts;
  public measurements confirm zero CPU/auto and one-per-call explicit GPU constructions.
- Review follow-up: all 28 Python binding tests pass with both GPU-enabled wheels,
  including a permanent zero-construction regression for sv/stab/MPS/Clifford auto
  and small non-Clifford auto. Absent-device tests now assert explicit constructor
  failure and one construction attempt: exercised with sandbox-hidden Metal and
  process-local `CUDA_VISIBLE_DEVICES=""`, as well as the real CUDA smoke path.
- Workspace and targeted Metal/Python clippy, formatting and diff checks passed locally.
  All eight GitHub CI jobs passed for candidate `333e5b5`, including CUDA/NCCL clippy.

## Reproduce

```bash
cargo test -p aleph-backend --lib
cargo test -p aleph-cli --lib --test cli --test gpu_probe
cargo test -p aleph-cli --features metal --test gpu_probe
cargo test -p aleph-metal --features metal --test device_probe -- --nocapture
cargo test -p aleph-cli --features cuda --test gpu_probe
cargo test -p aleph-cuda --features cuda --test device_probe -- --nocapture
cargo run --release -p aleph-cli --features metal --example probe_startup -- legacy 20
cargo run --release -p aleph-cli --features metal --example probe_startup -- lazy 20
python scripts/python/bench_probe_startup.py --backend sv --repeats 20
python scripts/python/bench_probe_startup.py --backend auto --repeats 20
python scripts/python/bench_probe_startup.py --backend cuda-f32 --repeats 20
```

For actual before/after, build each revision with its own `CARGO_TARGET_DIR`:
`cargo build --release --locked -p aleph-cli --features cuda` and
`maturin build --release --locked -m crates/aleph-py/Cargo.toml --features python,cuda`.
Extract each wheel with Python `zipfile` into a separate directory and select it with
`PYTHONPATH`; do not install over the other wheel. Invoke the candidate benchmark script
with each wheel in the interleaved order above. Wrap each subprocess with
`time.perf_counter()` for whole-process wall time; use `--repeats 1` for Python cold
processes and `--repeats 20` for warm calls. Compare CLI stdout across revisions and
assert candidate counters are zero for sv/auto and equal to calls for cuda-f32.
