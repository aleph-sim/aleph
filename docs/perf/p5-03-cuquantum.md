# P5-03 — cuQuantum (cuStateVec) backend

`CuStateVecBackend` integrates NVIDIA cuStateVec (`custatevecApplyMatrix`) as an
optional GPU state-vector backend. It is the performance reference we benchmark
our own hand-written CUDA kernels (`CudaSvBackend`, P5-02) against, and the
max-performance path for users on NVIDIA hardware.

## What it is

- A `Backend` impl that routes gate application through cuStateVec while reusing
  the P5-02 device-resident state (`CudaSvState`, interleaved `[re, im]` FP64) and
  the identical host-side readout (measure / sample / expectation / probabilities).
  The two GPU backends are drop-in interchangeable; the oracle suite pins both.
- Hand-written FFI (`src/cuquantum/sys.rs`) for five entry points — create /
  destroy / set-stream / apply-matrix (+ its workspace query) — transcribed from
  `custatevec.h` (cuStateVec 1.13) and `library_types.h` (CUDA 13.0). No
  `bindgen` build dependency for a four-function surface.
- Gated behind the `cuquantum` feature (a superset of `cuda`). `build.rs` links
  `libcustatevec` **only** when that feature is on, so the default build, the
  `cuda`-only build, macOS, and the CUDA-less CI runner never need cuQuantum
  installed. CI builds `aleph-cuda` with `--features cuda` only, so it is
  unaffected.

### Interop notes

- The state vector buffer cuStateVec mutates is the same `cudarc` device
  allocation our kernels use. cuStateVec shares the CUDA **primary context** and
  virtual address space that `cudarc` retains via the driver API, so a device
  pointer minted by `cudarc` is a valid `sv` argument — the standard
  driver/runtime interop contract. `custatevecSetStream` binds cuStateVec to
  `cudarc`'s stream so applies, allocations, and device→host readback stay
  ordered.
- **Bit ordering.** Both aleph and cuStateVec index qubit `q` as bit `q` of the
  linear state index (little-endian). For multi-target gates, `gate.matrix()`
  lays operands out MSB-first (`qubits[0]` = MSB of the matrix index) whereas
  cuStateVec's `targets[0]` is the LSB. Reversing the operand list reconciles
  the two so the same row-major matrix acts on the same physical qubits — pinned
  by the Toffoli/Ccz (M8×8) oracle case, where distinct per-qubit rotations make
  every amplitude different so an operand-order bug cannot hide.

## Correctness

Oracle suite (`tests/cuquantum_oracle.rs`), FP64 tolerance **1e-10**, n = 2..12:
cuStateVec matches **both** the CPU `NaiveSvBackend` and our `CudaSvBackend`
amplitude-for-amplitude across GHZ, QFT, Grover (multi-control diffusion),
random brickwall, and Toffoli + Ccz.

## Performance

RTX 4000 SFF Ada Generation (20 GiB, sm_89), driver 580, CUDA 13.0,
cuStateVec 1.13. Random brickwall, depth 20, FP64. Full end-to-end run
(includes the final device→host amplitude sync). EPYC CPU baseline is the
multi-thread `NaiveSvBackend` on the same box.

| n  | cuStateVec | aleph-cuda-sv (P5-02) | CPU SV   | cuSV vs CPU | aleph-sv ÷ cuSV |
|----|-----------:|----------------------:|---------:|------------:|----------------:|
| 24 |     2.224s |                2.507s |  10.148s |       4.56× |           1.13× |
| 26 |     9.574s |               10.992s |  43.868s |       4.58× |           1.15× |
| 28 |    41.064s |               48.645s | 188.219s |       4.58× |           1.18× |

Reproduce:

```bash
ALEPH_PERF_N=28 cargo test -p aleph-cuda --features cuquantum --release \
  -- --ignored --nocapture perf_cuquantum
```

### Reading the numbers

- cuStateVec is **4.56–4.58×** the multi-thread CPU SV on this (modest) GPU. The
  RTX 4000 Ada is bandwidth-limited relative to a datacenter card; the speedup
  is GPU/CPU-bandwidth bound, not a cuStateVec ceiling.
- Our **own** FP64 kernels (P5-02) are within **1.13–1.18×** of cuStateVec and the
  gap grows only slowly with n — comfortably inside the ROADMAP § 7 target of
  "GPU backend within 1.5× of cuQuantum standalone." For dense statevector on a
  single GPU we are not trying to beat NVIDIA (ROADMAP § 1); landing this close
  with hand-written NVRTC kernels is the honest validation that the P5-02 backend
  is well-tuned, and cuStateVec is the integrated max-performance path.
