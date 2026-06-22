# P5-05 — CPU↔GPU transfer optimization (GPU-resident readout)

Keep the state vector on the GPU across the whole circuit and compute readout
results on the device, so the only PCIe crossings are the initial `|0…0⟩` setup
and the final (small) results.

## The problem

After P5-02/03/04 the state was already device-resident *during gate
application* — but every readout downloaded the full `2^n` amplitudes to the
host and reduced on the CPU, and `measure` additionally **re-uploaded** the
collapsed state. For a 1 GiB state (n=26) that is a ~1 GiB download per
expectation/sample/probabilities call and a ~2 GiB round-trip per measurement —
exactly the spurious traffic this issue targets.

## What changed

A shared GPU readout module (`src/sv/readout.rs` + `readout.cu`) reduces the
device state on the GPU and copies back only the small result. Both GPU backends
hold a `GpuReadout` and delegate to it — they store the identical interleaved
`[re, im]` f64 buffer, so one kernel set serves both:

| op | on the GPU | crosses PCIe |
|----|-----------|--------------|
| `measure` | branch-prob reduction + **in-place collapse** | one scalar (16 B); no re-upload |
| `expectation_value` | Pauli reduction `Σ conj(ψ_i)ψ_{i⊕F}(−1)^…` | one scalar |
| `probabilities` | marginal histogram into `2^k` bins | the `2^k` marginal |
| `sample` | `|a|²` → inclusive scan (CDF) → per-shot upper-bound search | the `shots` indices |

Scalar reductions are a **two-pass tree reduction** (per-block subtotals → a
single-block final reduce) — no global atomics, so no cross-block contention.
The cheap validation (qubit range / duplicate / Pauli / degenerate-branch /
norm-drift) stays on the host where it costs nothing, matching the CPU backend's
semantics exactly.

`measure` collapses in place on the device: the state is mutated by a kernel and
never leaves the GPU.

## Correctness

`tests/readout_oracle.rs` pins **both** backends against the CPU `NaiveSvBackend`
at FP64 tolerance 1e-9:

- expectation over Z / X / Y / mixed / identity strings (with coefficients) —
  this caught a Pauli-phase sign bug (the `(−1)^popcount` must be evaluated at
  `i⊕flip`, not `i`, so Y terms differ);
- marginals over several qubit subsets;
- `measure` outcomes (same seed ⇒ identical branch decision and the GHZ chain
  exercises the degenerate/no-RNG path) and the collapsed amplitudes;
- `sample` empirical distribution vs the exact one (400k shots, ≤1% per bin) and
  same-seed reproducibility.

## Transfer verification

`tests/transfer.rs` asserts the lazy-transfer invariant with an in-process
device→host byte counter (`aleph_cuda::device_dtoh_bytes`) — more deterministic
than an Nsight trace and a regression guard against any reintroduced full-state
download:

- applying the whole circuit copies **0 bytes** device→host;
- each readout copies back far less than the state (the test ceiling is
  `state/64`; the actuals are a handful of bytes to a few KB).

## Performance

RTX 4000 SFF Ada, FP64. Expectation on a fully-populated state, GPU-resident vs
the bytes a full-state download would have moved:

| n  | state   | readout copies | full-state download | PCIe reduction | wall-clock |
|----|---------|---------------:|--------------------:|---------------:|-----------:|
| 24 | 256 MiB |          32 B  |             256 MiB |    8.4 million× |  10.4 ms/op |
| 26 |   1 GiB |          32 B  |               1 GiB |   33.6 million× |  42.3 ms/op |

The end-to-end readout is also **faster than the old path**: downloading 256 MiB
over this box's PCIe (~25 GB/s) already costs ~10 ms *before* the CPU reduction,
which the GPU path now does in the same time while moving ~10⁷× less data and
freeing the host. The reduction kernel itself is correctness-first and still
some way off peak bandwidth — further kernel tuning is P5-06 territory.

Reproduce:

```bash
ALEPH_READOUT_N=24 cargo test -p aleph-cuda --features cuda --release \
  -- --ignored --nocapture readout_throughput
```
