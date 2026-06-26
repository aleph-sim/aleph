# Q3-02 — GPU belief-propagation decoder (min-sum)

Phase Q3's second decoder: **min-sum belief propagation** over a Tanner graph, on the GPU. BP is the
opposite of Union-Find for hardware — dense, regular, message-passing — so it maps cleanly onto GPUs,
and it is the **bridge to qLDPC** (Phase Q5): unlike MWPM/UF, BP needs no graphlike DEM and handles
checks that touch many qubits. This ticket builds both halves: a CPU [`BpDecoder`] reference (also
the front end of the BP+OSD decoder, [Q5-02]) and a GPU [`CudaBp`] that decodes a **batch** of
syndromes, one thread per shot.

**Verdict — both acceptance criteria met:**

- **Converges to the correct correction on small surface/repetition codes.** The CPU decoder
  recovers every single-bit repetition-code error and reproduces low-weight surface-code syndromes
  (`bp.rs` unit tests).
- **Batched GPU throughput, matching the CPU reference within tolerance.** GPU vs CPU corrections are
  **numerically identical — zero disagreements** across `d ∈ {3,5,7,9,11}`, the repetition code, and
  a 100 000-shot batch (`tests/qec_bp_oracle.rs`); the GPU is **4.5–8.4× the single-thread CPU**.

## Design — one thread per shot, exact-match `double`

The kernel mirrors the GPU Union-Find decoder (Q3-01): **one GPU thread per syndrome shot**, each
running the full min-sum schedule on its own syndrome against a shared device-resident Tanner graph.
BP is regular enough that a block-cooperative variant could parallelise a single decode, but
one-thread-per-shot is the simplest design that (a) saturates the card on a large batch and (b)
guarantees the result matches the CPU.

Because every thread replays the **identical** message schedule in **`double`**, in the identical
edge order, the hard-decision error vector `ê` — and thus the correction — is reproduced
**bit-for-bit**, IEEE inf/NaN semantics included (a degree-1 check legitimately emits ±∞; the kernel
uses a true IEEE infinity sentinel so the arithmetic agrees with the CPU's `f64::INFINITY`). There
are no fused-multiply-add patterns in the loop, so no FP contraction can split GPU from CPU. The host
consumes the CPU decoder's own flattened Tanner arrays ([`BpDecoder::tanner`] → [`TannerGraph`]) — a
single source of truth for the layout.

The min-sum rule: variables are the DEM's error mechanisms (prior LLR `λ_v = ln((1-p_v)/p_v)`),
checks are detectors. Per iteration, check→variable messages take the violated-check sign flip times
the exclusive-minimum magnitude (scaled by `α`), variable→check messages re-sum the priors and
incoming messages, and BP stops once the hard decision satisfies `H ê = s`.

## Throughput (50 000-shot batch, uniform `p = 3 %`, α = 0.875, 64 iterations)

CPU is single-thread `Decoder::decode`; GPU is whole-batch decode including upload, launch and
download. Best of three each. **`mismatches` is the per-cell GPU≠CPU count — zero everywhere.**

| `d` | detectors | vars | edges | avg defects | CPU syn/s | **GPU syn/s** | speed-up | mismatches |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 3 | 16 | 40 | 64 | 1.9 | 498 000 | **2 605 000** | 5.2× | 0 |
| 5 | 72 | 186 | 336 | 9.5 | 27 300 | **228 000** | 8.4× | 0 |
| 7 | 192 | 512 | 960 | 26.5 | 5 180 | **26 800** | 5.2× | 0 |
| 9 | 400 | 1 090 | 2 080 | 56.6 | 1 920 | **8 590** | 4.5× | 0 |
| 11 | 720 | 1 992 | 3 840 | 103.5 | 861 | **5 950** | 6.9× | 0 |

**Reading the table.**

- The GPU beats the single-thread CPU at every distance, **4.5–8.4×**. The speed-up is not monotone
  in `d`: it is set by occupancy and per-thread divergence (iteration counts vary by syndrome), not
  by a clean O(·) law.
- **Absolute** BP throughput is well below the GPU UF decoder (Q3-01: 112 000 syn/s at `d=11` vs
  BP's 5 950): BP runs a *fixed* 64 message-passing iterations over every variable–check edge per
  shot, whereas UF terminates in ~O(d) cheap growth rounds. BP buys generality (arbitrary
  parity-check graphs / qLDPC), not speed, on surface codes.
- Pure BP is also **degeneracy-limited** on surface codes — split beliefs on homologically
  equivalent errors keep its standalone logical accuracy *below* MWPM/UF. That is expected and is
  exactly what **BP+OSD** ([Q5-02]) fixes by post-processing BP's soft output with ordered-statistics
  decoding. This ticket delivers the GPU BP kernel that BP+OSD will sit on; it does not claim BP is
  the best surface-code decoder.

## Reproduce

On the CUDA box (`openwebgui.splynx.com`, RTX 4000 SFF Ada, sm_89):

```bash
# Oracle: GPU == CPU BP corrections (repetition + surface + 1e5 batch).
cargo test -p aleph-cuda --features cuda --test qec_bp_oracle -- --nocapture

# Throughput table above (CSV to stdout; correctness guard in the last column).
cargo run --release -p aleph-cuda --features cuda --example qec_q3_gpu_bp -- 50000 2024

# CPU BP unit tests (repetition recovers errors; surface low-weight converges).
cargo test -p aleph-qec --lib bp
```

Raw data: [`data/qec-q3-gpu-bp.csv`](data/qec-q3-gpu-bp.csv).

## What this validates

- A **GPU belief-propagation kernel** that is provably equal to a CPU BP reference — the precondition
  for trusting any decoder — and faster on batch traffic. BP's regular, matrix-like message passing
  is a natural GPU fit, distinct from the irregular Union-Find growth (Q3-01).
- The **qLDPC on-ramp**: BP consumes an arbitrary parity-check graph (not just graphlike DEMs), so
  the bivariate-bicycle / gross-code work in Phase Q5 already has its decode kernel. The next step is
  layering OSD on top (BP+OSD, [Q5-02]) for competitive qLDPC accuracy.

## References

- M. P. C. Fossorier, M. Mihaljević, H. Imai, **Reduced complexity iterative decoding of LDPC codes**
  (min-sum), IEEE Trans. Commun. 47 (1999).
- J. Roffe, D. R. White, S. Burton, E. Campbell, **Decoding across the quantum LDPC code landscape**,
  Phys. Rev. Research 2, 043423 (2020) — BP+OSD.
- P. Panteleev, G. Kalachev, **Degenerate quantum LDPC codes with good finite length performance**,
  Quantum 5, 585 (2021).

[Q5-02]: ../qec/BACKLOG.md
[`BpDecoder`]: ../../crates/aleph-qec/src/bp.rs
[`BpDecoder::tanner`]: ../../crates/aleph-qec/src/bp.rs
[`TannerGraph`]: ../../crates/aleph-qec/src/bp.rs
[`CudaBp`]: ../../crates/aleph-cuda/src/qec/bp.rs
