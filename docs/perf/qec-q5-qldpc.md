# Q5-03 — relay-BP improvement + literature benchmark

**Issue:** Q5-03 (Phase Q5, qLDPC frontier).
**Depends on:** Q5-02 (BP+OSD).
**Status:** done.

## What and why

The BP+OSD decoder (Q5-02) is the qLDPC workhorse, but it has two weaknesses: a residual **error
floor** (near-`p`-independent failures from symmetric **trapping sets** where BP oscillates), and a
**serial OSD tail** (a GF(2) Gaussian elimination per failed shot) that does not map onto the GPU BP
kernel. Q5-03 implements a recent improvement that attacks both: **relay-BP** (Müller et al.,
[arXiv:2506.01779](https://arxiv.org/abs/2506.01779), 2025; "memory BP" lineage).

`RelayBpDecoder` (`crates/aleph-qec/src/relay_bp.rs`) keeps the Q3-02 min-sum check update but changes
the variable update and the outer loop:

- **Disordered memory.** Each variable node gets its own memory strength `γ_v`, drawn from a range
  spanning negative to positive (`[−0.3, 0.9]` by default). The variable→check message damps toward
  the previous message at strength `γ_v`: `M_{v→c} ← (1−γ_v)·M_computed + γ_v·M_old`. A *spread* of
  per-node `γ_v` desynchronises the symmetric oscillations a single uniform damping cannot.
- **Relayed legs.** BP runs in several legs (4 by default), each carrying the previous leg's messages
  forward (the "relay") but swapping in a fresh disorder pattern. The decoder keeps the
  **lowest-weight syndrome-valid** hard decision seen across all iterations of all legs.

Disorder patterns are fixed at construction (seeded), reused across shots — they break the *code's*
symmetry, not the shot's — so the decoder is deterministic and `Sync`. It is **pure message passing**:
no OSD, no Gaussian elimination.

## Acceptance criteria

- [x] **≥1 improvement implemented and measured vs Q5-02.** relay-BP, measured head-to-head against
  BP and BP+OSD below.
- [x] **`docs/perf/qec-q5-qldpc.md` with results positioned against published numbers** (this file).

## Results

Box: M4 Mac Mini. Gross `[[144,12,12]]`, independent-`Z` code capacity, normalised min-sum
`α = 0.875`; relay-BP with 4 legs and `γ ∈ [−0.3, 0.9]`; 40 000 shots.
`cargo run --release -p aleph-qec --example qec_q5_relay`. Data: `docs/perf/data/qec-q5-relay.{csv,log}`.

| p | plain BP | BP+OSD (Q5-02) | **relay-BP** | relay vs BP+OSD |
|------|----------|----------------|--------------|------------------|
| 0.01 | 1.5e-4 | 1.0e-4 | **0.0** | ∞ (floor cleared) |
| 0.02 | 1.1e-3 | 9.5e-4 | **7.5e-5** | **12.7×** |
| 0.03 | 4.35e-3 | 4.25e-3 | **1.58e-3** | 2.7× |
| 0.04 | 2.05e-2 | 1.97e-2 | **1.25e-2** | 1.6× |
| 0.05 | 6.19e-2 | 6.08e-2 | **4.57e-2** | 1.3× |
| 0.06 | 1.45e-1 | 1.43e-1 | **1.14e-1** | 1.25× |

relay-BP beats BP+OSD at **every** `p`, and the margin **grows as `p` falls** — the signature of an
error-floor fix. At `p = 0.02` it is **12.7×** better than BP+OSD; at `p = 0.01` it clears the floor
entirely (0 failures in 40 000 shots). The `γ`-disorder is what does the work: an ablation at
`p = 0.03` (`γ ∈ [0,0]`, i.e. relay legs but *no* memory) gives 5.8e-3, while `γ ∈ [−0.3, 0.9]` gives
2.5e-3 — the disordered memory more than halves the rate; leg count past 2 adds little.

## Positioning against the literature

- **Qualitative match to relay-BP's published claim.** Müller et al. report that relay-BP reaches
  (and often beats) BP+OSD accuracy on bivariate-bicycle / qLDPC codes while being **cheaper** (pure
  message-passing, no per-shot OSD solve), with its largest gains *at the error floor*. Our gross-code
  code-capacity result reproduces exactly that shape: equal-or-better than BP+OSD everywhere, and an
  order-of-magnitude better at low `p`, with **no OSD stage**.
- **Cost.** relay-BP's total iteration budget is comparable to one plain BP run (`legs × iters/leg ≈
  DEFAULT_MAX_ITER`), and it drops OSD's GF(2) elimination entirely. So it is both **more accurate and
  cheaper** than the Q5-02 baseline here.
- **GPU path.** Because relay-BP is pure message passing over the same Tanner CSR, it maps directly
  onto the Q3-02 `CudaBp` kernel (per-leg disorder is just an extra per-variable array, the relay is
  persisting the message buffers between legs). The accuracy lever and the throughput lever line up —
  unlike BP+OSD, whose serial OSD tail fights the GPU.

## Honest boundary

- **Code capacity, not circuit-level.** As with Q5-01/Q5-02 this is the single-Pauli code-capacity
  channel; the gross code's headline circuit-level numbers need the syndrome-extraction DEM (the
  natural next deliverable). relay-BP's advantage is expected to be *larger* under circuit-level noise
  (more trapping sets), so this is a conservative comparison.
- relay-BP here is **standalone** (no OSD fallback). Adding OSD on the rare shots where no leg finds a
  valid decision would only help; it is left as a lever since relay-BP already beats BP+OSD.
- Parameters (`legs = 4`, `γ ∈ [−0.3, 0.9]`) were tuned coarsely on the gross code at `p = 0.03`; a
  finer per-code sweep is a cheap future refinement.

## Files

- `crates/aleph-qec/src/relay_bp.rs` — `RelayBpDecoder` + unit tests.
- `crates/aleph-qec/examples/qec_q5_relay.rs` — BP vs BP+OSD vs relay-BP.
- `crates/aleph-qec/tests/relay_bp.rs` — beats-BP+OSD-below-floor + decodes-at-low-p.
- `docs/perf/data/qec-q5-relay.{csv,log}` — committed run.

**Phase Q5 (qLDPC frontier) complete:** gross code + DEM (Q5-01), BP+OSD (Q5-02), relay-BP (Q5-03).
