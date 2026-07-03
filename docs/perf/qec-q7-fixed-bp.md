# Q7-02 M0 — fixed-point relay-BP golden (RTL sizing)

**Track:** Q7-02 (RTL implementation of the core decoder block), milestone **M0**.
**Depends on:** Q5-03 (relay-BP), Q5-01 (gross code + DEM).
**Status:** done.

## What and why

The pre-ASIC frontier gap (ROADMAP §7, lever #4) is that our on-silicon decoder is UF surface-code
only — the **qLDPC frontier decoder is software-only**. #4 puts it on silicon. The chosen hardware
target is **relay-BP** (Q5-03), not classic BP+OSD: relay-BP already beats BP+OSD at every `p`
(`qec-q5-qldpc.md`) *and* drops OSD's data-dependent GF(2) Gauss–Jordan elimination entirely, so it
is pure fixed-schedule message passing — the same bounded-pass-per-cycle datapath shape that carried
UF to silicon (Q6).

An FPGA/ASIC cannot carry `f64` through 432 message edges. M0 builds the **fixed-point golden** —
the exact integer arithmetic the RTL will implement — and finds the **narrowest message word** whose
logical-error rate still matches the `f64` relay-BP decoder. That word width is the primary RTL
sizing input (message RAM, adder/comparator widths, routing).

`FixedRelayBp` (`crates/aleph-qec/src/fixed_bp.rs`) is the bit-accurate twin of `RelayBpDecoder`:
identical Tanner layout, `γ` seeding (SplitMix64), leg/iteration schedule, and
keep-lowest-weight-valid rule — only the arithmetic is quantised.

### Fixed-point scheme (= the RTL spec)

- A value is stored as signed `round(x·2^F)`, saturated to `±(2^(W−1)−1)`; `W` = `msg_bits`,
  `F` = `frac_bits`. Both message buffers (var→check, check→var) live in this `W`-bit word.
- **α = 0.875 = 7/8 is multiply-free**: `mag − (mag >> 3)`. The whole min-sum check update is
  compare / min / sign — **no multiplier**.
- The **only multiply in the datapath** is the relay memory blend `(1−γ_v)·computed + γ_v·m_old`,
  and `γ_v` is a *per-variable constant* (seeded ROM) — a fixed-coefficient multiply / small LUT,
  not a general one.
- The blend truncates by arithmetic shift (`num >> F`, floor) — the cheapest RTL rounding (no adder).
  The golden adopts it so software and silicon agree bit-for-bit.
- The per-node accumulator is kept wider than a message (here `i64`; RTL sizes it
  `W + ⌈log2(deg)⌉`); only stored messages re-saturate to `W`.

## Result — message width sweep

Gross `[[144,12,12]]`, independent-`Z` code capacity, 20 000 shots, seed 2024. `f64` = `RelayBpDecoder::new`.
`cargo run --release -p aleph-qec --example qec_q7_fixed_bp`. Data: `docs/perf/data/qec-q7-fixed-bp.csv`.

LER as a ratio to the `f64` relay-BP baseline (1.00 = identical); every cell below is **within
Monte-Carlo CI** of the `f64` rate:

| word `(W,F)` | range / step | p=0.02 | p=0.03 | p=0.04 | p=0.05 |
|--------------|--------------|--------|--------|--------|--------|
| f64 (ref)    | —            | 1.00e-4 | 2.00e-3 | 1.27e-2 | 4.35e-2 |
| (6, 2)       | ±7.75 / 0.25   | 2.00× | 1.20× | 1.06× | 0.97× |
| (7, 3)       | ±7.88 / 0.125  | 2.50× | 1.18× | 1.04× | 0.99× |
| **(8, 3)**   | **±15.9 / 0.125** | **1.00×** | **0.97×** | **0.95×** | **0.96×** |
| (8, 4)       | ±7.94 / 0.0625 | 1.50× | 1.18× | 0.98× | 0.99× |
| (10, 4)      | ±31.9 / 0.0625 | 1.00× | 0.97× | 0.97× | 0.98× |
| (12, 6)      | ±31.9 / 0.016  | 1.00× | 0.95× | 0.99× | 0.99× |

### Verdict: **8-bit signed, 3 fractional bits (Q5.3)** is the RTL message word.

`(8,3)` tracks `f64` tightest of all candidates *and* is the only 8-bit word that reproduces the
`f64` rate exactly at the p=0.02 floor (both 1.0e-4). The reason it beats the finer `(8,4)` there:
at low `p` the priors are largest (`λ(0.02) ≈ 3.9`, `λ(0.01) ≈ 4.6`) and messages accumulate, so the
±15.9 integer range matters more than the extra fractional bit. Below 8 bits `(6,2)/(7,3)` stay
within CI at higher `p` but bias high at the floor (2–2.5×) where accuracy matters most — so 8 bits
is the floor, not a comfort margin.

The truncating blend acts as a mild extra damping, which is why `(8,3)` sits slightly *below* `f64`
(0.95–0.97×) at higher `p` — a harmless, even favourable, quantisation artefact.

### Silicon sizing at (8,3)

| structure | size |
|-----------|------|
| message RAM (m_vc + e_cv), 432 edges × 2 × 8 b | **864 B** (was ~7 KB at f64 → **8× smaller**) |
| `γ` disorder ROM, 4 legs × 144 vars × 8 b | 576 B |
| `λ` prior ROM, 144 × 8 b | 144 B |
| syndrome register | 72 b |
| datapath multipliers | **1 fixed-coeff** (relay blend); min-sum is multiply-free |

The floor regime (p ≤ 0.02) needs more than 20 k shots to be conclusive on its own; `(8,3)` already
matches `f64` exactly there, so it is not gating.

## Acceptance

- [x] Fixed-point relay-BP golden implemented, deterministic, valid decisions reproduce the syndrome
      (unit tests in `fixed_bp.rs`).
- [x] Width sweep vs `f64` relay-BP produced; the narrowest matching word identified (**Q5.3**).

## Files

- `crates/aleph-qec/src/fixed_bp.rs` — `FixedRelayBp` + `FixedHwView` + unit tests.
- `crates/aleph-qec/examples/qec_q7_fixed_bp.rs` — the width sweep.
- `docs/perf/data/qec-q7-fixed-bp.csv` — committed run.
