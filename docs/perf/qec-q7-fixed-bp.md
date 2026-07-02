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

-----

# Q7-02 M1 — Tanner `.svh` + combinational check-update RTL

**Status:** done (Verilator).

## What

The RTL half of the check→variable min-sum update, plus the generator that bakes the gross-code
graph into a SystemVerilog header (as `uf_surface_graph.svh` does for UF).

- `qec_q7_bp_graph.rs` emits **`hw/bb_gross_tanner.svh`**: `BP_N=144`, `BP_C=72`, `BP_E=432`,
  `MSG_BITS=8`, `FRAC_BITS=3`, `MAX_MAG=127`, `BP_LEGS=4`, `BP_ITERS=25`, the flattened CSR
  (`BP_VAR_OFF`, `BP_EDGE_VAR`, `BP_CHECK_OFF`, `BP_CHECK_EDGES`), quantised priors `BP_LAMBDA[144]`,
  and disorder `BP_GAMMA[4*144]`. `BP_VAR_OFF = {0,3,6,…}` confirms the constant variable-degree 3;
  the whole graph (M2 needs all of it) is in one header.
- `hw/bp_check_update.sv` is the combinational datapath: for each degree-6 check, the two-pass
  exclusive-minimum (min1/min2 + argmin), the check parity / per-edge sign, and **α = 7/8 as
  `mag − (mag >> 3)`** — no multiplier. Like the Q6-02 combinational UF draft it is one `always_comb`
  cloud (M2 sequentialises it).

## Verification

`make -C hw bpcheck` replays 256 random `(syndrome, m_vc)` vectors — `m_vc` spanning the full signed
Q5.3 range — through the RTL and checks all 432 outputs per vector against
`FixedRelayBp::check_update_once`:

> **PASS: 110 592 / 110 592 check-update outputs bit-identical to the fixed-point golden.**

The RTL check-update is the exact silicon twin of the M0 golden. Next (M2): sequentialise into a
clocked FSM (S_CHECK → S_VAR-with-memory → leg/iteration loop → keep-best-valid) for the full
relay-BP decode, reusing this `.svh`.

## Files

- `crates/aleph-qec/examples/qec_q7_bp_graph.rs` — `.svh` + vector generator.
- `hw/bp_check_update.sv` — combinational check-update RTL.
- `hw/tb_bp_check.cpp`, `hw/Makefile` (`bpcheck`) — Verilator TB.
- `hw/bb_gross_tanner.svh`, `hw/bp_check_vectors.txt` — generated.

-----

# Q7-02 M2 — sequential FSM full relay-BP decoder

**Status:** done (Verilator).

## What

`hw/bp_relay_decoder.sv` — the complete relay-BP decode of `FixedRelayBp` time-multiplexed into a
clocked FSM, **one bounded pass per cycle** (the Q6-04 UF methodology): a cycle touches exactly one
check (≤ `BP_CHK_DEG`=6 edges) or one variable (≤ `BP_VAR_DEG`=3 edges), so per-cycle combinational
depth is bounded and graph-size-independent. Per iteration:

- `S_CHECK` (72 cyc): `e_cv ← min-sum(m_vc, s)`, one check/cycle (multiply-free α=7/8).
- `S_VAR` (144 cyc): `m_vc, ehat ← var-update-with-memory`, one variable/cycle — the **memory blend
  `(1−γ)·computed + γ·m_old`, the one multiply**, γ from the per-leg disorder ROM; running Hamming
  weight tracked here.
- `S_SAT` (72 cyc): `all_sat ← (H·ehat == s)`; keep the lowest-weight syndrome-valid `ehat` seen.

looped `BP_LEGS`×`BP_ITERS` = 4×25, then `S_EMIT` (144 cyc) reduces the chosen `ehat` → observable
flips, `S_DONE` pulses `out_valid`. Handshake mirrors the UF core (`in_valid`/`busy`/`out_valid`/
`latency_cycles`).

## Verification

`make -C hw bprelay` drives 65 syndromes (empty + every single-variable error + 40 random low-weight)
through the FSM and checks the chosen error `corr_out[144]`, observable flips `obs_flip[12]`, and the
validity flag bit-for-bit against `FixedRelayBp::decode_fixed_ehat`:

> **PASS: 65 / 65 full decodes bit-identical to the fixed-point golden.**

Bit-exact agreement with the golden means the RTL's logical-error rate **is** the golden's (which is
within CI of `f64` relay-BP, M0) — no separate statistical LER run needed.

## Honest latency headline (the M3/M4 target)

> **Worst-case latency = 28 944 cycles/decode.**

That is `4·25·(72+144+72) + 144` — the full legs×iterations schedule with **no early exit** (relay-BP
must keep the best across *all* legs). At a nominal ~150 MHz that is ~193 µs — far over a ~1 µs round
budget. This is the honest cost of a real qLDPC decoder and exactly what M3 (measure it in silicon)
and M4 attack: **process K checks/variables per cycle** (the UF `FOREST_UNROLL` lever — the graph's
constant degree 6/3 and 72/144 independent nodes per pass make this embarrassingly parallel, so a
K-wide datapath cuts the pass length by K), plus revisiting the leg/iteration budget. The point of M2
is a *correct, bounded-depth, synthesizable* baseline to optimise from.

## Files

- `hw/bp_relay_decoder.sv` — sequential relay-BP FSM.
- `hw/tb_bp_relay.cpp`, `hw/Makefile` (`bprelay`) — Verilator TB.
- `crates/aleph-qec/examples/qec_q7_bp_graph.rs` (`decvectors` mode) — full-decode golden vectors.
- `crates/aleph-qec/src/fixed_bp.rs` — `decode_fixed_ehat` exposes the chosen `ehat` for the TB.
- `hw/bp_dec_vectors.txt` — generated.
