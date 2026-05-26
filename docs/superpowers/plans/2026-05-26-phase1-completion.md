# Phase 1 completion plan

> **For future sessions:** This is the active phase plan after P1-03 merged (`f596e9a`, 2026-05-26). User decision recorded: ship the **full** Phase 1 backlog without skipping P1-05/06/07 — even ~10% wins per ticket are worth it. Strategic priority is to measure vs Qiskit Aer **first** before further SIMD work, but the measurement is informational (Phase 1 ships to completion regardless of how far ahead/behind we are).

**Status:** Active. P1-01, P1-02 (rolled back into P1-03), P1-03 (absorbs P1-04) merged on main. Remaining: 10 tickets (P1-05 through P1-14) + 1 new `[meta]` baseline task.

**ROADMAP §7 Phase 1 exit criterion:** "Single-thread time within 2× of Qiskit Aer for QFT, Grover, random circuits at 25 qubits." Currently unmeasured against Qiskit — Stage 0 below fixes that.

**Sticky context from ADR 0008 (read before any SIMD ticket):** The forensic finding that **layout choice dominates SIMD coverage on Zen 4** changes how P1-05/06/07 should be written. They were originally spec'd against the SoA backend; they now ship on AoS (which is where P1-03's win landed). Specs in BACKLOG are stale — each ticket needs a spec amendment before implementation.

---

## Stage 0 — Qiskit Aer baseline on EPYC

**Goal:** Concrete numbers for the ROADMAP §7 exit criterion. **Run before any further perf work** so we know whether we're 0.5× / 1× / 2× / 5× Qiskit on the canonical benchmarks. Numbers inform priority but **do not gate Phase 1 completion** — user explicitly chose to ship the full Phase-1 backlog regardless of where we stand vs Qiskit.

**Deliverable:** `docs/perf/phase1-vs-qiskit.md` with the table below populated, committed via a `[meta]` PR.

**Workload:**
- `qft/n20` single-thread (matches existing `crates/aleph-sv/benches/soa_vs_naive.rs`)
- `grover/n20` single-thread (need to add to benches if absent — check `benches/`)
- `random_circuit/n20` single-thread (need to construct a representative random circuit — see `benches/random.rs` if exists, else build per Google supremacy-style depth-20 random unitaries)

**Methodology:**
1. Install Qiskit Aer on EPYC (`pip install qiskit-aer`; pin version in `docs/perf/phase1-vs-qiskit.md`).
2. Write a small Python script: for each workload, construct the equivalent circuit in Qiskit, run with `AerSimulator(method='statevector')`, time 10 runs via `time.perf_counter`, report median.
3. Run aleph's existing benchmark binary with the same circuits via the OpenQASM parser to ensure circuit equivalence.
4. Document the side-by-side table with absolute times + ratios on the same EPYC machine, same RUSTFLAGS, same Python build.

**Acceptance:**
- All three workloads measured.
- Markdown doc in `docs/perf/phase1-vs-qiskit.md` with reproducible run instructions.
- One-paragraph interpretation: where we stand vs ROADMAP §7's "≤ 2×" target.
- Don't block on the verdict — proceed to Stage 1 regardless.

**Expected effort:** ~half a day. Mostly setup; the comparison itself is one script.

---

## Stage 1 — SIMD specialisations (P1-05, P1-06, P1-07, P1-08)

**Context after ADR 0008:** P1-03's win came from packed-complex AVX-512 on the **AoS** layout (`Vec<Complex>`). All subsequent SIMD work should follow that pattern: extend `kernels::aos::apply_*` rather than `kernels::soa::apply_*`. The SoA backend stays in tree for non-x86 hosts and for any future workload where its lower per-gate footprint pays off, but the canonical "fast x86 path" is AoS+AVX-512 from here on.

User accepted small wins (per-ticket ~10%): "якшо на кожном отримаєм 10 процентів це уже буде толково навіть дуже". The expected ROI per ticket is modest because the generic 2×2 kernel from P1-03 already covers all single-qubit gates. Specialisations skip arithmetic that the generic kernel can't (e.g., Pauli-X is a pure swap; diagonal gates touch only half the state).

### P1-05 — Specialised Pauli-X kernel

**Insight:** Pauli-X is `state[i] ↔ state[j]` — a pure swap, no multiplication. On AVX-512 + AoS, two `_mm512_loadu_pd` + two `_mm512_storeu_pd` (no `vfmaddsub`, no broadcast loads of m_re/m_im). Skips ~8 µops per inner iter vs the generic path. Expected: 1.2–1.5× over P1-03 generic on a Pauli-X heavy workload (random circuit has lots of these via decomp).

**Spec amendment required:** BACKLOG `[P1-05]` was written for SoA. Rewrite the design as a sibling helper to `kernels::aos::apply_1q_avx512` — an `apply_pauli_x_avx512` invoked from a `Gate::X` dispatch path. Or alternatively: detect `m == pauli_x()` inside `apply_1q_avx512` and branch to a swap loop. The dispatch-table approach is cleaner.

**Risk:** Y also swaps + multiplies; consider whether Y deserves the same treatment.

### P1-06 — Specialised diagonal-gate kernel

**Insight:** Diagonal gates (Z, S, T, Rz, Phase) have `m[0][1] == m[1][0] == 0`. Only `state[i] *= m[0][0]` and `state[j] *= m[1][1]`. Half the loads (one side at a time), no FMA needed. Expected: 1.5–2× over P1-03 generic on diagonal-heavy workloads.

**Spec amendment required:** Same pattern — sibling helper. Dispatch detects `Gate::Z`, `Gate::S`, `Gate::T`, `Gate::Sdg`, `Gate::Tdg`, `Gate::Rz`, `Gate::Phase` → diagonal path.

**Risk:** The `Phase` gate's matrix is parameter-dependent; ensure the diagonal-detection logic handles runtime parameters correctly.

### P1-07 — 2-qubit gate generic kernel + specialised CNOT/CZ/SWAP

**Insight:** This is the big one. Currently `kernels::aos::apply_2q` is scalar — no AVX-512. The packed-complex AVX-512 pattern from P1-03 extends naturally to 4×4 matrices: load 4 complex pairs at indices `{i, i|t1, i|t0, i|t_mask}`, do 4×4 matrix multiply via `vfmaddsub` (16 fmaddsub for the 16 entries; alternately, build a row-by-row accumulator).

**Specialised cases:**
- **CNOT** = swap with control: same as P1-05 Pauli-X but conditioned on the control bit. Pure swap of state[i_10] ↔ state[i_11] for indices where control=1.
- **CZ** = phase-flip with control: state[i_11] *= -1. Pure sign-flip on a fixed-mask subset.
- **SWAP** = unconditional swap of (i_01, i_10) pairs.

Expected: 2× on CNOT-heavy workloads (Bell states, GHZ chains, random circuits) — these go through apply_2q which has zero SIMD currently. Could be bigger than the 1.78× win we got on apply_1q.

**Spec amendment required:** BACKLOG `[P1-07]` was written assuming P1-01 SoA layout. Rewrite with AoS+AVX-512 substrate.

**Risk:** 4-stream load pattern (state[i_00], state[i_01], state[i_10], state[i_11]) might hit the same load-µop ceiling as SoA's 4-stream pattern. Validate on EPYC early via objdump + perf stat before going deep on intrinsics.

### P1-08 — Multi-controlled gates (Toffoli, CCZ, MCX)

**Insight:** Toffoli (CCX) acts on 8 amplitudes; CCZ acts on 4. Generic apply_3q currently in `kernels::aos.rs` is scalar. AVX-512 with 4 complex pairs per zmm could process the swap (for Toffoli) or sign-flip (for CCZ) with very few µops. But these gates are rarer in standard algorithms (QFT has none; Grover oracle uses one CCZ per depth round).

**Expected:** Modest (1.2–1.4×) but the kernel is small. Worth doing for completeness.

**Spec amendment required:** Same.

**Stage 1 collective deliverable:** All four SIMD specialisations land via individual PRs (`[P1-05]`, `[P1-06]`, `[P1-07]`, `[P1-08]`). After all four are in, re-run the Stage 0 Qiskit baseline to capture cumulative speedup.

---

## Stage 2 — IR-level optimisation passes (P1-09, P1-10, P1-11, P1-12, P1-13)

**Why after Stage 1, not before:** User chose to ship the full Phase-1 backlog; the SIMD tier work is in the BACKLOG and shouldn't be skipped. **However**, per CLAUDE.md hierarchy IR-opt is supposed to come BEFORE SIMD. The pragmatic ordering is: ship Stage 1 because the tickets are spec'd and well-scoped, then move to Stage 2 where the ROI per ticket is much higher (algorithmic, not micro-architectural).

**Why these are the actual game-changers:** Gate fusion can collapse a QFT-20 chain of ~210 gates down to ~40-60 fused gates by merging adjacent 1q rotations and absorbing controlled-phases into surrounding Hadamards. That's a ~3-5× win on state-vector sweep count alone, multiplicative with all the SIMD work.

### P1-09 — 1q gate fusion (foundational)

**Insight:** Adjacent 1q gates on the same qubit can be merged into a single matrix product. `Rz(a) · H · Rz(b)` on q0 becomes one matrix `M = Rz(a) · H · Rz(b)` applied as a single apply_1q. Reduces gate count and state-vector sweep count.

**Implementation lives in:** `crates/aleph-ir/` — a new optimisation pass `fuse_1q_runs` that walks the IR and replaces consecutive same-qubit single-qubit gates with one `Gate::U` (generic 2×2 unitary) carrying the product matrix.

**Testing:** Equivalence against unfused circuit via existing oracle harness. Property tests: fused circuit and unfused circuit produce same state vector within 1e-12.

### P1-10 — 2q gate fusion

**Insight:** Adjacent 2q gates on the same qubit pair, or a 2q gate sandwich'd between 1q gates on its target/control qubits — both can be fused into a single 4×4 matrix application.

**Risk:** Combinatorial complexity (`fuse_1q_into_2q_neighbours` has to handle several patterns). Keep the pass simple at first — only fuse adjacent 2q + adjacent 1q on the same pair.

### P1-12 — Gate cancellation (H·H, X·X, Rz(θ)·Rz(-θ))

**Insight:** Identity sequences appear after circuit transpilation. `H · H = I`, `X · X = I`, `Rz(θ) · Rz(-θ) = I`, `Rx(θ) · Rx(-θ) = I` etc. Drop them.

**Implementation:** Single-pass walk over IR, peephole-match adjacent gate pairs against identity patterns, delete both.

### P1-13 — Commutation analysis (foundational for better fusion)

**Insight:** `X · Y · X` on different qubits commute. Reordering can expose fusion opportunities that the naive forward walk misses. Build a commutation DAG, identify reorderings that expose adjacent same-qubit / same-pair gates, then run P1-09 / P1-10 over the reordered IR.

**Risk:** This is heavy. The simplest version (pairwise commutation table) is small; a full DAG-based optimiser is its own project. Aim for the simple version in Phase 1.

### P1-11 — Dead code elimination

**Insight:** Gates that don't affect measured qubits or that get cancelled out (via P1-12) can be removed.

**Implementation:** Reverse-walk from measurement nodes; mark live gates; drop the rest. Run after cancellation + fusion.

**Stage 2 collective deliverable:** A pipeline of IR passes — fuse 1q → fuse 2q → cancel → commute-then-fuse-again → DCE. Each pass tested for correctness via oracle harness. Cumulative effect on QFT-20 measured at the end.

**Risk for the whole stage:** ROADMAP §2 says "IR optimisation passes" plural — these aren't trivial. Each pass is its own design + spec + plan. Budget ~1 week per pass; the whole stage could be 3-5 weeks of full-time work.

---

## Stage 3 — Phase 1 closure (P1-14 + `[meta]` Phase 1 fixup)

**Mirrors the Phase 0 closure pattern** ([Phase 0 fixup] in MEMORY.md, PR #74).

### P1-14 — Phase 1 performance report

- Re-run all benchmarks (qft, grover, random, ghz at n10/15/20/25) on EPYC.
- Re-run Qiskit Aer baselines from Stage 0.
- Write `docs/perf/phase1.md` with the full table + analysis.
- Update `bencher.dev` baseline to current main.

### `[meta]` Phase 1 fixup

- Validate every ROADMAP §7 Phase-1 exit criterion: tick each one with the supporting bench number.
- Update BACKLOG with Phase-1 completion notes (mirror the Phase-0 pattern).
- ADRs as needed for any architectural decisions made during Stage 1-2.
- Update CLAUDE.md "Common Mistakes" if any new lessons emerge.
- Open question to settle: do we flip the default x86 backend to AoS (since P1-03 made it faster than SoA)? Decision in a separate ADR if yes.

**After Phase 1 closure:** Move to Phase 2 (multi-threading via `rayon`) or Phase 3 (Stabilizer / MPS backends — the real strategic differentiation per ROADMAP §2).

---

## Sequencing summary

```
Stage 0  →  Stage 1 (P1-05/06/07/08, in order)  →  Stage 2 (P1-09/10/12/13/11)  →  Stage 3 (P1-14 + meta)
~½ day      ~2-3 weeks                              ~3-5 weeks                      ~3-4 days
```

Each ticket is its own brainstorm → spec → plan → implementation → review → squash-merge cycle (per the established workflow from P0-06 onwards). Don't merge with red CI; don't merge without EPYC bench numbers for the perf-relevant tickets; don't skip the spec amendment for P1-05/06/07 (their original BACKLOG specs assumed SoA, which is now wrong).

## Open / unresolved questions

1. **Default backend on x86.** After P1-03, `kernels::aos` is faster than `kernels::soa` on EPYC for QFT. Backend selection currently isn't dispatched per-host — `NaiveSvBackend` and `SoaSvBackend` are siblings the user instantiates explicitly. Decide at Phase 1 closure whether to add automatic backend selection (or just flip the default).
2. **SoA backend's future.** If AoS dominates on x86, AoS dominates on Apple silicon by parity (M-series NEON auto-vec is close to AoS perf), and SoA never becomes the canonical choice, do we keep `SoaSvBackend` in tree? It still has educational/comparison value but is dead code in practice. Decide at Phase 1 closure.
3. **P1-13 scope.** Commutation analysis can be a 1-week peephole pass or a 1-month DAG-based optimiser. Pick the simpler version for Phase 1; defer the full thing to Phase 4+ if needed.
4. **AVX2-only path.** P1-03 dispatcher requires AVX-512F. Hosts with AVX2 but no AVX-512 (Intel pre-Skylake-X, AMD pre-Zen 4) hit the scalar path. Decide whether to add an AVX2 dispatcher branch (would benefit Intel laptops, older servers) or accept the AVX-512-or-scalar dichotomy.

These are not blockers — Stage 0 should kick off immediately, the questions get resolved at the relevant ticket.
