# [P2-09] Cache-Blocked Multi-Gate Application — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the N-DRAM-passes-per-gate-run into one pass by (A) a tile-major executor for runs of cache-confinable gates (`Instruction::TiledBlock`), and (B) a `RelabelQubits` pass that maps high-traffic qubits to low (cache-local) bit positions, made transparent by a single final index-remap.

**Architecture:** A `TileBlock` IR pass groups maximal runs of consecutive gates whose targets are all `< tile_bits` into `Instruction::TiledBlock`. `NaiveSvBackend` executes it tile-major: rayon over `2^tile_bits`-amplitude tiles, each gate applied to the hot tile sub-slice sequentially. A `RelabelQubits` pass permutes qubit indices for locality and records the permutation; `run_optimized` maps mid-circuit measure qubits through it and applies one final gather so the returned state is logical-order (oracle/measure/HasAmplitudes unchanged). The f64 hot path is the only one with a tiled executor; SoA/FP32 inherit a correct default-impl that replays gates.

**Tech Stack:** Rust 2021 (MSRV 1.89), rayon, criterion, proptest. Cache-miss validation via `perf stat` on the EPYC box (L2 = 1 MiB/core, L3 = 16 MiB/CCX, `perf` 7.0.0).

**Spec:** `docs/superpowers/specs/2026-06-04-p2-09-cache-blocking-design.md`

---

## Key codebase facts (verified)

- `Instruction` enum (`crates/aleph-ir/src/instruction.rs`): variants `Gate(GateInstance)`, `Measure {qubit, clbit}`, `Reset(u32)`, `Barrier(SmallVec<[u32;8]>)`, `DiagonalPhase(Box<DiagonalPhase>)`. Has `used_qubits()` / `used_clbits()` that match all variants exhaustively.
- `GateInstance { gate: Gate, qubits: SmallVec<[u32;4]>, controls: SmallVec<[u32;2]> }` (`crates/aleph-core/src/gate/instance.rs`). `gate.arity()`, `gate.matrix()`.
- `Circuit { num_qubits, num_clbits, instructions: Vec<Instruction>, metadata }` (`crates/aleph-ir/src/circuit.rs`). `optimize()` calls `PassPipeline::default_pipeline().run(self)`.
- `Pass` trait: `fn name(&self) -> &'static str`, `fn run(&self, &mut Circuit) -> Result<PassStats, PassError>`. Passes rebuild `circuit.instructions` (clone input, push to `out`, assign back) — see `fuse_diagonal.rs`.
- `default_pipeline()` = `[CancelInversePairs, DeadCodeElim, FuseDiagonalRuns, Fuse1qRuns, Fuse2q, FuseKq::default()]`.
- `Backend` trait (`crates/aleph-backend/src/lib.rs`): `allocate, apply_gate, measure, sample, expectation_value, probabilities, apply_diagonal_phase` (last has a default impl returning `UnsupportedInstruction`). The run loop is in `run_with_outcomes`.
- AoS scalar 1q generic body (`crates/aleph-sv/src/kernels/aos.rs:192-223`): for each `i` with `i & t_bit == 0 && (i & ctrl_mask) == ctrl_mask`, pair `{i, i|t_bit}` updated by the 2×2 matrix. 2q dense scalar at `aos.rs:1335`.
- Exhaustive `match self`/`match inst` over `Instruction` to update for a new variant: `instruction.rs` (`used_qubits`, `used_clbits`), `circuit.rs` (validation in `add_instruction`, any layer logic), `layers.rs`, `passes/*.rs` that match instructions (cancel, fuse_*), `aleph-backend/src/lib.rs` (`run_with_outcomes`), `aleph-parser/src/emit.rs` + `lower.rs`, `aleph-ir/src/error.rs` if it names instruction kinds. Strategy: add the variant, then `cargo build` and fix each non-exhaustive-match error the compiler points to.

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `crates/aleph-ir/src/tiled_block.rs` | `TiledBlock { gates: Vec<GateInstance>, tile_bits: u8 }` type | Create |
| `crates/aleph-ir/src/instruction.rs` | add `TiledBlock(Box<TiledBlock>)` variant + match arms | Modify |
| `crates/aleph-ir/src/passes/tile_block.rs` | `TileBlock` pass (group low-target runs) | Create |
| `crates/aleph-ir/src/passes/relabel.rs` | `RelabelQubits` pass (interaction → π, rewrite, record) | Create |
| `crates/aleph-ir/src/passes/mod.rs` | export + wire both passes into `default_pipeline` | Modify |
| `crates/aleph-ir/src/circuit.rs` | `qubit_permutation` field + accessor; new-variant match arms | Modify |
| `crates/aleph-ir/src/layers.rs` | new-variant match arm | Modify |
| `crates/aleph-ir/src/lib.rs` | export `TiledBlock`, `tiled_block` mod | Modify |
| `crates/aleph-sv/src/kernels/aos.rs` | `apply_1q_tile` / `apply_2q_tile` sequential per-tile helpers | Modify |
| `crates/aleph-sv/src/kernels/tuning.rs` | `tile_bits()` policy (CPU-model) | Modify |
| `crates/aleph-sv/src/backend.rs` | `NaiveSvBackend::apply_tiled_block` tiled executor | Modify |
| `crates/aleph-backend/src/lib.rs` | `apply_tiled_block` trait method (default replay) + run-loop dispatch + π handling + final gather | Modify |
| `crates/aleph-parser/src/{emit.rs,lower.rs}` | reject/ignore `TiledBlock` (pass-emitted only) | Modify |
| `crates/aleph-sv/src/perm.rs` | `bit_permute_state` (final gather) helper | Create |
| `benches/benches/cache_blocking.rs` | low-qubit-heavy + counter-case bench | Create |
| `docs/perf/phase2.md` | §11 P2-09 results | Modify |

---

## Phase A — Tile-major executor (the core driver)

### Task 1: `TiledBlock` IR type + `Instruction` variant

**Files:** Create `crates/aleph-ir/src/tiled_block.rs`; modify `instruction.rs`, `lib.rs`.

- [ ] **Step 1: Create the type.** `crates/aleph-ir/src/tiled_block.rs`:

```rust
//! `TiledBlock` — a maximal run of consecutive gates whose targets are all
//! below a cache-tile bit width, grouped so a backend can apply them
//! **tile-major** (one DRAM pass over the state instead of one per gate).
//! Produced only by `passes::TileBlock`; never by the parser. See
//! `docs/superpowers/specs/2026-06-04-p2-09-cache-blocking-design.md`.

use aleph_core::GateInstance;

/// A run of gates confined to the low `tile_bits` qubits (targets only;
/// controls may be higher and are masked per-tile by the executor).
#[derive(Debug, Clone)]
pub struct TiledBlock {
    /// Gates in original application order. Each gate's `qubits` (targets)
    /// are all `< tile_bits`; `controls` may be any qubit.
    pub gates: Vec<GateInstance>,
    /// log2 of the tile size in amplitudes. A tile is `2^tile_bits`
    /// contiguous amplitudes; targets `< tile_bits` pair within a tile.
    pub tile_bits: u8,
}

impl TiledBlock {
    /// All qubits touched by any gate in the block (targets ∪ controls).
    pub fn used_qubits(&self) -> smallvec::SmallVec<[u32; 6]> {
        let mut out: smallvec::SmallVec<[u32; 6]> = smallvec::SmallVec::new();
        for g in &self.gates {
            for &q in g.qubits.iter().chain(g.controls.iter()) {
                if !out.contains(&q) {
                    out.push(q);
                }
            }
        }
        out
    }
}
```

- [ ] **Step 2: Add the variant** in `instruction.rs`, after `DiagonalPhase`:

```rust
    /// A cache-tile-confinable run of gates, produced by
    /// `passes::TileBlock`. Never produced by the parser; only exists
    /// post-optimization. Boxed to keep the enum small.
    TiledBlock(Box<crate::tiled_block::TiledBlock>),
```

- [ ] **Step 3: Extend `used_qubits`/`used_clbits` match arms** in `instruction.rs`:

```rust
            // in used_qubits():
            Instruction::TiledBlock(tb) => out.extend(tb.used_qubits().iter().copied()),
```
`used_clbits` only matches `Measure`, so the new variant needs no arm there if that match uses `if let`; if it is a `match`, add `Instruction::TiledBlock(_) => {}`. (Read the function; it currently uses `if let Instruction::Measure`, so no change needed.)

- [ ] **Step 4: Export** in `crates/aleph-ir/src/lib.rs`: `mod tiled_block; pub use tiled_block::TiledBlock;`.

- [ ] **Step 5: Build and fix exhaustive matches.** Run `cargo build -p aleph-ir`. For every non-exhaustive-match error (in `circuit.rs`, `layers.rs`, `passes/*.rs`), add a `Instruction::TiledBlock(_) => { ... }` arm. For passes that walk instructions (cancel, fuse_*), the correct arm is to treat `TiledBlock` as an **opaque run-breaker / fence** (flush any in-progress run, emit verbatim) — these passes run BEFORE `TileBlock` in the pipeline so they never actually see one, but the match must compile. For `layers.rs`, treat its qubits (`tb.used_qubits()`) as occupying a layer like any multi-qubit op. Document each arm with a one-line comment.

- [ ] **Step 6: Add a unit test** in `tiled_block.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use aleph_core::Gate;
    use smallvec::smallvec;

    #[test]
    fn used_qubits_unions_targets_and_controls() {
        let tb = TiledBlock {
            gates: vec![
                GateInstance::new(Gate::H, smallvec![0u32]),
                GateInstance::controlled(Gate::X, smallvec![1u32], smallvec![5u32]),
            ],
            tile_bits: 4,
        };
        let mut q = tb.used_qubits().to_vec();
        q.sort();
        assert_eq!(q, vec![0, 1, 5]);
    }
}
```

- [ ] **Step 7:** `cargo test -p aleph-ir tiled_block` PASS; `cargo build --workspace` clean (all match sites fixed). Commit `[P2-09] Instruction::TiledBlock IR type + exhaustive-match arms`.

---

### Task 2: sequential per-tile kernel helpers

**Files:** Modify `crates/aleph-sv/src/kernels/aos.rs`.

The tiled executor parallelizes over tiles (outer rayon); within a tile it must apply gates **sequentially** (no nested rayon). Add sequential variants that mirror the scalar kernel bodies without `par_blocks`.

- [ ] **Step 1: Add `apply_1q_tile`** in `aos.rs` (mirror the generic scalar body at lines 192-223, sequential):

```rust
/// Sequential 1q application to a contiguous amplitude slice (one cache
/// tile). No `par_blocks` — the caller (`apply_tiled_block`) is already
/// rayon-parallel over tiles, so the inner walk must stay sequential to
/// avoid nested parallelism. `target` and every control bit must be
/// `< log2(slice.len())` OR the control is pre-masked away by the caller
/// (see `apply_tiled_block`): this kernel only consults `target` and the
/// controls it is given, all interpreted relative to the slice.
pub(crate) fn apply_1q_tile(
    slice: &mut [Complex],
    target: u32,
    controls: &[u32],
    m: &[[Complex; 2]; 2],
) {
    let t_bit = 1usize << target;
    let ctrl_mask = super::control_mask(controls);
    let len = slice.len();
    let mut i = 0;
    while i < len {
        if i & t_bit == 0 && (i & ctrl_mask) == ctrl_mask {
            let j = i | t_bit;
            let a = slice[i];
            let b = slice[j];
            slice[i] = m[0][0] * a + m[0][1] * b;
            slice[j] = m[1][0] * a + m[1][1] * b;
        }
        i += 1;
    }
}
```

- [ ] **Step 2: Add `apply_2q_tile`** mirroring `apply_2q_dense_scalar` (aos.rs:1335) sequentially. READ that function; reproduce its quartet index construction (`idx = [i, i|t1_bit, i|t0_bit, i|t_mask]`, MSB convention `targets[0]` = high) in a plain `while i < len` loop with the `(i & t_mask) == 0 && (i & ctrl_mask) == ctrl_mask` guard. Signature:
```rust
pub(crate) fn apply_2q_tile(slice: &mut [Complex], targets: [u32; 2], controls: &[u32], m: &[[Complex; 4]; 4])
```

- [ ] **Step 3: Test** `apply_*_tile` ≡ whole-state kernel on a full-size slice (when the slice IS the whole state and all qubits `< n`, the tile helper must equal `apply_1q`/`apply_2q`). In `aos.rs` tests:

```rust
    #[test]
    fn tile_1q_matches_whole_state_kernel() {
        let n = 6u32;
        let base: Vec<Complex> = (0..(1usize << n))
            .map(|k| Complex::new(k as f64 * 0.01, (k as f64 * 0.013).cos()))
            .collect();
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let m = [[Complex::new(h, 0.0), Complex::new(h, 0.0)],
                 [Complex::new(h, 0.0), Complex::new(-h, 0.0)]];
        for target in 0..n {
            let mut a = base.clone();
            let mut b = base.clone();
            apply_1q(&mut a, target, &[], &m);
            apply_1q_tile(&mut b, target, &[], &m);
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).norm() < 1e-12);
            }
        }
    }
```
Add the analogous `tile_2q_matches_whole_state_kernel` (random 4×4, all target pairs, ± a control).

- [ ] **Step 4:** `cargo test -p aleph-sv aos` PASS; clippy clean. Commit `[P2-09] sequential per-tile 1q/2q kernel helpers`.

---

### Task 3: `Backend::apply_tiled_block` trait method + run-loop dispatch (default replay)

**Files:** Modify `crates/aleph-backend/src/lib.rs`.

- [ ] **Step 1: Add the trait method** with a default impl that replays gates (so SoA/FP32/MPS stay correct without overriding):

```rust
    /// Apply a cache-tile-confinable run (`Instruction::TiledBlock`).
    ///
    /// Default implementation replays each gate via `apply_gate` in order
    /// — semantically identical to executing the gates individually, just
    /// without the tile-major cache benefit. State-vector backends with a
    /// tiled fast path override this; others (SoA, FP32, MPS) inherit the
    /// correct replay.
    fn apply_tiled_block(
        &mut self,
        state: &mut Self::State,
        block: &aleph_ir::TiledBlock,
    ) -> Result<(), BackendError> {
        for gate in &block.gates {
            self.apply_gate(state, gate)?;
        }
        Ok(())
    }
```

- [ ] **Step 2: Dispatch in the run loop.** In `run_with_outcomes`, add to the `match inst`:

```rust
            aleph_ir::Instruction::TiledBlock(tb) => {
                backend.apply_tiled_block(&mut state, tb)?;
            }
```

- [ ] **Step 3: Test** that the default replay equals per-gate application. In `aleph-backend` tests, build a small circuit, wrap two gates in a `TiledBlock`, run on the existing mock/`NaiveSvBackend`, compare to running the same two gates as plain `Gate`s. (Use `NaiveSvBackend` via a dev-dependency if available; the crate's tests already reference it — check. If not, the `MockBackend` in the test module suffices to prove replay calls `apply_gate` twice.)

- [ ] **Step 4:** `cargo test -p aleph-backend` PASS; `cargo build --workspace`. Commit `[P2-09] Backend::apply_tiled_block trait method + run-loop dispatch (default replay)`.

---

### Task 4: `NaiveSvBackend` tile-major executor

**Files:** Modify `crates/aleph-sv/src/backend.rs`.

- [ ] **Step 1: Implement `apply_tiled_block`** on `NaiveSvBackend`, overriding the default. Rayon over tiles; per tile, per gate, mask `≥ tile_bits` controls against the tile index and apply confined controls via the tile helpers:

```rust
    fn apply_tiled_block(
        &mut self,
        state: &mut Self::State,
        block: &aleph_ir::TiledBlock,
    ) -> Result<(), BackendError> {
        use rayon::prelude::*;
        let t = block.tile_bits as usize;
        let tile_len = 1usize << t;
        let n = state.num_qubits as usize;
        // Degenerate: tile covers the whole state (or more) → one tile,
        // sequential, equivalent to gate-major.
        if t >= n {
            for g in &block.gates {
                // No high controls exist (all qubits < n ≤ t); apply directly.
                apply_one_to_tile(&mut state.amps, g, t);
            }
            return Ok(());
        }
        let high_mask: usize = !((1usize << t) - 1); // bits ≥ t
        // SAFETY/correctness: tiles are disjoint contiguous sub-slices, so
        // par_chunks_mut over tile_len gives each rayon task an exclusive
        // tile. No cross-tile dependency (every gate target < t pairs within
        // the tile); bit-identical to sequential regardless of thread count.
        state
            .amps
            .par_chunks_mut(tile_len)
            .enumerate()
            .for_each(|(tile_idx, tile)| {
                let tile_base = (tile_idx << t) & high_mask; // high bits of this tile
                for g in &block.gates {
                    apply_gate_to_tile(tile, g, t, tile_base);
                }
            });
        Ok(())
    }
```

- [ ] **Step 2: Add the helper functions** in `backend.rs` (free fns or a small module). `apply_gate_to_tile` splits controls into `< t` (passed to the tile kernel, interpreted relative to the tile) and `≥ t` (tested against `tile_base`; if any is 0 the gate is skipped for this tile). `apply_one_to_tile` is the `t >= n` degenerate case (no high controls). Both dispatch on `g.gate` arity → `apply_1q_tile` / `apply_2q_tile`:

```rust
/// Apply one gate to a single tile sub-slice. `tile_base` holds the high
/// (≥ t) bits common to all amplitudes in this tile. Controls ≥ t are
/// constant across the tile: if any is 0 the gate does not fire here.
fn apply_gate_to_tile(tile: &mut [Complex], g: &GateInstance, t: usize, tile_base: usize) {
    // High controls gate the whole tile.
    for &c in &g.controls {
        if (c as usize) >= t {
            let bit = 1usize << c;
            if tile_base & bit == 0 {
                return; // control not satisfied for this tile → skip
            }
        }
    }
    let low_controls: smallvec::SmallVec<[u32; 2]> =
        g.controls.iter().copied().filter(|&c| (c as usize) < t).collect();
    dispatch_tile_kernel(tile, g, &low_controls);
}

fn apply_one_to_tile(tile: &mut [Complex], g: &GateInstance, _t: usize) {
    dispatch_tile_kernel(tile, g, &g.controls);
}

fn dispatch_tile_kernel(tile: &mut [Complex], g: &GateInstance, controls: &[u32]) {
    // TiledBlock only contains gates with a fixed-size matrix and targets < t
    // (the TileBlock pass guarantees this). Materialise the matrix and route
    // by arity to the sequential tile kernels.
    match g.gate.matrix().expect("TiledBlock gate has a representable matrix") {
        GateMatrix::M2x2(m) => crate::kernels::aos::apply_1q_tile(tile, g.qubits[0], controls, &m),
        GateMatrix::M4x4(m) => crate::kernels::aos::apply_2q_tile(tile, [g.qubits[0], g.qubits[1]], controls, &m),
        GateMatrix::M8x8(_) => unreachable!("TileBlock pass never groups 3q gates (Task 5)"),
    }
}
```
NOTE on `.expect`: the `TileBlock` pass (Task 5) only ever groups gates whose `matrix()` is representable and arity ≤ 2; document this invariant. If you prefer no `expect` in library code, return `BackendError::UnsupportedInstruction` instead and thread the `Result` — but since the pass guarantees the invariant, an `expect` with a clear message is acceptable here (mirrors how kernels assume validated input). **Decide and document.** (Recommended: thread `Result` to avoid any panic path.)

- [ ] **Step 3: bit-exact test — tile-major ≡ gate-major.** In `backend.rs` tests: build a state, an `Instruction::TiledBlock` with several 1q+2q gates on low targets (some with a high control), apply via `apply_tiled_block`; separately apply the same gates gate-by-gate via `apply_gate` on a clone; assert **exact** equality (`==` on every amplitude, no tolerance — same ops, no FP reorder). Cover `tile_bits` values: `< n` (multiple tiles), `== n` and `> n` (degenerate single tile), and a gate with a control `≥ tile_bits` (must fire only in tiles where the control bit is set).

- [ ] **Step 4: thread-invariance test.** Same `TiledBlock` under `RAYON_NUM_THREADS=1` vs default gives identical amplitudes (set via `rayon::ThreadPoolBuilder` scoped, or assert determinism by running twice).

- [ ] **Step 5:** `cargo test -p aleph-sv` PASS; clippy clean. Commit `[P2-09] NaiveSvBackend tile-major executor`.

---

### Task 5: `TileBlock` pass + tile policy

**Files:** Create `crates/aleph-ir/src/passes/tile_block.rs`; modify `passes/mod.rs`, `crates/aleph-sv/src/kernels/tuning.rs`.

- [ ] **Step 1: Tile policy.** In `tuning.rs`, add:
```rust
/// log2 of the cache-tile size in amplitudes for the L2-resident tiled
/// executor (P2-09). EPYC 8124P L2 = 1 MiB/core = 2^16 Complex<f64>; a
/// 2^15-amp tile (512 KiB) leaves working-set headroom. Conservative
/// default for non-detected CPUs.
pub(crate) fn tile_bits() -> u8 {
    // CPU-model dispatch could refine this (cf. ChunkPolicy); default 15.
    15
}
```
(`aleph-ir` can't depend on `aleph-sv`. So the *policy value* must be available to the pass without that dependency. Decision: the `TileBlock` pass takes `tile_bits` as a constructor parameter; the **backend/driver** chooses it from `tuning::tile_bits()` when building the pipeline. Since `default_pipeline()` lives in `aleph-ir` and can't call `aleph-sv`, define a plain `const DEFAULT_TILE_BITS: u8 = 15;` in `aleph-ir` for `default_pipeline`, and document that the backend may rebuild the pipeline with a tuned value. Keep it simple: `TileBlock::new(tile_bits)` + `TileBlock::default()` using the const.)

- [ ] **Step 2: The pass.** `crates/aleph-ir/src/passes/tile_block.rs`:

```rust
//! `TileBlock` — groups maximal runs of consecutive gates whose targets are
//! all `< tile_bits` into one `Instruction::TiledBlock`, so the backend can
//! apply them tile-major (one DRAM pass per run). Runs LAST in the pipeline,
//! after all fusion. See the P2-09 design spec.

use crate::passes::{Pass, PassError, PassStats};
use crate::{Circuit, Instruction, TiledBlock};
use aleph_core::GateInstance;

/// Default tile width (log2 amplitudes). EPYC L2-sized; see spec.
pub const DEFAULT_TILE_BITS: u8 = 15;

pub struct TileBlock {
    pub tile_bits: u8,
}

impl Default for TileBlock {
    fn default() -> Self {
        Self { tile_bits: DEFAULT_TILE_BITS }
    }
}

impl TileBlock {
    /// A gate is tile-confinable iff all its TARGETS are `< tile_bits`
    /// (controls may be higher — the executor masks them per tile) AND it
    /// has a fixed-size matrix of arity ≤ 2 (the tile kernels handle 1q/2q).
    fn confinable(&self, g: &GateInstance) -> bool {
        let tb = self.tile_bits as u32;
        g.qubits.len() <= 2
            && g.qubits.iter().all(|&q| q < tb)
            && g.gate.matrix().is_ok()
    }
}

impl Pass for TileBlock {
    fn name(&self) -> &'static str {
        "TileBlock"
    }

    fn run(&self, circuit: &mut Circuit) -> Result<PassStats, PassError> {
        let input = circuit.instructions.clone();
        let gates_before = input.len();
        let mut out: Vec<Instruction> = Vec::with_capacity(input.len());
        let mut run: Vec<GateInstance> = Vec::new();
        let mut blocks = 0u64;

        let flush = |run: &mut Vec<GateInstance>, out: &mut Vec<Instruction>, blocks: &mut u64, tile_bits: u8| {
            match run.len() {
                0 => {}
                1 => out.push(Instruction::Gate(run.remove(0))),
                _ => {
                    *blocks += 1;
                    out.push(Instruction::TiledBlock(Box::new(TiledBlock {
                        gates: std::mem::take(run),
                        tile_bits,
                    })));
                }
            }
            run.clear();
        };

        for inst in input {
            match inst {
                Instruction::Gate(g) if self.confinable(&g) => run.push(g),
                other => {
                    flush(&mut run, &mut out, &mut blocks, self.tile_bits);
                    out.push(other);
                }
            }
        }
        flush(&mut run, &mut out, &mut blocks, self.tile_bits);

        circuit.instructions = out;
        Ok(PassStats {
            gates_before,
            gates_after: circuit.instructions.len(),
            transformations: blocks,
        })
    }
}
```

- [ ] **Step 3: Wire into the pipeline.** In `passes/mod.rs`: `pub mod tile_block;`, `pub use tile_block::TileBlock;`, and append `Box::new(TileBlock::default())` as the LAST entry of `default_pipeline()`. Update the `default_pipeline` doc comment to mention TileBlock runs last.

- [ ] **Step 4: Unit tests** in `tile_block.rs`: (a) a run of 3 low-target 1q gates → one `TiledBlock` of 3; (b) a high-target gate splits two runs; (c) a `Measure`/`Barrier`/`DiagonalPhase` splits runs and is preserved; (d) a length-1 run stays a plain `Gate`; (e) a gate with a low target but high CONTROL is still confinable (goes in the block). Assert the emitted instruction shapes.

- [ ] **Step 5:** `cargo test -p aleph-ir tile_block` PASS; clippy clean. Commit `[P2-09] TileBlock pass + tile policy, wired last in default_pipeline`.

---

### Task 6: end-to-end oracle (TileBlock active, no relabel yet)

**Files:** Create `crates/aleph-sv/tests/tiled_oracle.rs` (or extend `run_optimized_oracle`).

- [ ] **Step 1: Oracle test.** For Tier-1 fixtures (GHZ/QFT/Grover/random at n ∈ {6,8,10,12}) — note `tile_bits` default is 15 > these n, so the executor hits the **degenerate single-tile** path; to exercise **multi-tile**, also construct each fixture with an explicit small `tile_bits` (e.g. build the circuit, run `TileBlock::new(4)` manually, then `run`) and compare against raw `run`. Assert amplitudes within **1e-12** (relabel not involved yet, so it's exact-ish; tolerance 1e-12 per AC).

```rust
// Pseudostructure — implementer fills in fixture builders (reuse fp32_equiv.rs
// builders or aleph_benches):
// for n in [6,8,10,12]:
//   let c = qft_circuit(n);  // and ghz/grover/random
//   let raw = run(&mut NaiveSvBackend::with_seed(0), &c)?;        // reference
//   let mut opt = c.clone(); opt.optimize()?;                     // pipeline incl. TileBlock(15)
//   let tiled = run(&mut NaiveSvBackend::with_seed(0), &opt)?;
//   assert_close(raw, tiled, 1e-12);
//   // force multi-tile:
//   let mut small = c.clone();
//   TileBlock::new(4).run(&mut small)?;  // only the tiling pass, tile_bits=4
//   let tiled4 = run(&mut NaiveSvBackend::with_seed(0), &small)?;
//   assert_close(raw, tiled4, 1e-12);
```

- [ ] **Step 2:** `cargo test -p aleph-sv --test tiled_oracle` PASS. **If any mismatch, the executor or pass has a bug — fix it, do not loosen 1e-12.** Then `cargo test --workspace` green.

- [ ] **Step 3:** Commit `[P2-09] end-to-end tiled oracle (TileBlock) @ 1e-12`. **Milestone: tile-major driver correct.**

---

## Phase B — RelabelQubits + permutation tracking

### Task 7: `Circuit.qubit_permutation` + bit-permute helper

**Files:** Modify `crates/aleph-ir/src/circuit.rs`; create `crates/aleph-sv/src/perm.rs`.

- [ ] **Step 1: Add the field** to `Circuit`: `pub(crate) qubit_permutation: Option<Box<[u32]>>` (None = identity; `π[logical] = physical` bit position). Initialize `None` in every constructor. Add accessor:
```rust
    /// The qubit permutation applied by `RelabelQubits` (`π[logical] =
    /// physical` bit). `None` = identity. Set by the pass; consumed by the
    /// run driver to un-permute results.
    pub fn qubit_permutation(&self) -> Option<&[u32]> {
        self.qubit_permutation.as_deref()
    }
    pub(crate) fn set_qubit_permutation(&mut self, perm: Box<[u32]>) {
        self.qubit_permutation = Some(perm);
    }
```
Ensure `clone()` carries it (derive already does). Passes that rebuild `instructions` must NOT clobber it.

- [ ] **Step 2: bit-permute helper.** `crates/aleph-sv/src/perm.rs`:
```rust
//! Final physical→logical state reorder for the P2-09 relabelling pass.
//! Given `perm` where `perm[logical] = physical` bit position, produce a
//! logical-order amplitude vector from a physical-order one: the basis
//! index `i` (logical) gathers from physical index `j` where bit `logical_q`
//! of `i` becomes bit `perm[logical_q]` of `j`.

use aleph_core::{AlignedBuf, Complex};

/// Remap `phys` (physical-bit order) into logical order per `perm`.
/// `perm.len() == num_qubits`; `perm` is a permutation of `0..num_qubits`.
pub(crate) fn bit_permute_state(phys: &[Complex], perm: &[u32]) -> AlignedBuf<Complex> {
    let n = perm.len();
    debug_assert_eq!(phys.len(), 1usize << n);
    let mut out = AlignedBuf::<Complex>::zeroed(phys.len());
    for i in 0..phys.len() {
        // i is the logical basis index; build the physical index j.
        let mut j = 0usize;
        for (lq, &pq) in perm.iter().enumerate() {
            if (i >> lq) & 1 == 1 {
                j |= 1usize << pq;
            }
        }
        out[i] = phys[j];
    }
    out
}
```

- [ ] **Step 3: Tests.** Identity perm → unchanged. A single swap perm (swap bits 0 and 2 on n=3) → matches a manual SWAP-gate application. Round-trip: applying `bit_permute_state` with `perm` then with `perm⁻¹` returns the original.

- [ ] **Step 4:** `cargo test -p aleph-sv perm` + `cargo test -p aleph-ir circuit` PASS. Commit `[P2-09] Circuit.qubit_permutation + bit_permute_state helper`.

---

### Task 8: `RelabelQubits` pass

**Files:** Create `crates/aleph-ir/src/passes/relabel.rs`; modify `passes/mod.rs`.

- [ ] **Step 1: The pass.** Build a per-qubit traffic score (count of gate appearances; optionally weight 2q co-occurrence), then assign the highest-traffic qubits to the lowest bit positions. Emit `π[logical] = physical`. Rewrite every instruction's qubit indices through the INVERSE map (a gate currently on logical qubit `q` moves to physical position `π[q]`). Record `π` on the circuit. Guard: only commit if the relabel raises the count of tile-confinable gates (targets `< tile_bits`) by a margin; else leave identity.

```rust
//! `RelabelQubits` — permute qubit indices so high-traffic qubits occupy
//! low (cache-local) bit positions, maximizing the gates the TileBlock pass
//! can confine. Records the permutation on the circuit; the run driver
//! un-permutes results. Runs FIRST in the pipeline. See the P2-09 spec.

use crate::passes::{Pass, PassError, PassStats};
use crate::{Circuit, Instruction};

pub struct RelabelQubits {
    pub tile_bits: u8,
}
impl Default for RelabelQubits {
    fn default() -> Self { Self { tile_bits: crate::passes::tile_block::DEFAULT_TILE_BITS } }
}

impl Pass for RelabelQubits {
    fn name(&self) -> &'static str { "RelabelQubits" }

    fn run(&self, circuit: &mut Circuit) -> Result<PassStats, PassError> {
        let n = circuit.num_qubits as usize;
        if n <= 1 || circuit.qubit_permutation.is_some() {
            return Ok(PassStats { gates_before: circuit.instructions.len(),
                                  gates_after: circuit.instructions.len(), transformations: 0 });
        }
        // 1. Traffic score per logical qubit.
        let mut score = vec![0u64; n];
        for inst in &circuit.instructions {
            for q in inst.used_qubits() { score[q as usize] += 1; }
        }
        // 2. physical_of[logical]: highest score → lowest physical bit.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&q| std::cmp::Reverse(score[q])); // most-active first
        let mut physical_of = vec![0u32; n];      // π[logical] = physical
        for (phys_bit, &logical) in order.iter().enumerate() {
            physical_of[logical] = phys_bit as u32;
        }
        // 3. Net-win guard: count confinable targets before vs after.
        let before = count_confinable(circuit, self.tile_bits, None);
        let after = count_confinable(circuit, self.tile_bits, Some(&physical_of));
        if after <= before {
            return Ok(PassStats { gates_before: circuit.instructions.len(),
                                  gates_after: circuit.instructions.len(), transformations: 0 });
        }
        // 4. Rewrite all instruction qubit indices through physical_of.
        for inst in &mut circuit.instructions {
            remap_instruction(inst, &physical_of);
        }
        circuit.set_qubit_permutation(physical_of.into_boxed_slice());
        Ok(PassStats { gates_before: circuit.instructions.len(),
                       gates_after: circuit.instructions.len(), transformations: 1 })
    }
}
```
Implement helpers `count_confinable(circuit, tile_bits, Option<&[u32]>)` (counts gates whose targets, optionally remapped, are all `< tile_bits`) and `remap_instruction(&mut Instruction, &[u32])` (rewrites `Gate.qubits`/`Gate.controls`, `Measure.qubit`, `Reset`, `Barrier`, `DiagonalPhase` conds, and `TiledBlock` — though TiledBlock won't exist yet at relabel time since relabel runs first). Map each qubit `q` → `physical_of[q]`.

- [ ] **Step 2: Wire FIRST** in `default_pipeline()`: prepend `Box::new(RelabelQubits::default())`.

- [ ] **Step 3: Unit tests.** (a) A circuit using only high qubits (e.g. all gates on qubits n-1, n-2 of a 6-qubit circuit) → relabel maps them to bits 0,1 and records π; verify gate qubits rewritten and π set. (b) An already-optimal low-qubit circuit → guard leaves π = None (no-op). (c) `remap_instruction` correctly rewrites a Measure qubit and a controlled gate.

- [ ] **Step 4:** `cargo test -p aleph-ir relabel` PASS; clippy clean. **Note:** this task makes `default_pipeline` relabel, but the run driver (Task 9) doesn't yet un-permute → the end-to-end oracle WILL break until Task 9. So between Task 8 and 9, run only the pass-unit tests, not the full oracle. Commit `[P2-09] RelabelQubits pass (records π), wired first`.

---

### Task 9: run driver — measure-qubit mapping + final un-permute

**Files:** Modify `crates/aleph-backend/src/lib.rs`.

- [ ] **Step 1: Thread π through `run_optimized_with_outcomes`.** After `optimized.optimize()?`, read `optimized.qubit_permutation()`. The physical state comes back from `run_with_outcomes`, but measure qubits inside must be physical and outcomes reported logical, and the final state must be un-permuted. Cleanest: do NOT reuse `run_with_outcomes` verbatim; inline a permutation-aware loop OR (simpler) keep `run_with_outcomes` but (a) the circuit's Measure qubits are ALREADY physical (relabel rewrote them), so mid-circuit measure already targets the right physical qubit — but the returned `MeasurementRecord.qubit` would be physical, not logical. Map it back. And (b) after the run, gather the state to logical order.

**Design problem to resolve first:** the final state gather needs the concrete amplitude buffer, but `run_optimized` is generic over `Backend` and `B::State` is opaque, and `aleph-backend` cannot depend on `aleph-sv`. So the un-permute cannot be done directly in the generic driver. The resolution (Step 2) is a `Backend` trait hook the backend implements; the driver only orchestrates.

- [ ] **Step 2: Add the `unpermute_state` trait hook + orchestrate in the driver.** Add to the `Backend` trait a method with a default impl that **errors** (so relabelling is only honored by backends that can un-permute):
```rust
    /// Reorder a physical-bit-order state into logical order per `perm`
    /// (`perm[logical] = physical`), undoing a `RelabelQubits` permutation.
    /// Default errors; state-vector backends override. Called by
    /// `run_optimized_with_outcomes` exactly once, only when the optimized
    /// circuit carries a permutation.
    fn unpermute_state(
        &mut self,
        _state: &mut Self::State,
        _perm: &[u32],
    ) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedInstruction { kind: "unpermute_state" })
    }
```
Then rewrite `run_optimized_with_outcomes` to orchestrate (no `aleph-sv` dependency, no extra trait bound):
```rust
pub fn run_optimized_with_outcomes<B: Backend>(
    backend: &mut B,
    circuit: &Circuit,
) -> Result<(B::State, Vec<MeasurementRecord>), BackendError> {
    let mut optimized = circuit.clone();
    optimized.optimize()?;
    let perm = optimized.qubit_permutation().map(|p| p.to_vec());
    let (mut state, mut outcomes) = run_with_outcomes(backend, &optimized)?;
    if let Some(perm) = perm {
        // Measure qubits were rewritten to physical by RelabelQubits; report
        // them logical. logical_of[physical] = logical.
        let logical_of = invert_perm(&perm);
        for rec in &mut outcomes {
            rec.qubit = logical_of[rec.qubit as usize];
        }
        // Single final gather: physical-order state → logical order.
        backend.unpermute_state(&mut state, &perm)?;
    }
    Ok((state, outcomes))
}

/// `inv[perm[l]] = l` — invert a qubit permutation.
fn invert_perm(perm: &[u32]) -> Vec<u32> {
    let mut inv = vec![0u32; perm.len()];
    for (logical, &physical) in perm.iter().enumerate() {
        inv[physical as usize] = logical as u32;
    }
    inv
}
```
SoA/FP32 backends don't override `unpermute_state`, so a relabelled circuit reaching them via `run_optimized` surfaces a clear `UnsupportedInstruction` error rather than a silent wrong answer — acceptable for now (document it; a follow-up adds their hook). `NaiveSvBackend` overrides it (Step 3).

- [ ] **Step 3: Add `unpermute_state` to the `Backend` trait** (default errors) and override in `NaiveSvBackend`:
```rust
    fn unpermute_state(&mut self, state: &mut CpuState, perm: &[u32]) -> Result<(), BackendError> {
        let logical = crate::perm::bit_permute_state(&state.amps, perm);
        state.amps = logical;
        Ok(())
    }
```

- [ ] **Step 4: End-to-end oracle WITH relabelling, 1e-12.** Extend `tiled_oracle.rs`: construct **high-qubit-heavy** Tier-1 circuits (gates deliberately on high qubits so RelabelQubits fires), run via `run_optimized` (relabel + tile + final un-permute), compare to raw `run` of the ORIGINAL circuit within 1e-12. This is the AC #3 transparency gate. Also verify a measurement-bearing circuit reports logical-qubit outcomes.

- [ ] **Step 5:** `cargo test --workspace` green (the full oracle now passes again with relabel+unpermute). Commit `[P2-09] run_optimized: measure-qubit mapping + final un-permute (relabel transparent @ 1e-12)`.

---

### Task 10: permutation property tests + thread-invariance

**Files:** Create/extend `crates/aleph-sv/tests/relabel_property.rs`.

- [ ] **Step 1: proptest** — random circuit (n∈4..8, random 1q/2q gates) run via `run_optimized` (relabel may fire) equals raw `run` of the same circuit within 1e-12. Uses `aleph-test` strategies.
- [ ] **Step 2: thread-invariance** — `run_optimized` result on a relabel+tile circuit is identical across `RAYON_NUM_THREADS` 1 vs many.
- [ ] **Step 3:** `cargo test -p aleph-sv --test relabel_property` PASS; `cargo test --workspace` green; clippy/fmt clean. Commit `[P2-09] relabel/tile property + thread-invariance tests`.

---

## Phase C — bench, EPYC perf-stat, docs, PR

### Task 11: cache-blocking bench + EPYC perf-stat + docs + PR

**Files:** Create `benches/benches/cache_blocking.rs`; modify `benches/Cargo.toml`, `docs/perf/phase2.md`.

- [ ] **Step 1: Bench.** Two workloads, each via `run_optimized` (full pipeline incl. relabel+tile):
  - **low-qubit-heavy** circuit at n ∈ {22, 25}: many 1q/2q gates concentrated on low qubits (a deep run of single-qubit rotations + nearest-neighbour CNOTs on qubits 0..6, repeated) — the regime tiling helps. Compare a `run_optimized`-with-TileBlock build vs a baseline build with TileBlock removed from the pipeline (add a `Circuit::optimize_no_tiling()` test helper, or bench the pre-TileBlock circuit vs post). Register gated behind `scaling-bench`.
  - **counter-case**: random brick-wall (high-qubit-spanning) — expected no win.
- [ ] **Step 2: EPYC perf-stat (AC #1).** Build the bench binary, deliver via git bundle, idle-check, run:
```bash
perf stat -e cache-misses,LLC-load-misses,L1-dcache-load-misses \
  ./target/release/deps/cache_blocking-<hash> --bench <low-qubit case>
```
on both the tiled and non-tiled builds; record the L2/L3 miss reduction. Also capture wall-clock (AC #2). Follow the EPYC ops in the spec.
- [ ] **Step 3: docs.** Add `docs/perf/phase2.md` §11 with: the perf-stat cache-miss table (tiled vs non-tiled), wall-clock speedup in the cache-resident regime, the honest counter-case (random brick-wall: no win), and the tile_bits sweep result. State plainly whether AC #1/#2 are met.
- [ ] **Step 4: full CI-parity** locally: `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --check`, `cargo check --target x86_64-unknown-linux-gnu -p aleph-sv`.
- [ ] **Step 5: PR.** `git push -u origin p2-09-cache-blocking`; `gh pr create` titled `[P2-09] Cache-blocked multi-gate application`, body with `Closes #109`, approach, test results (oracle 1e-12, bit-exact tile-major≡gate-major, perm round-trip, thread-invariance), the EPYC perf-stat numbers, and the `🤖 Generated with [Claude Code](https://claude.com/claude-code)` footer.

---

## Validation Ops (EPYC)

- [[aleph-bench-server]] `ssh root@195.154.249.85`; deliver via git bundle (not GitHub push); idle-check ([[feedback-check-server-clean]]); export toolchain on PATH; `RUSTFLAGS="-C target-cpu=native"`; `rm -rf /root/aleph-* /root/*.bundle` after. `perf` 7.0.0 present; L2 = 1 MiB/core. `cargo check --target x86_64-unknown-linux-gnu` validates SIMD locally (aarch64 dev box).

## Self-Review notes

- **Spec coverage:** TiledBlock driver → Tasks 1-5; tile-major≡gate-major bit-exact → Task 4/5; oracle 1e-12 (AC #3) → Tasks 6, 9; RelabelQubits + tracking → Tasks 7-9 (single final gather via `unpermute_state` hook); perf-stat cache-miss (AC #1) + cache-resident speedup (AC #2) → Task 11; tile policy → Task 5; thread-invariance → Tasks 4, 10.
- **Layering caveat surfaced:** `aleph-backend` can't depend on `aleph-sv`, so the final un-permute is a `Backend::unpermute_state` trait hook (default errors, `NaiveSvBackend` overrides) — Task 9 Step 2/3. This keeps the oracle/measure/HasAmplitudes interfaces unchanged (state is logical-order on return) per the spec's "single final remap" decision.
- **Correctness-first flags:** Task 4 recommends threading a `Result` rather than `.expect` in `dispatch_tile_kernel`; Task 8's relabel guard is conservative (identity when no net win) so correctness never depends on the heuristic; Tasks 6/9 oracle tolerances must not be loosened to mask a real bug.
- **Type consistency:** `TiledBlock {gates, tile_bits}`, `apply_1q_tile`/`apply_2q_tile`, `apply_tiled_block`, `unpermute_state`, `bit_permute_state`, `qubit_permutation`/`set_qubit_permutation`, `RelabelQubits`, `TileBlock`/`DEFAULT_TILE_BITS`, `tuning::tile_bits` — consistent across tasks.
