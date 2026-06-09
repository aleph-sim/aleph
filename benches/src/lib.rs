//! `aleph-benches`: workspace-level benchmark harness.
//!
//! The actual benchmarks live in `benches/*.rs` and are run via
//! `cargo bench --bench <name>` or `cargo bench --workspace`.  This
//! `lib.rs` exposes shared circuit-builders the individual bench
//! files invoke through `NaiveSvBackend` via `aleph_backend::run`.
//!
//! See `docs/benchmarking.md` at the repo root for the benchmark
//! policy (when to add, how to interpret results, how bencher.dev
//! tracks history). The link is intentionally not a `cargo doc`
//! intra-doc link because the file lives outside the crate.

use aleph_core::{Complex, Gate, GateInstance};
use aleph_ir::Circuit;
use smallvec::smallvec;

/// Linear cross-entropy benchmarking (XEB) value of a final state vector,
/// in the exact noiseless (collision-probability) form
/// `XEB = 2^n · Σ_x p(x)² − 1`, where `p(x) = |amp_x|²`.
///
/// For a Porter–Thomas (well-scrambled) circuit this is ≈ 1; for the uniform
/// distribution it is 0. Equivalent to the experimental
/// `2^n·⟨p(x_i)⟩ − 1` when the samples `x_i` are drawn from the ideal
/// distribution itself (the noiseless case). See Arute et al., Nature 574 (2019).
///
/// # Panics
/// Panics if `amps` is empty or its length is not a power of two.
#[must_use]
pub fn linear_xeb(amps: &[Complex]) -> f64 {
    let dim = amps.len();
    // `is_power_of_two()` already rejects 0, so this also guarantees `dim > 0`.
    assert!(
        dim.is_power_of_two(),
        "state length must be a non-zero power of two"
    );
    let sum_p_sq: f64 = amps.iter().map(|a| a.norm_sqr().powi(2)).sum();
    dim as f64 * sum_p_sq - 1.0
}

/// Bell pair on 2 qubits: `H q[0]; CX q[0], q[1]` → `(|00⟩ + |11⟩)/√2`.
#[must_use]
pub fn bell_circuit() -> Circuit {
    let mut c = Circuit::new(2, 0);
    let _ = c.h(0);
    let _ = c.cnot(0, 1);
    c
}

/// GHZ state on `n` qubits: `H q[0]; CX q[0],q[1]; CX q[1],q[2]; …`
/// → `(|0…0⟩ + |1…1⟩)/√2`.
#[must_use]
pub fn ghz_circuit(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    let _ = c.h(0);
    for t in 1..n {
        let _ = c.cnot(t - 1, t);
    }
    c
}

/// Textbook QFT on `n` qubits per Nielsen & Chuang § 5.1: per-qubit
/// `H` followed by a descending ladder of controlled-`Phase` gates.
/// (Closing SWAPs that reverse the qubit order are omitted — they
/// don't affect bench-relevant gate-application cost.)
#[must_use]
pub fn qft_circuit(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for j in 0..n {
        let _ = c.h(j);
        for k in (j + 1)..n {
            // Controlled-Phase(π / 2^(k-j)) with `k` as control, `j`
            // as target.  No builder shortcut for cphase yet —
            // construct via `GateInstance::controlled(Phase, ...)`.
            // Phase is diagonal so the controlled form commutes with
            // its qubit ordering, but we match the textbook (control
            // = higher-index qubit).
            let theta = std::f64::consts::PI / (1u64 << (k - j)) as f64;
            let _ = c.add_gate(GateInstance::controlled(
                Gate::Phase(theta.into()),
                smallvec![j],
                smallvec![k],
            ));
        }
    }
    c
}

/// Inverse QFT on `n` qubits: `qft_circuit(n)`'s instructions in reverse order,
/// each gate replaced by its inverse (`H→H`, `Phase(θ)→Phase(−θ)`), preserving
/// the control/target qubits. `qft_circuit(n)` followed by `qft_inverse_circuit(n)`
/// is the identity.
#[must_use]
pub fn qft_inverse_circuit(n: u32) -> aleph_ir::Circuit {
    let fwd = qft_circuit(n);
    let mut inv = aleph_ir::Circuit::new(n, 0);
    for inst in fwd.instructions().iter().rev() {
        match inst {
            aleph_ir::Instruction::Gate(g) => {
                let mut g2 = g.clone(); // preserves qubits AND controls
                g2.gate = g.gate.inverse();
                inv.add_instruction(aleph_ir::Instruction::Gate(g2))
                    .unwrap();
            }
            other => {
                inv.add_instruction(other.clone()).unwrap();
            }
        }
    }
    inv
}

/// A small non-trivial state prep: H on every qubit, then a phase rotation on
/// each, producing a generic (non-basis) state for round-trip testing.
#[cfg(test)]
fn generic_prep_circuit(n: u32) -> aleph_ir::Circuit {
    use aleph_core::Param;
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(GateInstance::new(Gate::H, vec![q])).unwrap();
    }
    for q in 0..n {
        c.add_gate(GateInstance::new(
            Gate::Rz(Param::Concrete(0.1 * (q as f64 + 1.0))),
            vec![q],
        ))
        .unwrap();
    }
    c
}

/// A low-qubit-heavy circuit: `depth` layers of single-qubit Rz+Rx
/// rotations and nearest-neighbour CNOTs confined to the lowest `width`
/// qubits, on an `n`-qubit register (the high qubits stay idle).
///
/// This is the regime the P2-09 tile-major executor targets: most gates
/// are tile-confinable (`TileBlock` default `tile_bits = 15` ≥ `width`),
/// so the tile executor collapses many DRAM passes into one.  Angles are
/// deterministic (function of `(layer, q)`) so no rand dependency.
///
/// # Panics
/// Panics if `width < 2` or `width > n`.
#[must_use]
pub fn low_qubit_heavy_circuit(n: u32, width: u32, depth: usize) -> Circuit {
    assert!(width >= 2 && width <= n, "width must be in [2, n]");
    let mut c = Circuit::new(n, 0);
    for layer in 0..depth {
        // 1q rotations on every active qubit — same angle idiom as
        // random_brickwall_circuit so the builder stays consistent.
        for q in 0..width {
            let theta = ((layer as f64 + 1.0) * 0.123 + q as f64 * 0.071) % std::f64::consts::TAU;
            let _ = c.rz(theta, q);
            let _ = c.rx(theta * 1.13, q);
        }
        // Nearest-neighbour CNOT layer inside the active window.
        // Even layers: (0,1),(2,3),…; odd layers: (1,2),(3,4),…
        let offset = (layer & 1) as u32;
        let mut q = offset;
        while q + 1 < width {
            let _ = c.cnot(q, q + 1);
            q += 2;
        }
    }
    c
}

/// Brick-wall random-circuit-shaped workload, `depth` layers of
/// alternating-pair CNOTs interleaved with random 1q rotations.  Not
/// a real Sycamore-style random circuit (no Haar-random SU(4)
/// blocks), but the bandwidth shape and gate count match what a
/// state-vector backend pays per layer.
///
/// The rotation angles are deterministic (function of `(layer, q)`)
/// so the bench is reproducible without bringing rand into the
/// dep tree.
#[must_use]
pub fn random_brickwall_circuit(n: u32, depth: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for layer in 0..depth {
        // 1q rotation on every qubit — fills the time-axis with
        // single-qubit work.
        for q in 0..n {
            let theta = ((layer as f64) + (q as f64) * 0.37).cos();
            let _ = c.rz(theta, q);
            let _ = c.rx(theta * 1.13, q);
        }
        // CNOT layer: even layers pair (0,1),(2,3),…; odd layers
        // offset by 1 to pair (1,2),(3,4),…  Standard brick-wall.
        let offset = (layer & 1) as u32;
        let mut q = offset;
        while q + 1 < n {
            let _ = c.cnot(q, q + 1);
            q += 2;
        }
    }
    c
}

#[cfg(test)]
mod qft_roundtrip_tests {
    use super::*;
    use aleph_backend::run;
    use aleph_core::Complex;
    use aleph_sv::NaiveSvBackend;

    /// Apply `circuit` to a freshly allocated |0…0⟩ and return amplitudes.
    fn run_amps(circuit: &aleph_ir::Circuit) -> Vec<Complex> {
        let mut b = NaiveSvBackend::with_seed(7);
        let state = run(&mut b, circuit).expect("run");
        // `HasAmplitudes`/state amplitudes accessor — match the crate's API.
        state.amplitudes().to_vec()
    }

    #[test]
    fn qft_then_inverse_is_identity_on_zero_state() {
        let n = 6;
        let mut c = qft_circuit(n);
        // Append the inverse so the combined circuit should be the identity.
        for inst in qft_inverse_circuit(n).instructions() {
            c.add_instruction(inst.clone()).unwrap();
        }
        let amps = run_amps(&c);
        assert!((amps[0].re - 1.0).abs() < 1e-10, "amp[0] should be 1");
        assert!(amps[0].im.abs() < 1e-10);
        for (k, a) in amps.iter().enumerate().skip(1) {
            assert!(a.norm() < 1e-10, "amp[{k}] should be ~0");
        }
    }

    #[test]
    fn qft_then_inverse_is_identity_on_generic_state() {
        // Per the P1-13 lesson, a |0…0⟩-only check misses bugs. Prep a generic
        // state with a layer of H + T-like rotations, snapshot it, then apply
        // QFT∘QFT⁻¹ and assert the state is unchanged.
        let n = 5;
        let prep = generic_prep_circuit(n); // defined below
        let before = run_amps(&prep);

        let mut c = prep.clone();
        for inst in qft_circuit(n).instructions() {
            c.add_instruction(inst.clone()).unwrap();
        }
        for inst in qft_inverse_circuit(n).instructions() {
            c.add_instruction(inst.clone()).unwrap();
        }
        let after = run_amps(&c);

        assert_eq!(before.len(), after.len());
        for (k, (x, y)) in before.iter().zip(after.iter()).enumerate() {
            assert!((x - y).norm() < 1e-10, "amp[{k}] changed: {x:?} vs {y:?}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_ir::Instruction;

    /// Sanity: the low-qubit-heavy circuit (width=6 < tile_bits=15)
    /// must produce at least one `TiledBlock` after `optimize()`, so the
    /// cache-blocking bench actually exercises the tile-major executor.
    #[test]
    fn low_qubit_heavy_optimize_produces_tiled_block() {
        let mut c = low_qubit_heavy_circuit(12, 6, 10);
        c.optimize().expect("optimize should not fail");
        let has_tiled = c
            .instructions()
            .iter()
            .any(|i| matches!(i, Instruction::TiledBlock(_)));
        assert!(
            has_tiled,
            "optimize() on low_qubit_heavy_circuit(12,6,10) must produce \
             at least one TiledBlock (width=6 < tile_bits=15 means all \
             active-window gates are tile-confinable)"
        );
    }

    #[test]
    fn linear_xeb_uniform_is_zero() {
        // Uniform distribution: every p(x) = 1/D, so 2^n * sum(1/D^2) - 1
        // = D * (D * 1/D^2) - 1 = 0. (A fully depolarized / unscrambled output.)
        let n = 4u32;
        let dim = 1usize << n;
        let amp = Complex::new((1.0 / dim as f64).sqrt(), 0.0);
        let amps = vec![amp; dim];
        assert!(linear_xeb(&amps).abs() < 1e-12);
    }

    #[test]
    fn linear_xeb_peaked_is_dim_minus_one() {
        // A single basis state carries all probability: XEB = D*1 - 1 = D - 1.
        let n = 4u32;
        let dim = 1usize << n;
        let mut amps = vec![Complex::new(0.0, 0.0); dim];
        amps[0] = Complex::new(1.0, 0.0);
        assert!((linear_xeb(&amps) - (dim as f64 - 1.0)).abs() < 1e-12);
    }
}

/// Rotated surface code (Fowler et al. 2012; rotated variant per Tomita &
/// Svore 2014). Distance `d` (odd ≥ 3): `d²` data qubits + `d²−1` ancillas
/// (`2d²−1` total). See docs/superpowers/specs/2026-06-09-p4-07-surface-code-design.md.
#[derive(Clone, Debug)]
pub struct Ancilla {
    pub index: u32,
    pub is_x: bool,
    pub data_neighbours: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct SurfaceCode {
    pub distance: usize,
    pub num_qubits: usize,
    pub data: Vec<u32>,
    pub ancillas: Vec<Ancilla>,
    pub logical_x: Vec<u32>,
    pub logical_z: Vec<u32>,
}

impl SurfaceCode {
    /// Build the rotated surface code of distance `d` (odd, ≥ 3).
    ///
    /// # Panics
    /// Panics if `d < 3` or `d` is even.
    #[must_use]
    pub fn new(distance: usize) -> Self {
        let d = distance;
        assert!(
            d >= 3 && d % 2 == 1,
            "distance must be odd and >= 3, got {d}"
        );
        let di = d as i32;
        let didx = |r: i32, c: i32| -> u32 { (r as u32) * d as u32 + c as u32 };

        let data: Vec<u32> = (0..(d * d) as u32).collect();
        let mut ancillas: Vec<Ancilla> = Vec::with_capacity(d * d - 1);
        let mut next = (d * d) as u32;

        // Candidate plaquette centres (r,c), r,c ∈ {-1,…,d-1}; owns the in-grid
        // members of {(r,c),(r,c+1),(r+1,c),(r+1,c+1)}. Type X iff (r+c) even.
        for r in -1..di {
            for c in -1..di {
                let mut nbrs: Vec<u32> = Vec::with_capacity(4);
                for (rr, cc) in [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)] {
                    if (0..di).contains(&rr) && (0..di).contains(&cc) {
                        nbrs.push(didx(rr, cc));
                    }
                }
                let is_x = (r + c).rem_euclid(2) == 0;
                let keep = match nbrs.len() {
                    4 => true,
                    2 => {
                        let horizontal_edge = r == -1 || r == di - 1;
                        let vertical_edge = c == -1 || c == di - 1;
                        (horizontal_edge && is_x) || (vertical_edge && !is_x)
                    }
                    _ => false, // corners (1 neighbour) dropped
                };
                if keep {
                    ancillas.push(Ancilla {
                        index: next,
                        is_x,
                        data_neighbours: nbrs,
                    });
                    next += 1;
                }
            }
        }

        // Logical X = data column 0 (top↔bottom); logical Z = data row 0 (left↔right).
        let logical_x: Vec<u32> = (0..d as u32).map(|r| r * d as u32).collect();
        let logical_z: Vec<u32> = (0..d as u32).collect();

        Self {
            distance: d,
            num_qubits: 2 * d * d - 1,
            data,
            ancillas,
            logical_x,
            logical_z,
        }
    }

    /// Ancilla measurement order (construction order; matches the Stim program).
    #[must_use]
    pub fn ancilla_order(&self) -> Vec<u32> {
        self.ancillas.iter().map(|a| a.index).collect()
    }

    /// One syndrome-extraction cycle as gates (no measurements). X-ancillas:
    /// `H a; CX a d…; H a`. Z-ancillas: `CX d… a`. Caller measures ancillas
    /// in `ancilla_order()` afterwards.
    #[must_use]
    pub fn cycle_gates(&self) -> Vec<GateInstance> {
        let mut gates = Vec::new();
        for a in self.ancillas.iter().filter(|a| a.is_x) {
            gates.push(GateInstance::new(Gate::H, vec![a.index]));
            for &d in &a.data_neighbours {
                gates.push(GateInstance::new(Gate::Cnot, vec![a.index, d]));
            }
            gates.push(GateInstance::new(Gate::H, vec![a.index]));
        }
        for a in self.ancillas.iter().filter(|a| !a.is_x) {
            for &d in &a.data_neighbours {
                gates.push(GateInstance::new(Gate::Cnot, vec![d, a.index]));
            }
        }
        gates
    }
}

#[cfg(test)]
mod surface_tests {
    use super::*;
    use aleph_backend::Backend;
    use aleph_stab::StabilizerBackend;

    // Symplectic anticommutation of two supports given as (data-set, is_x):
    // two Paulis anticommute iff the X-support of one overlaps the Z-support
    // of the other in an odd total count. For an all-X op P and all-Z op Q on
    // data sets A, B: they anticommute iff |A ∩ B| is odd.
    fn anticommute_xz(x_support: &[u32], z_support: &[u32]) -> bool {
        let zset: std::collections::HashSet<u32> = z_support.iter().copied().collect();
        x_support.iter().filter(|q| zset.contains(q)).count() % 2 == 1
    }

    #[test]
    fn counts_are_correct() {
        for d in [3usize, 5, 7, 9, 11] {
            let sc = SurfaceCode::new(d);
            assert_eq!(sc.data.len(), d * d, "d={d} data count");
            assert_eq!(sc.ancillas.len(), d * d - 1, "d={d} ancilla count");
            assert_eq!(sc.num_qubits, 2 * d * d - 1, "d={d} total");
            let xs = sc.ancillas.iter().filter(|a| a.is_x).count();
            let zs = sc.ancillas.iter().filter(|a| !a.is_x).count();
            assert_eq!(xs, (d * d - 1) / 2, "d={d} X-ancilla count");
            assert_eq!(zs, (d * d - 1) / 2, "d={d} Z-ancilla count");
            // Every ancilla weight is 2 or 4; indices are unique and contiguous.
            for a in &sc.ancillas {
                assert!(
                    a.data_neighbours.len() == 2 || a.data_neighbours.len() == 4,
                    "d={d} ancilla {} weight {}",
                    a.index,
                    a.data_neighbours.len()
                );
            }
            let mut idx: Vec<u32> = sc.ancillas.iter().map(|a| a.index).collect();
            idx.sort_unstable();
            let expect: Vec<u32> = ((d * d) as u32..(2 * d * d - 1) as u32).collect();
            assert_eq!(idx, expect, "d={d} ancilla indices contiguous after data");
        }
    }

    #[test]
    fn all_stabilizers_commute() {
        // X-ancilla (all-X) vs Z-ancilla (all-Z) must share an even number of
        // data qubits. Same-type pairs always commute.
        for d in [3usize, 5, 7, 9, 11] {
            let sc = SurfaceCode::new(d);
            for ax in sc.ancillas.iter().filter(|a| a.is_x) {
                for az in sc.ancillas.iter().filter(|a| !a.is_x) {
                    assert!(
                        !anticommute_xz(&ax.data_neighbours, &az.data_neighbours),
                        "d={d}: X-anc {} and Z-anc {} anticommute",
                        ax.index,
                        az.index
                    );
                }
            }
        }
    }

    #[test]
    fn logicals_commute_with_stabilizers_and_anticommute_each_other() {
        for d in [3usize, 5, 7, 9, 11] {
            let sc = SurfaceCode::new(d);
            assert_eq!(sc.logical_x.len(), d, "d={d} logical X weight");
            assert_eq!(sc.logical_z.len(), d, "d={d} logical Z weight");
            // logical_x (all-X) commutes with every Z-stabilizer.
            for az in sc.ancillas.iter().filter(|a| !a.is_x) {
                assert!(
                    !anticommute_xz(&sc.logical_x, &az.data_neighbours),
                    "d={d}: logical_x anticommutes with Z-anc {}",
                    az.index
                );
            }
            // logical_z (all-Z) commutes with every X-stabilizer.
            for ax in sc.ancillas.iter().filter(|a| a.is_x) {
                assert!(
                    !anticommute_xz(&ax.data_neighbours, &sc.logical_z),
                    "d={d}: logical_z anticommutes with X-anc {}",
                    ax.index
                );
            }
            // The two logicals anticommute (overlap on exactly one data qubit).
            assert!(
                anticommute_xz(&sc.logical_x, &sc.logical_z),
                "d={d}: logicals must anticommute"
            );
        }
    }

    /// Run one cycle from data |0…0⟩ and return the measured outcome for each
    /// ancilla, in `ancilla_order()`.
    fn run_cycle(sc: &SurfaceCode, seed: u64, pre: &[GateInstance]) -> Vec<bool> {
        let mut be = StabilizerBackend::with_seed(seed);
        let mut t = be.allocate(sc.num_qubits as u32).unwrap();
        for g in pre {
            be.apply_gate(&mut t, g).unwrap();
        }
        for g in sc.cycle_gates() {
            be.apply_gate(&mut t, &g).unwrap();
        }
        sc.ancilla_order()
            .iter()
            .map(|&a| be.measure(&mut t, a).unwrap())
            .collect()
    }

    #[test]
    fn z_syndrome_is_zero_from_ground_state() {
        // From |0…0⟩, every Z-stabilizer is +1 ⇒ its ancilla measures 0,
        // for any seed. (X-ancillas are random and not asserted here.)
        for d in [3usize, 5] {
            let sc = SurfaceCode::new(d);
            let order = sc.ancilla_order();
            for seed in [0u64, 1, 7, 42] {
                let out = run_cycle(&sc, seed, &[]);
                for (k, &anc) in order.iter().enumerate() {
                    let is_z = !sc.ancillas.iter().find(|a| a.index == anc).unwrap().is_x;
                    if is_z {
                        assert!(!out[k], "d={d} seed={seed}: Z-ancilla {anc} fired from |0>");
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "distance must be odd")]
    fn rejects_even_distance() {
        let _ = SurfaceCode::new(4);
    }

    #[test]
    #[should_panic(expected = "distance must be odd")]
    fn rejects_too_small_distance() {
        let _ = SurfaceCode::new(1);
    }
}
