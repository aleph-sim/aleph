//! Rotated surface code geometry and a multi-round **memory-Z** experiment.
//!
//! Geometry follows Fowler et al. 2012 / Tomita & Svore 2014 (rotated variant),
//! mirroring the layout in `aleph-benches` so the two stay consistent.
//!
//! The memory-Z experiment measures only the **Z stabilizers** over `rounds`
//! rounds, starting from `|0…0⟩`. Every measured stabilizer is then deterministic
//! in the noiseless circuit (data `|0⟩` is a `Z` eigenstate), which keeps the
//! whole experiment in the regime where both the DEM builder
//! ([`crate::build_dem`]) and the Q0-02 frame sampler are exact. It detects `X`
//! data errors and stores the logical-`Z` observable — the standard single-basis
//! decoder benchmark. (X-stabilizer / memory-X is the mirror image; later work.)

use aleph_core::{Gate, GateInstance};
use aleph_ir::{Circuit, Instruction};

use crate::builder::{build_dem, AnnotatedCircuit, CircuitNoise, ErrorMechanism};
use crate::dem::DetectorErrorModel;
use crate::error::Result;

/// One stabilizer ancilla and the data qubits it checks.
#[derive(Clone, Debug)]
struct Ancilla {
    index: u32,
    is_x: bool,
    data_neighbours: Vec<u32>,
}

/// Rotated surface code of odd distance `d`: `d²` data qubits (indices `0..d²`)
/// and `d²−1` ancillas (indices `d²..2d²−1`).
#[derive(Clone, Debug)]
pub struct SurfaceCode {
    /// Code distance (odd, ≥ 3).
    pub distance: usize,
    /// Total qubits = `2d² − 1`.
    pub num_qubits: usize,
    ancillas: Vec<Ancilla>,
    /// Logical-Z support: data row 0 (`0..d`).
    logical_z: Vec<u32>,
}

impl SurfaceCode {
    /// Build the rotated surface code of distance `d` (odd, ≥ 3).
    ///
    /// # Panics
    /// If `d < 3` or `d` is even.
    pub fn new(distance: usize) -> Self {
        let d = distance;
        assert!(
            d >= 3 && d % 2 == 1,
            "distance must be odd and >= 3, got {d}"
        );
        let di = d as i32;
        let didx = |r: i32, c: i32| -> u32 { (r as u32) * d as u32 + c as u32 };

        let mut ancillas: Vec<Ancilla> = Vec::with_capacity(d * d - 1);
        let mut next = (d * d) as u32;
        // Plaquette centres (r,c), r,c ∈ {-1,…,d-1}; type X iff (r+c) even.
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
                    _ => false,
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
        let logical_z: Vec<u32> = (0..d as u32).collect(); // data row 0
        Self {
            distance: d,
            num_qubits: 2 * d * d - 1,
            ancillas,
            logical_z,
        }
    }

    /// The Z-type ancillas, in construction order.
    fn z_ancillas(&self) -> Vec<&Ancilla> {
        self.ancillas.iter().filter(|a| !a.is_x).collect()
    }

    /// Build the `rounds`-round memory-Z experiment.
    pub fn memory_z_experiment(&self, rounds: usize) -> MemoryExperiment {
        assert!(rounds >= 1, "need at least one round");
        let d = self.distance;
        let nd = d * d;
        let z_anc = self.z_ancillas();
        let nz = z_anc.len();

        let mut inst: Vec<Instruction> = Vec::new();
        // Per-record bookkeeping.
        let mut meas_qubit: Vec<u32> = Vec::new();
        let mut meas_instr: Vec<usize> = Vec::new();
        let mut round_starts: Vec<usize> = Vec::with_capacity(rounds);
        // ancilla_rec[r][k] = measurement-record index of z-ancilla k in round r.
        let mut ancilla_rec: Vec<Vec<usize>> = Vec::with_capacity(rounds);
        // Circuit-level noise geometry (Q-surface circuit-level DEM). `at` follows
        // ErrorMechanism semantics (the error is present for instructions ≥ at).
        let mut cnot_sites: Vec<(usize, u32, u32)> = Vec::new(); // (at = CNOT index, control, target)
        let mut reset_sites: Vec<(usize, u32)> = Vec::new(); // (at = index after reset, ancilla qubit)

        let mut clbit = 0u32;
        let push_measure = |inst: &mut Vec<Instruction>,
                            meas_qubit: &mut Vec<u32>,
                            meas_instr: &mut Vec<usize>,
                            clbit: &mut u32,
                            q: u32|
         -> usize {
            let rec = meas_qubit.len();
            meas_instr.push(inst.len());
            meas_qubit.push(q);
            inst.push(Instruction::Measure {
                qubit: q,
                clbit: *clbit,
            });
            *clbit += 1;
            rec
        };

        for _r in 0..rounds {
            round_starts.push(inst.len());
            // Z-stabilizer extraction: CX(data -> ancilla).
            for a in &z_anc {
                for &dq in &a.data_neighbours {
                    cnot_sites.push((inst.len(), dq, a.index));
                    inst.push(Instruction::Gate(GateInstance::new(
                        Gate::Cnot,
                        vec![dq, a.index],
                    )));
                }
            }
            // Measure + reset each Z-ancilla.
            let mut this_round: Vec<usize> = Vec::with_capacity(nz);
            for a in &z_anc {
                let rec = push_measure(
                    &mut inst,
                    &mut meas_qubit,
                    &mut meas_instr,
                    &mut clbit,
                    a.index,
                );
                this_round.push(rec);
                inst.push(Instruction::Reset(a.index));
                // A faulty reset leaves the ancilla X-flipped *after* the reset clears it.
                reset_sites.push((inst.len(), a.index));
            }
            ancilla_rec.push(this_round);
        }

        // Final data readout (Z basis).
        let final_data_error_at = inst.len();
        let mut data_rec: Vec<usize> = vec![0; nd];
        for q in 0..nd as u32 {
            let rec = push_measure(&mut inst, &mut meas_qubit, &mut meas_instr, &mut clbit, q);
            data_rec[q as usize] = rec;
        }

        // --- Detectors (round differences + final from data). Order fixes D-index. ---
        let mut detectors: Vec<Vec<usize>> = Vec::new();
        for (r, round_recs) in ancilla_rec.iter().enumerate() {
            for (k, &rec) in round_recs.iter().enumerate() {
                if r == 0 {
                    detectors.push(vec![rec]);
                } else {
                    detectors.push(vec![rec, ancilla_rec[r - 1][k]]);
                }
            }
        }
        // Final: each Z-stabilizer reconstructed from data XOR its last ancilla round.
        for (k, a) in z_anc.iter().enumerate() {
            let mut recs: Vec<usize> = a
                .data_neighbours
                .iter()
                .map(|&q| data_rec[q as usize])
                .collect();
            recs.push(ancilla_rec[rounds - 1][k]);
            detectors.push(recs);
        }

        // Logical-Z observable from the final data measurements.
        let observable: Vec<usize> = self
            .logical_z
            .iter()
            .map(|&q| data_rec[q as usize])
            .collect();

        let circuit = {
            let mut c = Circuit::new(self.num_qubits as u32, clbit.max(1));
            for i in inst {
                c.add_instruction(i).expect("valid instruction");
            }
            c
        };

        MemoryExperiment {
            annotated: AnnotatedCircuit {
                circuit,
                detectors,
                observables: vec![observable],
            },
            distance: d,
            rounds,
            num_qubits: self.num_qubits,
            data_qubits: (0..nd as u32).collect(),
            round_starts,
            final_data_error_at,
            meas_qubit,
            meas_instr,
            num_ancilla_records: rounds * nz,
            cnot_sites,
            reset_sites,
        }
    }
}

/// A built memory-Z experiment: the annotated circuit plus the metadata needed
/// to enumerate phenomenological error mechanisms and to emit an equivalent Stim
/// program.
#[derive(Clone, Debug)]
pub struct MemoryExperiment {
    /// Circuit + detector / observable definitions.
    pub annotated: AnnotatedCircuit,
    /// Code distance.
    pub distance: usize,
    /// Number of rounds.
    pub rounds: usize,
    /// Total qubits.
    pub num_qubits: usize,
    data_qubits: Vec<u32>,
    round_starts: Vec<usize>,
    final_data_error_at: usize,
    meas_qubit: Vec<u32>,
    meas_instr: Vec<usize>,
    num_ancilla_records: usize,
    /// `(at = CNOT instruction index, control, target)` for every CNOT — site of a two-qubit
    /// depolarizing channel in the circuit-level model.
    cnot_sites: Vec<(usize, u32, u32)>,
    /// `(at = index just after each ancilla reset, ancilla qubit)` — site of a preparation flip.
    reset_sites: Vec<(usize, u32)>,
}

impl MemoryExperiment {
    /// Time coordinate (round index) of every detector, for streaming / sliding-window decoding
    /// (Q4-01). Detectors are emitted round-by-round (`nz` stabilizer detectors per round, in index
    /// order), so the round-difference detectors live at their round `r ∈ 0..rounds`; the final
    /// data-readout block lives at time `rounds` (one slice past the last round). Times span
    /// `0..=rounds`, giving `rounds + 1` time slices the sliding window can cut across.
    pub fn detector_rounds(&self) -> Vec<usize> {
        let nz = self.num_ancilla_records / self.rounds;
        let n_round_dets = self.rounds * nz;
        (0..self.annotated.detectors.len())
            .map(|d| {
                if d < n_round_dets {
                    d / nz
                } else {
                    self.rounds
                }
            })
            .collect()
    }

    /// Phenomenological error mechanisms: an independent `X` error of probability
    /// `p_data` on every data qubit before every round and before final readout,
    /// and a measurement flip of probability `p_meas` on every ancilla
    /// measurement (an `X` on the ancilla at its measurement, cleared by reset).
    pub fn phenomenological_mechanisms(&self, p_data: f64, p_meas: f64) -> Vec<ErrorMechanism> {
        let mut mechs = Vec::new();
        for &at in &self.round_starts {
            for &q in &self.data_qubits {
                mechs.push(ErrorMechanism {
                    prob: p_data,
                    x: vec![q],
                    z: vec![],
                    at,
                });
            }
        }
        for &q in &self.data_qubits {
            mechs.push(ErrorMechanism {
                prob: p_data,
                x: vec![q],
                z: vec![],
                at: self.final_data_error_at,
            });
        }
        for rec in 0..self.num_ancilla_records {
            mechs.push(ErrorMechanism {
                prob: p_meas,
                x: vec![self.meas_qubit[rec]],
                z: vec![],
                at: self.meas_instr[rec],
            });
        }
        mechs
    }

    /// Mechanisms that flip *every* measurement record (ancilla and data) with
    /// probability `p` — an `X` on the measured qubit at its measurement. This is
    /// exactly the error set produced by the frame sampler's `measure_flip`, used
    /// to cross-check [`build_dem`] against [`aleph_stab::sample_noisy`].
    pub fn measurement_flip_mechanisms(&self, p: f64) -> Vec<ErrorMechanism> {
        (0..self.meas_qubit.len())
            .map(|rec| ErrorMechanism {
                prob: p,
                x: vec![self.meas_qubit[rec]],
                z: vec![],
                at: self.meas_instr[rec],
            })
            .collect()
    }

    /// Circuit-level error mechanisms (`X`-sector) for the memory-Z experiment.
    ///
    /// This is the realistic upgrade over [`Self::phenomenological_mechanisms`]: instead of one data
    /// error per round and a measurement flip, every *gate and operation* in the syndrome-extraction
    /// circuit is a fault site. Because memory-Z detects `X` errors, only each depolarizing channel's
    /// `X`-sector contributes (`build_dem` discards mechanisms that flip no detector and no
    /// observable). The sources, following the standard circuit-level model:
    ///
    /// * **CNOT** — a two-qubit depolarizing channel: `X(c)`, `X(t)`, `X(c)X(t)` each at
    ///   `4/15·p_cnot`. The `X(c)X(t)` correlated component (and the lone `X(t)` on the ancilla) is
    ///   what produces **hook errors** — the diagonal space-time edges that make a circuit-level DEM
    ///   qualitatively harder than the phenomenological one.
    /// * **Idle / storage** — a single-qubit depolarizing `X` at `2/3·p_idle` on every data qubit
    ///   once per round (placed at the round start, before that round's CNOTs).
    /// * **Preparation** — an `X` flip at `p_init` on the initial `|0…0⟩` prep of every qubit (at
    ///   `t=0`) and after every per-round ancilla reset.
    /// * **Measurement** — an `X` flip at `p_meas` on every measurement record (ancilla and final
    ///   data readout).
    ///
    /// The error placed at a CNOT is conjugated *by* that CNOT (a pre-gate channel), matching the
    /// Stim program [`Self::stim_program_circuit_level`] emits, so the two agree edge-for-edge.
    pub fn circuit_level_mechanisms(&self, noise: CircuitNoise) -> Vec<ErrorMechanism> {
        let x1 = |prob: f64, q: u32, at: usize| ErrorMechanism {
            prob,
            x: vec![q],
            z: vec![],
            at,
        };
        let mut mechs = Vec::new();

        // CNOT two-qubit depolarizing, X-sector: X(c), X(t), X(c)X(t) at 4/15·p_cnot.
        let cnot_w = noise.p_cnot * 4.0 / 15.0;
        for &(at, c, t) in &self.cnot_sites {
            mechs.push(x1(cnot_w, c, at));
            mechs.push(x1(cnot_w, t, at));
            mechs.push(ErrorMechanism {
                prob: cnot_w,
                x: vec![c, t],
                z: vec![],
                at,
            });
        }

        // Idle/storage: one single-qubit depolarizing X per data qubit per round (X weight 2/3).
        let idle_w = noise.p_idle * 2.0 / 3.0;
        for &at in &self.round_starts {
            for &q in &self.data_qubits {
                mechs.push(x1(idle_w, q, at));
            }
        }

        // Preparation: initial |0…0⟩ prep of every qubit at t=0, plus every per-round ancilla reset.
        for q in 0..self.num_qubits as u32 {
            mechs.push(x1(noise.p_init, q, 0));
        }
        for &(at, q) in &self.reset_sites {
            mechs.push(x1(noise.p_init, q, at));
        }

        // Measurement flips: an X on every measured qubit at its measurement record.
        for rec in 0..self.meas_qubit.len() {
            mechs.push(x1(noise.p_meas, self.meas_qubit[rec], self.meas_instr[rec]));
        }

        mechs
    }

    /// Build the circuit-level [`DetectorErrorModel`] for this experiment under `noise`.
    ///
    /// Convenience wrapper over [`build_dem`] + [`Self::circuit_level_mechanisms`].
    ///
    /// # Errors
    /// Propagates DEM-construction errors ([`crate::Error::Propagation`]).
    pub fn circuit_level_dem(&self, noise: CircuitNoise) -> Result<DetectorErrorModel> {
        build_dem(&self.annotated, &self.circuit_level_mechanisms(noise))
    }

    /// Emit an equivalent Stim program with the same `X`-sector circuit-level noise, so Stim's
    /// `detector_error_model` can be cross-checked against [`Self::circuit_level_dem`] edge-for-edge.
    /// Detectors/observables are emitted in [`AnnotatedCircuit`] order (so `D{i}`/`L{i}` match ours).
    pub fn stim_program_circuit_level(&self, noise: CircuitNoise) -> String {
        let ac = &self.annotated;
        let insts = ac.circuit.instructions();
        let total_recs = self.meas_qubit.len();
        let rel = |rec: usize| -> String { format!("rec[-{}]", total_recs - rec) };

        // Group the rate-free X-sector noise channels by the instruction index they precede.
        let cnot_w = noise.p_cnot * 4.0 / 15.0;
        let idle_w = noise.p_idle * 2.0 / 3.0;
        let mut emit_before: std::collections::BTreeMap<usize, Vec<String>> =
            std::collections::BTreeMap::new();
        let mut push = |at: usize, s: String| emit_before.entry(at).or_default().push(s);

        for &(at, c, t) in &self.cnot_sites {
            push(at, format!("E({cnot_w}) X{c}"));
            push(at, format!("E({cnot_w}) X{t}"));
            push(at, format!("E({cnot_w}) X{c} X{t}"));
        }
        for &at in &self.round_starts {
            for &q in &self.data_qubits {
                push(at, format!("E({idle_w}) X{q}"));
            }
        }
        for &(at, q) in &self.reset_sites {
            push(at, format!("E({}) X{q}", noise.p_init));
        }
        for rec in 0..self.meas_qubit.len() {
            push(
                self.meas_instr[rec],
                format!("E({}) X{}", noise.p_meas, self.meas_qubit[rec]),
            );
        }

        let mut s = String::new();
        // Initial prep of all qubits in |0⟩ with a prep-flip channel each (t=0 errors).
        let all_qubits: Vec<String> = (0..self.num_qubits).map(|q| q.to_string()).collect();
        s.push_str(&format!("R {}\n", all_qubits.join(" ")));
        for q in 0..self.num_qubits as u32 {
            s.push_str(&format!("E({}) X{q}\n", noise.p_init));
        }

        for (i, inst) in insts.iter().enumerate() {
            if let Some(errs) = emit_before.get(&i) {
                for e in errs {
                    s.push_str(e);
                    s.push('\n');
                }
            }
            match inst {
                Instruction::Gate(gi) => {
                    let q = &gi.qubits;
                    s.push_str(&format!("CX {} {}\n", q[0], q[1]));
                }
                Instruction::Measure { qubit, .. } => s.push_str(&format!("M {qubit}\n")),
                Instruction::Reset(q) => s.push_str(&format!("R {q}\n")),
                _ => {}
            }
        }
        for recs in &ac.detectors {
            let parts: Vec<String> = recs.iter().map(|&r| rel(r)).collect();
            s.push_str(&format!("DETECTOR {}\n", parts.join(" ")));
        }
        for (o, recs) in ac.observables.iter().enumerate() {
            let parts: Vec<String> = recs.iter().map(|&r| rel(r)).collect();
            s.push_str(&format!("OBSERVABLE_INCLUDE({o}) {}\n", parts.join(" ")));
        }
        s
    }

    /// Emit an equivalent Stim program with the same phenomenological noise, so
    /// Stim's `detector_error_model` can be cross-checked against [`build_dem`].
    /// Detectors are emitted in the same order as [`AnnotatedCircuit::detectors`]
    /// (so Stim's `D{i}` matches ours) using `rec[-k]` relative indexing.
    pub fn stim_program(&self, p_data: f64, p_meas: f64) -> String {
        let ac = &self.annotated;
        let total_recs = self.meas_qubit.len();
        // record index -> rec[-k] (k counted from the end, after all measurements).
        let rel = |rec: usize| -> String { format!("rec[-{}]", total_recs - rec) };

        let mut s = String::new();
        let all_qubits: Vec<String> = (0..self.num_qubits).map(|q| q.to_string()).collect();
        s.push_str(&format!("R {}\n", all_qubits.join(" ")));

        // Re-walk the same instruction list, inserting noise at the matching points.
        let data_list: Vec<String> = self.data_qubits.iter().map(|q| q.to_string()).collect();
        let mut rebuilt_round = 0usize;
        let insts = ac.circuit.instructions();
        let mut i = 0usize;
        while i < insts.len() {
            // Data X errors at each round start and at final readout.
            if rebuilt_round < self.round_starts.len() && i == self.round_starts[rebuilt_round] {
                s.push_str(&format!("X_ERROR({p_data}) {}\n", data_list.join(" ")));
                rebuilt_round += 1;
            }
            if i == self.final_data_error_at {
                s.push_str(&format!("X_ERROR({p_data}) {}\n", data_list.join(" ")));
            }
            match &insts[i] {
                Instruction::Gate(gi) => {
                    let q = &gi.qubits;
                    s.push_str(&format!("CX {} {}\n", q[0], q[1]));
                }
                Instruction::Measure { qubit, .. } => {
                    // Ancilla measurements carry a measurement-flip error.
                    let is_ancilla = *qubit as usize >= self.distance * self.distance;
                    if is_ancilla {
                        s.push_str(&format!("X_ERROR({p_meas}) {qubit}\n"));
                    }
                    s.push_str(&format!("M {qubit}\n"));
                }
                Instruction::Reset(q) => s.push_str(&format!("R {q}\n")),
                _ => {}
            }
            i += 1;
        }

        // Detectors and observable, after all measurements.
        for recs in &ac.detectors {
            let parts: Vec<String> = recs.iter().map(|&r| rel(r)).collect();
            s.push_str(&format!("DETECTOR {}\n", parts.join(" ")));
        }
        for (o, recs) in ac.observables.iter().enumerate() {
            let parts: Vec<String> = recs.iter().map(|&r| rel(r)).collect();
            s.push_str(&format!("OBSERVABLE_INCLUDE({o}) {}\n", parts.join(" ")));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_dem;

    #[test]
    fn geometry_counts() {
        for d in [3usize, 5, 7, 9] {
            let sc = SurfaceCode::new(d);
            assert_eq!(sc.num_qubits, 2 * d * d - 1);
            // Rotated code: (d²-1)/2 Z-ancillas.
            assert_eq!(
                sc.z_ancillas().len(),
                (d * d - 1) / 2,
                "d={d} z-ancilla count"
            );
        }
    }

    #[test]
    fn dem_is_graphlike_and_sized() {
        // Every error in a phenomenological memory-Z DEM flips ≤ 2 detectors
        // (graphlike), and detector count = rounds * nz + nz (final).
        for d in [3usize, 5] {
            for rounds in [1usize, 3] {
                let sc = SurfaceCode::new(d);
                let exp = sc.memory_z_experiment(rounds);
                let nz = (d * d - 1) / 2;
                assert_eq!(exp.annotated.detectors.len(), rounds * nz + nz);
                let mechs = exp.phenomenological_mechanisms(0.01, 0.01);
                let dem = build_dem(&exp.annotated, &mechs).unwrap();
                assert_eq!(dem.detectors, rounds * nz + nz);
                assert_eq!(dem.observables, 1);
                for e in &dem.errors {
                    assert!(
                        e.dets.len() <= 2,
                        "d={d} rounds={rounds}: non-graphlike error on {:?}",
                        e.dets
                    );
                }
                assert!(!dem.errors.is_empty());
            }
        }
    }

    #[test]
    fn circuit_level_dem_sized() {
        // The circuit-level DEM has the same detector/observable layout as the phenomenological
        // one (same circuit, same detectors), just a richer mechanism set. It must be non-empty
        // and produce sensible per-error detector supports.
        use crate::CircuitNoise;
        for d in [3usize, 5] {
            for rounds in [1usize, 3] {
                let sc = SurfaceCode::new(d);
                let exp = sc.memory_z_experiment(rounds);
                let nz = (d * d - 1) / 2;
                let dem = exp.circuit_level_dem(CircuitNoise::uniform(0.001)).unwrap();
                assert_eq!(dem.detectors, rounds * nz + nz, "d={d} rounds={rounds}");
                assert_eq!(dem.observables, 1);
                assert!(!dem.errors.is_empty());
                // Every fired mechanism flips at least one detector or the observable (build_dem
                // drops the trivial ones, e.g. an X on a Z-ancilla that no detector sees).
                for e in &dem.errors {
                    assert!(!e.dets.is_empty() || !e.obs.is_empty());
                }
            }
        }
    }

    #[test]
    fn circuit_level_dem_is_graphlike() {
        // The value of this test is the empirical claim: with the surface code's CNOT schedule,
        // the X-sector circuit-level DEM is graphlike (every single-fault mechanism flips ≤ 2
        // detectors), so MWPM / Union-Find decode it directly — no hyperedge decomposition needed.
        use crate::CircuitNoise;
        for d in [3usize, 5] {
            for rounds in [1usize, 2, 3] {
                let exp = SurfaceCode::new(d).memory_z_experiment(rounds);
                let dem = exp.circuit_level_dem(CircuitNoise::uniform(0.001)).unwrap();
                let max = dem.errors.iter().map(|e| e.dets.len()).max().unwrap_or(0);
                assert!(
                    max <= 2,
                    "d={d} rounds={rounds}: circuit-level DEM has a {max}-detector hyperedge"
                );
            }
        }
    }

    #[test]
    fn dem_cross_checks_frame_sampler_on_measurement_noise() {
        // End-to-end, no Stim: under measurement-flip-only noise (where the
        // memory-Z circuit stays fully deterministic, so the frame sampler is
        // exact), each detector's empirical firing rate from
        // `aleph_stab::sample_noisy` must match the DEM's predicted rate (the
        // odd-parity combination of the edges touching that detector). This
        // validates detector wiring + probabilities through an independent code
        // path from the DEM builder.
        let exp = SurfaceCode::new(3).memory_z_experiment(2);
        let q = 0.04;
        let dem = build_dem(&exp.annotated, &exp.measurement_flip_mechanisms(q)).unwrap();

        let shots = 200_000u32;
        let out = aleph_stab::sample_noisy(
            &exp.annotated.circuit,
            &aleph_stab::PauliNoise::none().with_measure_flip(q),
            shots,
            7,
        )
        .unwrap();

        let combine = |edges: &[f64]| edges.iter().fold(0.0, |a, &p| a + p - 2.0 * a * p);
        for (di, recs) in exp.annotated.detectors.iter().enumerate() {
            let mut fires = 0u64;
            for s in 0..shots as usize {
                if recs.iter().fold(false, |b, &r| b ^ out.get(s, r)) {
                    fires += 1;
                }
            }
            let emp = fires as f64 / shots as f64;
            let touching: Vec<f64> = dem
                .errors
                .iter()
                .filter(|e| e.dets.contains(&(di as u32)))
                .map(|e| e.prob)
                .collect();
            let pred = combine(&touching);
            assert!(
                (emp - pred).abs() < 0.01,
                "detector {di}: empirical {emp} vs predicted {pred}"
            );
        }
    }

    #[test]
    fn d3_single_round_hand_check() {
        // d=3, 1 round: 4 Z-ancillas → 4 round-0 detectors + 4 final detectors = 8.
        let sc = SurfaceCode::new(3);
        let exp = sc.memory_z_experiment(1);
        assert_eq!(exp.annotated.detectors.len(), 8);
        let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(0.01, 0.02)).unwrap();
        // Some error must touch the logical observable (data errors on row 0).
        assert!(
            dem.errors.iter().any(|e| !e.obs.is_empty()),
            "observable reachable"
        );
        // All probabilities are one of the two inputs or a merge thereof, in (0,1).
        for e in &dem.errors {
            assert!(e.prob > 0.0 && e.prob < 1.0, "prob {} out of range", e.prob);
        }
    }
}
