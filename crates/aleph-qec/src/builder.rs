//! Detector-error-model construction from an annotated Clifford circuit.
//!
//! A [`DetectorErrorModel`] is a property of the circuit + noise, not of any
//! particular run: error mechanism `E` flips detector `D` iff `E`, propagated
//! through the remaining Cliffords, flips an odd number of the measurements that
//! `D` is the parity of. We build it by *symbolic Pauli propagation*
//! ([`aleph_stab::propagate_pauli_flips`]) — deterministic and exact, with no
//! dependence on measurement randomness.
//!
//! Mechanisms with identical (detector, observable) support are merged into a
//! single edge whose probability is the odd-parity combination of the parts
//! (`p₁ ⊕ p₂ = p₁ + p₂ − 2p₁p₂`), matching Stim's DEM.

use std::collections::BTreeMap;

use aleph_ir::Circuit;
use rayon::prelude::*;

use crate::dem::{DemError, DetectorErrorModel};
use crate::error::Result;

/// A Clifford circuit plus the detector and observable definitions over its
/// measurement record.
///
/// Measurements are indexed in circuit order (the i-th `Measure` instruction is
/// record `i`). A detector / observable is the XOR (parity) of the records in
/// its list. A *detector* is a parity that is deterministic in the noiseless
/// circuit (so any flip signals an error); an *observable* is the logical
/// quantity whose flip is a logical error.
#[derive(Clone, Debug)]
pub struct AnnotatedCircuit {
    /// The Clifford circuit (gates, measurements, resets).
    pub circuit: Circuit,
    /// Each detector as the list of measurement-record indices it XORs.
    pub detectors: Vec<Vec<usize>>,
    /// Each logical observable as the list of measurement-record indices it XORs.
    pub observables: Vec<Vec<usize>>,
}

/// One independent physical error mechanism for DEM construction: the Pauli
/// `∏ X^{x} Z^{z}` occurring with probability `prob`, inserted just before
/// instruction index `at`. A measurement error is expressed as an `X` on the
/// measured qubit at its `Measure` instruction (it flips that record and is then
/// cleared by the following reset).
#[derive(Clone, Debug)]
pub struct ErrorMechanism {
    /// Probability the mechanism fires.
    pub prob: f64,
    /// Qubits carrying an X component.
    pub x: Vec<u32>,
    /// Qubits carrying a Z component.
    pub z: Vec<u32>,
    /// Instruction index the error is inserted before.
    pub at: usize,
}

/// Circuit-level depolarizing noise strengths for a syndrome-extraction experiment, shared by the
/// surface-code ([`crate::SurfaceCode::circuit_level_dem`]) and BB-code
/// ([`crate::BBCode::circuit_level_dem`]) circuit-level DEMs.
///
/// Following the standard circuit-level model, each source contributes only the Pauli sector the
/// experiment detects: a two-qubit depolarizing channel after every CNOT (its 15 non-identity Paulis
/// split so each of the relevant-sector components `P⊗I`, `I⊗P`, `P⊗P` appears with weight `4/15`),
/// a single-qubit depolarizing channel on idle data qubits (the detected Pauli with weight `2/3`),
/// and basis-flip errors at preparation and measurement.
#[derive(Clone, Copy, Debug)]
pub struct CircuitNoise {
    /// Two-qubit depolarizing rate per CNOT.
    pub p_cnot: f64,
    /// Preparation (reset-into-basis) error rate.
    pub p_init: f64,
    /// Measurement flip rate.
    pub p_meas: f64,
    /// Single-qubit depolarizing rate on an idle qubit.
    pub p_idle: f64,
}

impl CircuitNoise {
    /// The standard uniform circuit-level model: every source at the same physical rate `p`.
    pub fn uniform(p: f64) -> Self {
        Self {
            p_cnot: p,
            p_init: p,
            p_meas: p,
            p_idle: p,
        }
    }

    /// The **SI1000** superconducting-inspired noise model (Gidney, Newman, Fowler, Broughton,
    /// "A Fault-Tolerant Honeycomb Memory", arXiv:2108.10457): a ~1000 ns cycle where the dominant
    /// error is the long idle/measurement window, so the sources are *unequal* — two-qubit gates at
    /// `p`, reset at `2p`, measurement flip at `5p`, and idle at `2p` (the measurement-window idle,
    /// the largest single-qubit contribution).
    ///
    /// This maps SI1000 onto our four-source [`CircuitNoise`] model. It is faithful for circuits
    /// without single-qubit gates (e.g. the memory-Z Z-extraction, whose only single-qubit op is the
    /// idle); SI1000's separate `p/10` single-qubit-*gate* depolarizing applies wherever the circuit
    /// has explicit 1q gates (e.g. the Hadamards in memory-X) and is not represented by these four
    /// rates — it is the smallest term and is omitted.
    pub fn si1000(p: f64) -> Self {
        Self {
            p_cnot: p,
            p_init: 2.0 * p,
            p_meas: 5.0 * p,
            p_idle: 2.0 * p,
        }
    }
}

/// Build a [`DetectorErrorModel`] for `ac` under the given error `mechanisms`.
///
/// Each mechanism is propagated to its detector/observable support; mechanisms
/// with no support (undetectable / logically trivial) are dropped; the rest are
/// merged by support with odd-parity probability combination.
///
/// # Errors
/// Propagates [`crate::Error::Propagation`] if the circuit contains a gate the
/// stabilizer engine cannot handle.
pub fn build_dem(
    ac: &AnnotatedCircuit,
    mechanisms: &[ErrorMechanism],
) -> Result<DetectorErrorModel> {
    let n = ac.circuit.num_qubits() as usize;

    // (detectors, observables, probability) for one propagated mechanism.
    type Support = (Vec<u32>, Vec<u32>, f64);

    // Propagate each mechanism to its (detector, observable) support. This is the dominant cost
    // for circuit-level DEMs (thousands of mechanisms × a deep circuit), and each propagation is
    // independent, so it parallelises cleanly; the cheap merge is done sequentially afterwards.
    let supports: Vec<Support> = mechanisms
        .par_iter()
        .map(|m| -> Result<Option<Support>> {
            let mut xv = vec![false; n];
            let mut zv = vec![false; n];
            for &q in &m.x {
                xv[q as usize] = true;
            }
            for &q in &m.z {
                zv[q as usize] = true;
            }
            let flips = aleph_stab::propagate_pauli_flips(&ac.circuit, &xv, &zv, m.at)?;

            let dets: Vec<u32> = ac
                .detectors
                .iter()
                .enumerate()
                .filter(|(_, recs)| parity(recs, &flips))
                .map(|(i, _)| i as u32)
                .collect();
            let obs: Vec<u32> = ac
                .observables
                .iter()
                .enumerate()
                .filter(|(_, recs)| parity(recs, &flips))
                .map(|(i, _)| i as u32)
                .collect();

            if dets.is_empty() && obs.is_empty() {
                Ok(None) // undetectable and logically trivial
            } else {
                Ok(Some((dets, obs, m.prob)))
            }
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();

    // Merge by (detectors, observables) support; value = combined probability.
    let mut edges: BTreeMap<(Vec<u32>, Vec<u32>), f64> = BTreeMap::new();
    for (dets, obs, prob) in supports {
        let e = edges.entry((dets, obs)).or_insert(0.0);
        *e = xor_combine(*e, prob);
    }

    let errors = edges
        .into_iter()
        .map(|((dets, obs), prob)| DemError { prob, dets, obs })
        .collect();
    Ok(DetectorErrorModel {
        detectors: ac.detectors.len(),
        observables: ac.observables.len(),
        errors,
    })
}

/// XOR-parity of the measurement records in `recs` under the flip vector.
fn parity(recs: &[usize], flips: &[bool]) -> bool {
    recs.iter().fold(false, |acc, &m| acc ^ flips[m])
}

/// Probability that an odd number of two independent events fire.
fn xor_combine(a: f64, b: f64) -> f64 {
    a + b - 2.0 * a * b
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_core::{Gate, GateInstance};
    use aleph_ir::Instruction;

    #[test]
    fn xor_combine_is_odd_parity() {
        assert!((xor_combine(0.0, 0.1) - 0.1).abs() < 1e-15);
        assert!((xor_combine(0.1, 0.1) - 0.18).abs() < 1e-15);
        assert!((xor_combine(0.5, 0.5) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn single_zz_check_dem() {
        // ZZ check: CNOT d0->anc, CNOT d1->anc, M anc. One detector = [m0].
        // X on d0 (prob p) and X on d1 (prob p) each flip the detector → merge.
        let mut c = Circuit::new(3, 1);
        c.add_instruction(Instruction::Gate(GateInstance::new(
            Gate::Cnot,
            vec![0u32, 2u32],
        )))
        .unwrap();
        c.add_instruction(Instruction::Gate(GateInstance::new(
            Gate::Cnot,
            vec![1u32, 2u32],
        )))
        .unwrap();
        c.measure(2, 0).unwrap();
        let ac = AnnotatedCircuit {
            circuit: c,
            detectors: vec![vec![0]],
            observables: vec![],
        };
        let mechs = vec![
            ErrorMechanism {
                prob: 0.1,
                x: vec![0],
                z: vec![],
                at: 0,
            },
            ErrorMechanism {
                prob: 0.1,
                x: vec![1],
                z: vec![],
                at: 0,
            },
            // A Z on d0 flips nothing → dropped.
            ErrorMechanism {
                prob: 0.1,
                x: vec![],
                z: vec![0],
                at: 0,
            },
        ];
        let dem = build_dem(&ac, &mechs).unwrap();
        assert_eq!(dem.detectors, 1);
        assert_eq!(dem.errors.len(), 1, "two X mechs merge, Z mech dropped");
        assert_eq!(dem.errors[0].dets, vec![0]);
        assert!((dem.errors[0].prob - xor_combine(0.1, 0.1)).abs() < 1e-12);
    }
}
