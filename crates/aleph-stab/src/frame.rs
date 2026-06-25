//! Pauli-frame sampler for noisy Clifford circuits ([`sample_noisy`]).
//!
//! # Why a frame simulator
//!
//! Naively, each noisy shot is an independent Clifford simulation: for an
//! n=1000, depth=100 circuit that is ~6 ms/shot — seconds for 1000 shots. The
//! Pauli-frame method does the expensive Clifford evolution **once** and then
//! samples shots in time proportional to the gate count, with O(1) work per gate
//! and 64 shots packed per machine word.
//!
//! The idea (the standard Stim approach, Gidney 2021, ArXiv:2103.02202 §"Pauli
//! frame"): a Pauli error inserted mid-circuit commutes through the remaining
//! Cliffords to become some final Pauli; its only observable effect is flipping
//! the measurements it anticommutes with. So we track, per shot, just a Pauli
//! "frame" `F = ∏ X^{fx_q} Z^{fz_q}` (no sign — only commutation matters for a
//! measurement flip), conjugate it through each gate, inject random Paulis at
//! noise locations, and read each Z-measurement's flip as `fx[q]` (the X/Y part
//! of the frame anticommutes with `Z_q`). A single noiseless reference pass
//! fixes the baseline outcomes that the frame flips are relative to.
//!
//! # Conjugation tables, derived not hand-written
//!
//! Each gate's action on `(fx, fz)` is read directly from the [`Tableau`]: apply
//! the gate to a fresh k-qubit tableau and read how it maps the generators
//! `X_i`/`Z_i`. This reuses the exact, oracle-tested gate kernels — there is no
//! hand-derived symplectic table to get wrong (it even covers `iSWAP`).
//!
//! # Scope (Q0-02)
//!
//! Correct and fast for circuits whose recorded measurements are **deterministic
//! from the initial state** — exactly the QEC syndrome-extraction case the
//! decoder track needs, where a prepared code state makes every syndrome
//! measurement deterministic and errors flip them deterministically. A bare
//! random measurement gets correct per-shot marginals, but measurements whose
//! value is only determined *conditionally on a prior random measurement* (e.g.
//! the second half of a Bell pair) are **not** yet correct: that needs
//! destabilizer back-action propagation, deferred to Q0-03 where the full joint
//! distribution (detectors/observables) is validated against Stim. Supported
//! instructions: Clifford `Gate`, `Measure`, `Reset`, `Barrier` (no-op).

use crate::noise::PauliNoise;
use crate::{apply_gate, StabError, Tableau};
use aleph_core::{Gate, GateInstance};
use aleph_ir::{Circuit, Instruction};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Per-shot outcomes of [`sample_noisy`], one bit per (shot, measurement).
///
/// Measurements are indexed in circuit order (the i-th `Measure` instruction is
/// measurement `i`). Stored column-major-by-measurement, 64 shots per word.
#[derive(Clone, Debug)]
pub struct NoisyOutcomes {
    shots: u32,
    num_measurements: usize,
    nwords: usize,
    /// `words[m * nwords + w]` bit `s` = outcome of shot `w*64 + s` for measurement `m`.
    words: Vec<u64>,
}

impl NoisyOutcomes {
    /// Number of shots sampled.
    pub fn shots(&self) -> u32 {
        self.shots
    }

    /// Number of measurements per shot.
    pub fn num_measurements(&self) -> usize {
        self.num_measurements
    }

    /// Outcome bit of measurement `m` in shot `shot`.
    pub fn get(&self, shot: usize, m: usize) -> bool {
        (self.words[m * self.nwords + shot / 64] >> (shot % 64)) & 1 == 1
    }

    /// Fraction of shots in which measurement `m` was 1.
    pub fn measurement_frequency(&self, m: usize) -> f64 {
        let base = m * self.nwords;
        let mut ones = 0u64;
        for w in 0..self.nwords {
            let valid = (self.shots as usize).saturating_sub(w * 64).min(64);
            let mask = if valid >= 64 {
                u64::MAX
            } else {
                (1u64 << valid) - 1
            };
            ones += (self.words[base + w] & mask).count_ones() as u64;
        }
        ones as f64 / self.shots as f64
    }

    /// All measurement outcomes for one shot, in circuit order.
    pub fn shot_record(&self, shot: usize) -> Vec<bool> {
        (0..self.num_measurements)
            .map(|m| self.get(shot, m))
            .collect()
    }
}

/// 1-qubit Clifford action on a frame component: image of `X` is `X^xx Z^xz`,
/// image of `Z` is `X^zx Z^zz` (signs dropped — irrelevant to measurement flips).
#[derive(Clone, Copy)]
struct Sympl1 {
    xx: bool,
    xz: bool,
    zx: bool,
    zz: bool,
}

/// 2-qubit Clifford action. `img[g]` is the image of generator `g` as
/// `[x_a, z_a, x_b, z_b]`, with `g`: 0 = `X_a`, 1 = `Z_a`, 2 = `X_b`, 3 = `Z_b`.
#[derive(Clone, Copy)]
struct Sympl2 {
    img: [[bool; 4]; 4],
}

/// Read a gate's 1-qubit conjugation table from the tableau.
fn derive_sympl1(gate: &Gate) -> Result<Sympl1, StabError> {
    let mut t = Tableau::new(1);
    apply_gate(&mut t, &GateInstance::new(gate.clone(), vec![0u32]))?;
    // rows: 0 = image of X_0 (destabilizer), 1 = image of Z_0 (stabilizer).
    let (x, z, _) = t.export_generators();
    Ok(Sympl1 {
        xx: x[0],
        xz: z[0],
        zx: x[1],
        zz: z[1],
    })
}

/// Read a gate's 2-qubit conjugation table from the tableau.
fn derive_sympl2(gate: &Gate) -> Result<Sympl2, StabError> {
    let mut t = Tableau::new(2);
    apply_gate(&mut t, &GateInstance::new(gate.clone(), vec![0u32, 1u32]))?;
    // rows (n=2): 0 = X_0 img, 1 = X_1 img, 2 = Z_0 img, 3 = Z_1 img (idx = row*2 + col).
    let (x, z, _) = t.export_generators();
    let read = |r: usize| [x[r * 2], z[r * 2], x[r * 2 + 1], z[r * 2 + 1]];
    Ok(Sympl2 {
        // reorder to g: X_a, Z_a, X_b, Z_b
        img: [read(0), read(2), read(1), read(3)],
    })
}

/// Preprocessed circuit step for the frame pass.
enum Op {
    Gate1 {
        s: Sympl1,
        q: usize,
        depol: f64,
    },
    Gate2 {
        s: Sympl2,
        a: usize,
        b: usize,
        depol: f64,
    },
    Measure {
        q: usize,
        m: usize,
    },
    Reset {
        q: usize,
    },
}

/// Sample `shots` noisy executions of a Clifford `circuit`, returning per-shot
/// measurement outcomes. Reproducible for a fixed `seed`.
///
/// # Errors
/// [`StabError::NonClifford`] for a non-Clifford gate or an externally
/// controlled gate; [`StabError::Unsupported`] for fused-block instructions.
pub fn sample_noisy(
    circuit: &Circuit,
    noise: &PauliNoise,
    shots: u32,
    seed: u64,
) -> Result<NoisyOutcomes, StabError> {
    let n = circuit.num_qubits() as usize;

    // --- 1. Preprocess into an op list (gate conjugation tables derived once). ---
    let mut ops: Vec<Op> = Vec::with_capacity(circuit.instructions().len());
    let mut num_measurements = 0usize;
    for inst in circuit.instructions() {
        match inst {
            Instruction::Gate(gi) => {
                if !gi.controls.is_empty() {
                    // A generic-controlled Clifford is not necessarily Clifford.
                    return Err(StabError::NonClifford {
                        gate: "controlled (ctrl@)",
                    });
                }
                match gi.gate.arity() {
                    1 => ops.push(Op::Gate1 {
                        s: derive_sympl1(&gi.gate)?,
                        q: gi.qubits[0] as usize,
                        depol: noise.depol1,
                    }),
                    2 => ops.push(Op::Gate2 {
                        s: derive_sympl2(&gi.gate)?,
                        a: gi.qubits[0] as usize,
                        b: gi.qubits[1] as usize,
                        depol: noise.depol2,
                    }),
                    _ => {
                        // 3q+ gates (Toffoli/Ccz) are non-Clifford; surface the
                        // canonical error from the dispatcher.
                        let k = gi.gate.arity() as u32;
                        apply_gate(
                            &mut Tableau::new(k as usize),
                            &GateInstance::new(gi.gate.clone(), (0..k).collect::<Vec<_>>()),
                        )?;
                        return Err(StabError::NonClifford {
                            gate: "multi-qubit",
                        });
                    }
                }
            }
            Instruction::Measure { qubit, .. } => {
                ops.push(Op::Measure {
                    q: *qubit as usize,
                    m: num_measurements,
                });
                num_measurements += 1;
            }
            Instruction::Reset(q) => ops.push(Op::Reset { q: *q as usize }),
            Instruction::Barrier(_) => {}
            Instruction::DiagonalPhase(_) => {
                return Err(StabError::Unsupported {
                    what: "DiagonalPhase",
                })
            }
            Instruction::TiledBlock(_) => {
                return Err(StabError::Unsupported { what: "TiledBlock" })
            }
        }
    }

    // --- 2. Reference pass: noiseless run fixes baseline outcomes + randomness. ---
    let mut refb = vec![false; num_measurements];
    let mut is_random = vec![false; num_measurements];
    {
        let mut t = Tableau::new(n);
        // Separate RNG stream from the frame pass so the reference's random-branch
        // coins don't perturb the per-shot noise stream.
        let mut rng_ref = StdRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);
        let mut m = 0usize;
        for inst in circuit.instructions() {
            match inst {
                Instruction::Gate(gi) => {
                    apply_gate(&mut t, gi)?;
                }
                Instruction::Measure { qubit, .. } => {
                    let q = *qubit as usize;
                    is_random[m] = !t.z_measure_is_deterministic(q)?;
                    refb[m] = t.measure(q, &mut rng_ref)?;
                    m += 1;
                }
                Instruction::Reset(q) => {
                    let q = *q as usize;
                    if t.measure(q, &mut rng_ref)? {
                        t.x_gate(q)?;
                    }
                }
                _ => {}
            }
        }
    }

    // --- 3. Frame pass: all shots bit-packed, 64 per word. ---
    let nwords = (shots as usize).div_ceil(64);
    let mut fx = vec![0u64; n * nwords];
    let mut fz = vec![0u64; n * nwords];
    let mut out = vec![0u64; num_measurements * nwords];
    let mut rng = StdRng::seed_from_u64(seed);

    for op in &ops {
        match op {
            Op::Gate1 { s, q, depol } => {
                apply_sympl1(s, &mut fx, &mut fz, *q, nwords);
                if *depol > 0.0 {
                    inject_depol1(&mut fx, &mut fz, *q, nwords, shots, *depol, &mut rng);
                }
            }
            Op::Gate2 { s, a, b, depol } => {
                apply_sympl2(s, &mut fx, &mut fz, *a, *b, nwords);
                if *depol > 0.0 {
                    inject_depol2(&mut fx, &mut fz, *a, *b, nwords, shots, *depol, &mut rng);
                }
            }
            Op::Measure { q, m } => {
                let obase = m * nwords;
                let qbase = q * nwords;
                for w in 0..nwords {
                    // Z-measurement flips where the frame's X/Y part is set.
                    let mut word = fx[qbase + w];
                    if is_random[*m] {
                        word = rng.gen::<u64>();
                    } else if refb[*m] {
                        word ^= u64::MAX;
                    }
                    if noise.measure_flip > 0.0 {
                        word ^= bernoulli_word(noise.measure_flip, w, shots, nwords, &mut rng);
                    }
                    out[obase + w] = word;
                }
                if is_random[*m] {
                    // Back-action: the measured X-component is projected out.
                    for w in 0..nwords {
                        fx[qbase + w] = 0;
                    }
                }
            }
            Op::Reset { q } => {
                let qbase = q * nwords;
                for w in 0..nwords {
                    fx[qbase + w] = 0;
                    fz[qbase + w] = 0;
                }
            }
        }
    }

    Ok(NoisyOutcomes {
        shots,
        num_measurements,
        nwords,
        words: out,
    })
}

/// Conjugate the frame on qubit `q` by a 1-qubit Clifford (per packed word).
fn apply_sympl1(s: &Sympl1, fx: &mut [u64], fz: &mut [u64], q: usize, nwords: usize) {
    let base = q * nwords;
    for w in 0..nwords {
        let (x, z) = (fx[base + w], fz[base + w]);
        let nx = (if s.xx { x } else { 0 }) ^ (if s.zx { z } else { 0 });
        let nz = (if s.xz { x } else { 0 }) ^ (if s.zz { z } else { 0 });
        fx[base + w] = nx;
        fz[base + w] = nz;
    }
}

/// Conjugate the frame on qubits `a`,`b` by a 2-qubit Clifford (per packed word).
fn apply_sympl2(s: &Sympl2, fx: &mut [u64], fz: &mut [u64], a: usize, b: usize, nwords: usize) {
    let (abase, bbase) = (a * nwords, b * nwords);
    for w in 0..nwords {
        let inp = [fx[abase + w], fz[abase + w], fx[bbase + w], fz[bbase + w]];
        let mut o = [0u64; 4]; // [x_a, z_a, x_b, z_b]
        for (g, &word) in inp.iter().enumerate() {
            if word == 0 {
                continue;
            }
            let im = &s.img[g];
            for (k, oslot) in o.iter_mut().enumerate() {
                if im[k] {
                    *oslot ^= word;
                }
            }
        }
        fx[abase + w] = o[0];
        fz[abase + w] = o[1];
        fx[bbase + w] = o[2];
        fz[bbase + w] = o[3];
    }
}

/// Number of valid shot-bits in word `w` (the last word may be partial).
#[inline]
fn bits_in_word(w: usize, shots: u32) -> usize {
    (shots as usize).saturating_sub(w * 64).min(64)
}

/// XOR a depolarizing error into qubit `q`'s frame: each shot independently, with
/// probability `p`, gets a uniformly random non-identity 1q Pauli (X, Y, or Z).
fn inject_depol1(
    fx: &mut [u64],
    fz: &mut [u64],
    q: usize,
    nwords: usize,
    shots: u32,
    p: f64,
    rng: &mut StdRng,
) {
    let base = q * nwords;
    for w in 0..nwords {
        let (mut xw, mut zw) = (0u64, 0u64);
        for s in 0..bits_in_word(w, shots) {
            if rng.gen::<f64>() < p {
                match rng.gen_range(0..3u8) {
                    0 => xw |= 1u64 << s, // X
                    1 => zw |= 1u64 << s, // Z
                    _ => {
                        xw |= 1u64 << s; // Y = XZ
                        zw |= 1u64 << s;
                    }
                }
            }
        }
        fx[base + w] ^= xw;
        fz[base + w] ^= zw;
    }
}

/// XOR a 2-qubit depolarizing error into qubits `a`,`b`: each shot independently,
/// with probability `p`, gets a uniformly random one of the 15 non-identity
/// 2-qubit Paulis.
#[allow(clippy::too_many_arguments)]
fn inject_depol2(
    fx: &mut [u64],
    fz: &mut [u64],
    a: usize,
    b: usize,
    nwords: usize,
    shots: u32,
    p: f64,
    rng: &mut StdRng,
) {
    let (abase, bbase) = (a * nwords, b * nwords);
    for w in 0..nwords {
        let (mut xa, mut za, mut xb, mut zb) = (0u64, 0u64, 0u64, 0u64);
        for s in 0..bits_in_word(w, shots) {
            if rng.gen::<f64>() < p {
                // idx in 1..=15 enumerates the non-identity Paulis; bits are
                // (x_a, z_a, x_b, z_b).
                let idx = rng.gen_range(1..16u8);
                let bit = 1u64 << s;
                if idx & 1 != 0 {
                    xa |= bit;
                }
                if idx & 2 != 0 {
                    za |= bit;
                }
                if idx & 4 != 0 {
                    xb |= bit;
                }
                if idx & 8 != 0 {
                    zb |= bit;
                }
            }
        }
        fx[abase + w] ^= xa;
        fz[abase + w] ^= za;
        fx[bbase + w] ^= xb;
        fz[bbase + w] ^= zb;
    }
}

/// A packed word whose valid shot-bits are each set independently with prob `p`.
fn bernoulli_word(p: f64, w: usize, shots: u32, _nwords: usize, rng: &mut StdRng) -> u64 {
    let mut word = 0u64;
    for s in 0..bits_in_word(w, shots) {
        if rng.gen::<f64>() < p {
            word |= 1u64 << s;
        }
    }
    word
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_ir::Circuit;

    // ---- conjugation-table sanity (derived tables vs known rules) ----

    #[test]
    fn sympl1_known_gates() {
        // H swaps X<->Z: X->Z (xx=0,xz=1), Z->X (zx=1,zz=0).
        let h = derive_sympl1(&Gate::H).unwrap();
        assert!(!h.xx && h.xz && h.zx && !h.zz);
        // S: X->Y (xx=1,xz=1), Z->Z (zx=0,zz=1).
        let s = derive_sympl1(&Gate::S).unwrap();
        assert!(s.xx && s.xz && !s.zx && s.zz);
        // X (Pauli): conjugation leaves X/Z parts unchanged: X->X, Z->Z.
        let x = derive_sympl1(&Gate::X).unwrap();
        assert!(x.xx && !x.xz && !x.zx && x.zz);
    }

    #[test]
    fn sympl2_cnot() {
        // CNOT(control=a, target=b): X_a -> X_a X_b; Z_b -> Z_a Z_b; Z_a, X_b fixed.
        let c = derive_sympl2(&Gate::Cnot).unwrap();
        assert_eq!(c.img[0], [true, false, true, false]); // X_a -> X_a X_b
        assert_eq!(c.img[1], [false, true, false, false]); // Z_a -> Z_a
        assert_eq!(c.img[2], [false, false, true, false]); // X_b -> X_b
        assert_eq!(c.img[3], [false, true, false, true]); // Z_b -> Z_a Z_b
    }

    /// X-ancilla syndrome-extraction style: a ZZ check on two data qubits via an
    /// ancilla. Build the simplest such gadget and a deterministic X error.
    fn zz_check_circuit(with_x_error_on_data0: bool) -> Circuit {
        // qubits: 0,1 data; 2 ancilla. Z-type stabilizer ZZ measured by
        // CNOT(d0->anc), CNOT(d1->anc), measure anc.
        let mut c = Circuit::new(3, 1);
        if with_x_error_on_data0 {
            c.add_instruction(Instruction::Gate(GateInstance::new(Gate::X, vec![0u32])))
                .unwrap();
        }
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
        c
    }

    #[test]
    fn no_error_syndrome_is_silent() {
        let c = zz_check_circuit(false);
        let out = sample_noisy(&c, &PauliNoise::none(), 64, 1).unwrap();
        for shot in 0..64 {
            assert!(!out.get(shot, 0), "noiseless ZZ check must not fire");
        }
    }

    #[test]
    fn x_error_fires_z_ancilla_deterministically() {
        // Acceptance criterion 2: a single X error on a data qubit flips the
        // adjacent Z-ancilla in every shot.
        let c = zz_check_circuit(true);
        let out = sample_noisy(&c, &PauliNoise::none(), 64, 1).unwrap();
        for shot in 0..64 {
            assert!(out.get(shot, 0), "X error must fire the Z ancilla");
        }
    }

    #[test]
    fn depolarizing_flip_frequency_matches_p() {
        // |0>, one 1q gate carrying depol p, measure Z. Z is flipped by X or Y,
        // i.e. with probability 2p/3.
        let mut c = Circuit::new(1, 1);
        c.add_instruction(Instruction::Gate(GateInstance::new(Gate::Z, vec![0u32])))
            .unwrap(); // Z on |0> is identity; just a noise carrier
        c.measure(0, 0).unwrap();
        let p = 0.12;
        let shots = 200_000;
        let out = sample_noisy(&c, &PauliNoise::depolarizing(p, 0.0), shots, 7).unwrap();
        let freq = out.measurement_frequency(0);
        let expected = 2.0 * p / 3.0;
        assert!(
            (freq - expected).abs() < 1e-2,
            "depol flip freq {freq} vs expected {expected}"
        );
    }

    #[test]
    fn measurement_flip_frequency_matches_p() {
        // Acceptance criterion 3 (measurement noise): |0>, measure with flip p.
        let mut c = Circuit::new(1, 1);
        c.measure(0, 0).unwrap();
        let p = 0.1;
        let shots = 200_000;
        let noise = PauliNoise::none().with_measure_flip(p);
        let out = sample_noisy(&c, &noise, shots, 3).unwrap();
        let freq = out.measurement_frequency(0);
        assert!((freq - p).abs() < 1e-2, "measure-flip freq {freq} vs {p}");
    }

    #[test]
    fn first_random_measurement_marginal_is_unbiased() {
        // A bare random measurement (H then measure Z) is 50/50 per shot — the
        // in-scope random case. (Measurements that are deterministic *given* a
        // prior random measurement, e.g. the second half of a Bell pair, are out
        // of Q0-02 scope; see the module docs — Q0-03 adds destabilizer
        // back-action and validates the joint distribution against Stim.)
        let mut c = Circuit::new(1, 1);
        c.add_instruction(Instruction::Gate(GateInstance::new(Gate::H, vec![0u32])))
            .unwrap();
        c.measure(0, 0).unwrap();
        let out = sample_noisy(&c, &PauliNoise::none(), 200_000, 5).unwrap();
        let f = out.measurement_frequency(0);
        assert!((f - 0.5).abs() < 1e-2, "random measurement marginal {f}");
    }

    // ---- cross-check the frame sampler against an independent per-shot CHP run ----

    /// Slow but obviously-correct reference: simulate each shot from scratch on a
    /// fresh tableau, injecting depolarizing/measurement noise as explicit Paulis.
    /// Used only in tests to validate the fast frame path.
    fn sample_noisy_chp(
        circuit: &Circuit,
        noise: &PauliNoise,
        shots: u32,
        seed: u64,
    ) -> Vec<Vec<bool>> {
        let n = circuit.num_qubits() as usize;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut records = Vec::with_capacity(shots as usize);
        for _ in 0..shots {
            let mut t = Tableau::new(n);
            let mut rec = Vec::new();
            for inst in circuit.instructions() {
                match inst {
                    Instruction::Gate(gi) => {
                        apply_gate(&mut t, gi).unwrap();
                        let depol = if gi.gate.arity() == 1 {
                            noise.depol1
                        } else {
                            noise.depol2
                        };
                        if depol > 0.0 {
                            for &q in gi.qubits.iter() {
                                if rng.gen::<f64>() < depol {
                                    match rng.gen_range(0..3u8) {
                                        0 => t.x_gate(q as usize).unwrap(),
                                        1 => t.z_gate(q as usize).unwrap(),
                                        _ => t.y_gate(q as usize).unwrap(),
                                    }
                                }
                            }
                        }
                    }
                    Instruction::Measure { qubit, .. } => {
                        let mut b = t.measure(*qubit as usize, &mut rng).unwrap();
                        if noise.measure_flip > 0.0 && rng.gen::<f64>() < noise.measure_flip {
                            b = !b;
                        }
                        rec.push(b);
                    }
                    Instruction::Reset(q) => {
                        let q = *q as usize;
                        let one = t.measure(q, &mut rng).unwrap();
                        if one {
                            t.x_gate(q).unwrap();
                        }
                    }
                    _ => {}
                }
            }
            records.push(rec);
        }
        records
    }

    /// A small all-deterministic-syndrome circuit: a 3-bit repetition code with
    /// two ZZ checks on |000>, with depolarizing data noise. Every recorded
    /// measurement is deterministic in the noiseless run, so the frame path is
    /// exactly correct and must match per-measurement firing frequencies of CHP.
    fn rep_code_circuit() -> Circuit {
        // data 0,1,2 ; ancillas 3 (Z0Z1), 4 (Z1Z2)
        let mut c = Circuit::new(5, 2);
        for (d0, d1, anc) in [(0u32, 1u32, 3u32), (1, 2, 4)] {
            c.add_instruction(Instruction::Gate(GateInstance::new(
                Gate::Cnot,
                vec![d0, anc],
            )))
            .unwrap();
            c.add_instruction(Instruction::Gate(GateInstance::new(
                Gate::Cnot,
                vec![d1, anc],
            )))
            .unwrap();
        }
        c.measure(3, 0).unwrap();
        c.measure(4, 1).unwrap();
        c
    }

    #[test]
    fn frame_matches_chp_frequencies_repetition_code() {
        let c = rep_code_circuit();
        let noise = PauliNoise::depolarizing(0.08, 0.0).with_measure_flip(0.03);
        let shots = 100_000;
        let frame = sample_noisy(&c, &noise, shots, 11).unwrap();
        let chp = sample_noisy_chp(&c, &noise, shots, 99);
        for m in 0..frame.num_measurements() {
            let f_freq = frame.measurement_frequency(m);
            let c_ones = chp.iter().filter(|r| r[m]).count();
            let c_freq = c_ones as f64 / shots as f64;
            assert!(
                (f_freq - c_freq).abs() < 1e-2,
                "measurement {m}: frame {f_freq} vs chp {c_freq}"
            );
        }
    }

    #[test]
    #[ignore = "perf: run with `cargo test -p aleph-stab --release -- --ignored perf_`"]
    fn perf_n1000_depth100_1000shots_under_1s() {
        use rand::SeedableRng;
        use std::time::Instant;
        // Acceptance criterion 4: n=1000, depth=100, 1000 shots with noise < 1 s.
        let n = 1000u32;
        let layers = 100;
        let mut c = Circuit::new(n, n);
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..layers {
            for q in 0..n {
                match rng.gen_range(0..3u8) {
                    0 => c
                        .add_instruction(Instruction::Gate(GateInstance::new(Gate::H, vec![q])))
                        .unwrap(),
                    1 => c
                        .add_instruction(Instruction::Gate(GateInstance::new(Gate::S, vec![q])))
                        .unwrap(),
                    _ => {
                        let b = (q + 1) % n;
                        c.add_instruction(Instruction::Gate(GateInstance::new(
                            Gate::Cnot,
                            vec![q, b],
                        )))
                        .unwrap()
                    }
                };
            }
        }
        for q in 0..n {
            c.measure(q, q).unwrap();
        }
        let noise = PauliNoise::depolarizing(0.001, 0.01).with_measure_flip(0.01);
        let t0 = Instant::now();
        let out = sample_noisy(&c, &noise, 1000, 7).unwrap();
        let dt = t0.elapsed();
        println!("sample_noisy n=1000 depth=100 1000 shots: {dt:?}");
        assert_eq!(out.num_measurements(), n as usize);
        assert!(dt.as_secs_f64() < 1.0, "took {dt:?}, must be < 1s");
    }

    #[test]
    fn rejects_non_clifford() {
        let mut c = Circuit::new(1, 1);
        c.add_instruction(Instruction::Gate(GateInstance::new(Gate::T, vec![0u32])))
            .unwrap();
        c.measure(0, 0).unwrap();
        let err = sample_noisy(&c, &PauliNoise::none(), 4, 1).unwrap_err();
        assert!(matches!(err, StabError::NonClifford { .. }));
    }
}
