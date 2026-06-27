//! Bivariate-bicycle (BB) / **gross** codes (Q5-01) — the qLDPC frontier (Bravyi et al.,
//! [arXiv:2308.07915](https://arxiv.org/abs/2308.07915)).
//!
//! A BB code is a CSS code built from two commuting polynomials over the group algebra of
//! `Z_ℓ × Z_m`. Let `x` and `y` be the cyclic shifts on the two factors (`xᵉ = I`, `yᵐ = I`,
//! `xy = yx`), each acting on `ℓm` cells. Pick `A = Σ xᵃyᵇ` and `B = Σ xᶜyᵈ` (three monomials each).
//! With `n = 2ℓm` qubits split into a left and a right block, the parity checks are
//!
//! ```text
//! H_X = [ A | B ]        H_Z = [ Bᵀ | Aᵀ ]
//! ```
//!
//! The CSS condition `H_X H_Zᵀ = A B + B A = 0 (mod 2)` holds automatically because `A` and `B`
//! commute. Every check has weight 6 (three monomials per polynomial), every qubit sits in 6 checks,
//! and crucially **each qubit's error lights 3 checks of one type** — the syndrome graph is a
//! *hypergraph*, not a matching graph, which is exactly why these codes need belief propagation
//! (Q3-02) and BP+OSD (Q5-02) rather than MWPM.
//!
//! The **gross code** `[[144, 12, 12]]` is `ℓ = 12, m = 6`, `A = x³ + y + y²`, `B = y³ + x + x²`.
//! [`BBCode::gross`] builds it; [`BBCode::n`]/[`BBCode::k`] verify `n = 144`, `k = 12` from the GF(2)
//! ranks of the checks (`d = 12` is from the paper — exact minimum distance of a `[[144,12]]` code is
//! intractable to recompute here).
//!
//! [`BBCode::code_capacity_dem`] emits a [`DetectorErrorModel`] for the standard code-capacity
//! benchmark (independent `Z` noise, the `X`-checks as detectors, the dual logical-`X` operators as
//! observables). Feed it to [`TannerGraph::new`](crate::TannerGraph) / `BpDecoder` for decoding.
//!
//! [`BBCode::circuit_level_dem`] (Q5-04) goes one level deeper: it lays down the **depth-7
//! syndrome-extraction circuit** of Bravyi et al. (the exact CNOT schedule from the authors'
//! reference implementation), runs a `rounds`-cycle **memory-`X`** experiment, and emits a DEM
//! under **circuit-level depolarizing noise** (faulty CNOTs, init, measurement, idle). Detectors
//! are the `X`-check round differences (plus a final difference reconstructed from a transversal
//! `X` readout); the model is the `Z`-error sector — the circuit-level analogue of
//! `code_capacity_dem`, directly comparable to the published BB-code thresholds (~0.7%).

use crate::builder::{build_dem, AnnotatedCircuit, ErrorMechanism};
use crate::dem::{DemError, DetectorErrorModel};
use crate::error::Result;
use aleph_core::{Gate, GateInstance};
use aleph_ir::{Circuit, Instruction};

/// A bivariate-bicycle CSS code over `Z_ℓ × Z_m`.
#[derive(Clone, Debug)]
pub struct BBCode {
    l: usize,
    m: usize,
    /// `X`-check rows of `H_X = [A | B]`: for each of the `ℓm` checks, the qubit indices it touches
    /// (`0..ℓm` left block, `ℓm..2ℓm` right block). Weight 6.
    hx_rows: Vec<Vec<usize>>,
    /// `Z`-check rows of `H_Z = [Bᵀ | Aᵀ]`. Weight 6.
    hz_rows: Vec<Vec<usize>>,
    /// Logical `Z` operators (a basis of `ker H_X / rowspace H_Z`), as `n`-bit qubit supports.
    lz: Vec<BitVec>,
    /// Logical `X` operators, the **symplectic dual** of `lz` (`lx[i]·lz[j] = δ_ij`).
    lx: Vec<BitVec>,
    /// Per `X`-check, its data-qubit neighbours **in monomial order** `[A₁,A₂,A₃,B₁,B₂,B₃]`
    /// (left-block qubits from `A`, right-block from `B`). The depth-7 syndrome schedule
    /// ([`SX`]) indexes into this, so the order matters (unlike the sorted [`Self::hx_rows`]).
    xchk_nbrs: Vec<Vec<usize>>,
    /// Per `Z`-check, its data-qubit neighbours in monomial order `[B₁,B₂,B₃,A₁,A₂,A₃]`
    /// (left-block from `Bᵀ`, right-block from `Aᵀ`). Indexed by [`SZ`].
    zchk_nbrs: Vec<Vec<usize>>,
}

impl BBCode {
    /// Build a BB code from `ℓ`, `m`, and the monomial exponent pairs `(a, b)` (meaning `xᵃ yᵇ`) of
    /// `A` and `B`. Computes the checks, verifies the CSS condition, and extracts a dual logical
    /// basis.
    ///
    /// # Panics
    /// If `ℓ == 0` or `m == 0`, or if the CSS condition `H_X H_Zᵀ = 0` fails (a malformed code).
    pub fn new(l: usize, m: usize, a_monos: &[(usize, usize)], b_monos: &[(usize, usize)]) -> Self {
        assert!(l > 0 && m > 0, "ℓ and m must be positive");
        let lm = l * m;
        let n = 2 * lm;

        // Cell index k = i*m + j with i in 0..ℓ, j in 0..m. A monomial xᵖyᵠ maps cell (i,j) to
        // ((i+p) mod ℓ, (j+q) mod m). Column support of M = forward map; row support = inverse map.
        let cell = |i: usize, j: usize| i * m + j;
        let coords = |k: usize| (k / m, k % m);
        let fwd = |monos: &[(usize, usize)], c: usize| -> Vec<usize> {
            let (ci, cj) = coords(c);
            monos
                .iter()
                .map(|&(p, q)| cell((ci + p) % l, (cj + q) % m))
                .collect()
        };
        let inv = |monos: &[(usize, usize)], r: usize| -> Vec<usize> {
            let (ri, rj) = coords(r);
            monos
                .iter()
                .map(|&(p, q)| cell((ri + l - p % l) % l, (rj + m - q % m) % m))
                .collect()
        };

        // Conventions follow the Bravyi et al. reference implementation so the depth-7 schedule
        // (which pairs X- and Z-check CNOTs by monomial index) measures both stabiliser types
        // without mutual disturbance. H_X check c (a row of [A|B]): A acts forward — qubit
        // `nonzero(A_k[c,:])` = `fwd`. H_Z check c (a row of [Bᵀ|Aᵀ]): the transpose acts backward —
        // qubit `nonzero(B_k[:,c])` = `inv`.
        let hx_rows: Vec<Vec<usize>> = (0..lm)
            .map(|c| {
                let mut row = fwd(a_monos, c);
                row.extend(fwd(b_monos, c).into_iter().map(|q| q + lm));
                row.sort_unstable();
                row
            })
            .collect();
        let hz_rows: Vec<Vec<usize>> = (0..lm)
            .map(|c| {
                let mut row = inv(b_monos, c);
                row.extend(inv(a_monos, c).into_iter().map(|q| q + lm));
                row.sort_unstable();
                row
            })
            .collect();

        // Bit-vector parity checks over n qubits.
        let hx: Vec<BitVec> = hx_rows.iter().map(|r| BitVec::from_iter(n, r)).collect();
        let hz: Vec<BitVec> = hz_rows.iter().map(|r| BitVec::from_iter(n, r)).collect();

        // CSS condition: every X-check commutes with every Z-check (even overlap).
        for x in &hx {
            for z in &hz {
                assert!(x.dot(z) == 0, "CSS condition H_X H_Zᵀ = 0 violated");
            }
        }

        // Logical Z = ker(H_X) mod rowspace(H_Z); logical X = ker(H_Z) mod rowspace(H_X).
        let lz = quotient_basis(&gf2_kernel(&hx, n), &hz, n);
        let lx_raw = quotient_basis(&gf2_kernel(&hz, n), &hx, n);
        let lx = symplectic_dualize(&lx_raw, &lz, n);

        // Monomial-ordered neighbour lists for the depth-7 schedule. These are the same qubit
        // sets as `hx_rows`/`hz_rows` (asserted in tests) but kept in the order the schedule
        // indexes: X-check c couples [A₁,A₂,A₃] (left) then [B₁,B₂,B₃] (right); Z-check c couples
        // [B₁,B₂,B₃] (left) then [A₁,A₂,A₃] (right). `inv`/`fwd` preserve the monomial order they
        // were given, matching the reference implementation (sbravyi/BivariateBicycleCodes).
        let xchk_nbrs: Vec<Vec<usize>> = (0..lm)
            .map(|c| {
                let mut v = fwd(a_monos, c);
                v.extend(fwd(b_monos, c).into_iter().map(|q| q + lm));
                v
            })
            .collect();
        let zchk_nbrs: Vec<Vec<usize>> = (0..lm)
            .map(|c| {
                let mut v = inv(b_monos, c);
                v.extend(inv(a_monos, c).into_iter().map(|q| q + lm));
                v
            })
            .collect();

        Self {
            l,
            m,
            hx_rows,
            hz_rows,
            lz,
            lx,
            xchk_nbrs,
            zchk_nbrs,
        }
    }

    /// The `[[144, 12, 12]]` **gross** code: `ℓ = 12, m = 6, A = x³ + y + y², B = y³ + x + x²`
    /// (Bravyi et al. Table 3).
    pub fn gross() -> Self {
        Self::new(12, 6, &[(3, 0), (0, 1), (0, 2)], &[(0, 3), (1, 0), (2, 0)])
    }

    /// Number of physical qubits `n = 2ℓm`.
    pub fn n(&self) -> usize {
        2 * self.l * self.m
    }

    /// Number of checks of each type (`ℓm`).
    pub fn num_checks(&self) -> usize {
        self.l * self.m
    }

    /// Number of logical qubits `k = n − rank(H_X) − rank(H_Z)` (= the size of the dual logical
    /// basis).
    pub fn k(&self) -> usize {
        self.lz.len()
    }

    /// `(ℓ, m)`.
    pub fn params(&self) -> (usize, usize) {
        (self.l, self.m)
    }

    /// `X`-check rows (`H_X = [A|B]`), each a sorted list of qubit indices.
    pub fn hx_rows(&self) -> &[Vec<usize>] {
        &self.hx_rows
    }

    /// `Z`-check rows (`H_Z = [Bᵀ|Aᵀ]`).
    pub fn hz_rows(&self) -> &[Vec<usize>] {
        &self.hz_rows
    }

    /// Code-capacity [`DetectorErrorModel`] for independent `Z` noise at physical rate `p`: one
    /// mechanism per qubit (a `Z` error), its detectors the `X`-checks that contain the qubit
    /// (3 of them — a hyperedge), its observables the dual logical-`X` operators it anticommutes
    /// with. Detectors are the `ℓm` `X`-checks; observables are the `k` logicals. This is the DEM
    /// `BpDecoder`/BP+OSD (Q5-02) decode; the `Z`-noise direction is decoded by `X`-checks, and the
    /// `X`-noise direction is the mirror image under `A ↔ B`.
    pub fn code_capacity_dem(&self, p: f64) -> DetectorErrorModel {
        let n = self.n();
        // For each qubit q: which X-checks contain it, and which logical-X operators cover it.
        let mut check_of_qubit: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (c, row) in self.hx_rows.iter().enumerate() {
            for &q in row {
                check_of_qubit[q].push(c as u32);
            }
        }
        let errors = (0..n)
            .map(|q| {
                let dets = check_of_qubit[q].clone();
                let obs: Vec<u32> = (0..self.lx.len())
                    .filter(|&o| self.lx[o].get(q))
                    .map(|o| o as u32)
                    .collect();
                DemError::new(p, dets, obs)
            })
            .collect();
        DetectorErrorModel {
            detectors: self.num_checks(),
            observables: self.lz.len(),
            errors,
        }
    }

    /// `X`-check `c`'s data-qubit neighbours in monomial order (see [`Self::xchk_nbrs`]).
    pub fn xcheck_neighbours(&self) -> &[Vec<usize>] {
        &self.xchk_nbrs
    }

    /// `Z`-check `c`'s data-qubit neighbours in monomial order (see [`Self::zchk_nbrs`]).
    pub fn zcheck_neighbours(&self) -> &[Vec<usize>] {
        &self.zchk_nbrs
    }

    /// Build the `rounds`-cycle **memory-`X`** syndrome-extraction experiment using the Bravyi
    /// depth-7 CNOT schedule ([`SX`]/[`SZ`]).
    ///
    /// Qubit layout: data `0..n` (left block `0..ℓm`, right block `ℓm..2ℓm`), `X`-check ancillas
    /// `n..n+ℓm`, `Z`-check ancillas `n+ℓm..n+2ℓm`. Data start in `|+⟩^n` (a `+1` eigenstate of
    /// every `X`-stabiliser and logical `X`), so the `X`-check syndromes and the logical-`X`
    /// observable are deterministic in the noiseless circuit — the regime [`build_dem`] and the
    /// frame sampler require. Each cycle: prepare ancillas, run the 7 CNOT rounds (`X`-ancilla
    /// controls data, data controls `Z`-ancilla), measure ancillas (`X`-checks in the `X` basis via
    /// `H`+measure, `Z`-checks in the `Z` basis); a final transversal `X` readout of the data closes
    /// the last detector.
    ///
    /// # Panics
    /// If `rounds == 0` or the code is not weight-6 (the depth-7 schedule needs three monomials
    /// per polynomial).
    pub fn memory_x_experiment(&self, rounds: usize) -> BBMemoryExperiment {
        assert!(rounds >= 1, "need at least one round");
        assert!(
            self.xchk_nbrs.iter().all(|v| v.len() == 6)
                && self.zchk_nbrs.iter().all(|v| v.len() == 6),
            "depth-7 schedule requires weight-6 checks (3 monomials per polynomial)"
        );
        let lm = self.num_checks();
        let n = self.n();
        let nq = n + 2 * lm;
        let xanc = |c: usize| (n + c) as u32;
        let zanc = |c: usize| (n + lm + c) as u32;

        let mut inst: Vec<Instruction> = Vec::new();
        let mut clbit = 0u32;
        let mut rec_count = 0usize; // running count of Measure instructions = next record index

        // Geometry of the circuit-level noise, rate-free (probabilities are applied later by
        // `circuit_level_mechanisms`). `(at, qubit[, qubit])` with `at` the instruction index the
        // error is inserted *before* (matching `ErrorMechanism::at`).
        let mut cnot_sites: Vec<(usize, u32, u32)> = Vec::new();
        let mut prep_sites: Vec<(usize, u32)> = Vec::new();
        let mut meas_sites: Vec<(usize, u32)> = Vec::new();
        let mut idle_sites: Vec<(usize, u32)> = Vec::new();

        let measure = |inst: &mut Vec<Instruction>,
                       rec_count: &mut usize,
                       clbit: &mut u32,
                       q: u32|
         -> usize {
            let rec = *rec_count;
            *rec_count += 1;
            inst.push(Instruction::Measure {
                qubit: q,
                clbit: *clbit,
            });
            *clbit += 1;
            rec
        };

        // Initial data |+⟩^n: reset + H, with a basis-flip (Z) prep error after each H.
        for q in 0..n as u32 {
            inst.push(Instruction::Reset(q));
            inst.push(Instruction::Gate(GateInstance::new(Gate::H, vec![q])));
            prep_sites.push((inst.len(), q));
        }

        // xrec[cycle][c] = measurement-record index of X-check c in that cycle.
        let mut xrec: Vec<Vec<usize>> = Vec::with_capacity(rounds);

        for _cycle in 0..rounds {
            // Prepare ancillas: X-checks in |+⟩ (reset+H, Z prep error), Z-checks in |0⟩ (reset;
            // its X prep error is the X-error sector, not modelled by this Z-sector DEM).
            for c in 0..lm {
                inst.push(Instruction::Reset(xanc(c)));
                inst.push(Instruction::Gate(GateInstance::new(Gate::H, vec![xanc(c)])));
                prep_sites.push((inst.len(), xanc(c)));
            }
            for c in 0..lm {
                inst.push(Instruction::Reset(zanc(c)));
            }

            // 7 CNOT rounds. Within a round the X-check and Z-check CNOTs act on disjoint qubits
            // (the depth-7 property, asserted by `depth7_schedule_is_conflict_free`), so emission
            // order within a round is immaterial. The *measurement* staggering is not: following the
            // reference, the Z-checks are measured at round 6 **before** that round's X-check CNOTs.
            // Measuring Z later (after the round-6 X-CNOTs) would let an `X` spread by a round-6
            // X-CNOT be pulled back as a `Z` hook onto an X-ancilla at the Z-measurement, disturbing
            // the X-stabilisers. Idle data qubits in a round take an idle (Z) error at the round
            // start.
            for t in 0..7 {
                // round 6 (sZ idle): measure Z-checks before the round's X-check CNOTs.
                if SZ[t].is_none() {
                    for c in 0..lm {
                        measure(&mut inst, &mut rec_count, &mut clbit, zanc(c));
                        inst.push(Instruction::Reset(zanc(c)));
                    }
                }
                let round_start = inst.len();
                let mut data_touched = vec![false; n];
                if let Some(d) = SX[t] {
                    for c in 0..lm {
                        let tgt = self.xchk_nbrs[c][d] as u32;
                        inst.push(Instruction::Gate(GateInstance::new(
                            Gate::Cnot,
                            vec![xanc(c), tgt],
                        )));
                        cnot_sites.push((inst.len(), xanc(c), tgt));
                        data_touched[tgt as usize] = true;
                    }
                }
                if let Some(d) = SZ[t] {
                    for c in 0..lm {
                        let ctrl = self.zchk_nbrs[c][d] as u32;
                        inst.push(Instruction::Gate(GateInstance::new(
                            Gate::Cnot,
                            vec![ctrl, zanc(c)],
                        )));
                        cnot_sites.push((inst.len(), ctrl, zanc(c)));
                        data_touched[ctrl as usize] = true;
                    }
                }
                for (q, &hit) in data_touched.iter().enumerate() {
                    if !hit {
                        idle_sites.push((round_start, q as u32));
                    }
                }
            }

            // Measure X-checks in the X basis (Z measurement-flip error before H). Reset returns each
            // ancilla to |0⟩ for the next cycle's prep.
            let mut this_x = Vec::with_capacity(lm);
            for c in 0..lm {
                meas_sites.push((inst.len(), xanc(c)));
                inst.push(Instruction::Gate(GateInstance::new(Gate::H, vec![xanc(c)])));
                this_x.push(measure(&mut inst, &mut rec_count, &mut clbit, xanc(c)));
                inst.push(Instruction::Reset(xanc(c)));
            }
            xrec.push(this_x);
        }

        // Final transversal X readout of the data (Z measurement-flip error before H).
        let mut data_rec = vec![0usize; n];
        #[allow(clippy::needless_range_loop)] // q is the data qubit index, used throughout the body
        for q in 0..n {
            meas_sites.push((inst.len(), q as u32));
            inst.push(Instruction::Gate(GateInstance::new(
                Gate::H,
                vec![q as u32],
            )));
            data_rec[q] = measure(&mut inst, &mut rec_count, &mut clbit, q as u32);
        }

        // Detectors: X-check round differences (round 0 is the raw outcome; deterministic in |+⟩),
        // then a final block reconstructing each X-stabiliser from the data readout XOR the last
        // ancilla round.
        let mut detectors: Vec<Vec<usize>> = Vec::with_capacity((rounds + 1) * lm);
        for cycle in 0..rounds {
            for (c, &rec) in xrec[cycle].iter().enumerate() {
                if cycle == 0 {
                    detectors.push(vec![rec]);
                } else {
                    detectors.push(vec![rec, xrec[cycle - 1][c]]);
                }
            }
        }
        for (c, hx_row) in self.hx_rows.iter().enumerate() {
            let mut recs: Vec<usize> = hx_row.iter().map(|&q| data_rec[q]).collect();
            recs.push(xrec[rounds - 1][c]);
            detectors.push(recs);
        }

        // Observables: logical-X operators reconstructed from the transversal X readout. A logical
        // Z error (the error sector this DEM tracks) anticommutes with logical X and flips it.
        let observables: Vec<Vec<usize>> = self
            .lx
            .iter()
            .map(|lx| (0..n).filter(|&q| lx.get(q)).map(|q| data_rec[q]).collect())
            .collect();

        let circuit = {
            let mut c = Circuit::new(nq as u32, clbit.max(1));
            for i in inst {
                c.add_instruction(i).expect("valid instruction");
            }
            c
        };

        BBMemoryExperiment {
            annotated: AnnotatedCircuit {
                circuit,
                detectors,
                observables,
            },
            rounds,
            num_qubits: nq,
            num_data: n,
            num_checks: lm,
            cnot_sites,
            prep_sites,
            meas_sites,
            idle_sites,
        }
    }

    /// Circuit-level [`DetectorErrorModel`] for the gross code: the `Z`-error sector of a
    /// `rounds`-cycle memory-`X` experiment under depth-7 syndrome extraction with `noise`.
    ///
    /// This is the circuit-level analogue of [`Self::code_capacity_dem`]: feed it to `BpDecoder` /
    /// `OsdDecoder` exactly the same way. Convention `rounds = code distance` for a fair memory
    /// benchmark.
    ///
    /// # Errors
    /// Propagates [`crate::Error::Propagation`] from the underlying Pauli propagation.
    pub fn circuit_level_dem(
        &self,
        rounds: usize,
        noise: CircuitNoise,
    ) -> Result<DetectorErrorModel> {
        let exp = self.memory_x_experiment(rounds);
        build_dem(&exp.annotated, &exp.circuit_level_mechanisms(noise))
    }
}

/// Depth-7 CNOT schedule of Bravyi et al. ([arXiv:2308.07915](https://arxiv.org/abs/2308.07915)),
/// transcribed from the authors' reference implementation
/// ([sbravyi/BivariateBicycleCodes](https://github.com/sbravyi/BivariateBicycleCodes),
/// `decoder_setup.py`: `sX = ['idle',1,4,3,5,0,2]`). `SX[t]` is the monomial-neighbour index
/// (`0..6` into [`BBCode::xchk_nbrs`]) each `X`-check couples to in round `t`, or `None` for the
/// idle slot. Over the 7 rounds every check touches all six of its neighbours exactly once.
const SX: [Option<usize>; 7] = [None, Some(1), Some(4), Some(3), Some(5), Some(0), Some(2)];
/// `Z`-check half of the schedule (`sZ = [3,5,0,1,2,4,'idle']`); indexes [`BBCode::zchk_nbrs`].
const SZ: [Option<usize>; 7] = [Some(3), Some(5), Some(0), Some(1), Some(2), Some(4), None];

/// Circuit-level depolarizing noise strengths for [`BBCode::circuit_level_dem`].
///
/// Following Bravyi et al.'s model, each source contributes its `Z`-component to the `Z`-error
/// sector: a two-qubit depolarizing channel after every CNOT (its 15 non-identity Paulis split so
/// each of `Z⊗I`, `I⊗Z`, `Z⊗Z` appears with weight `4/15`), a single-qubit depolarizing channel on
/// idle qubits (`Z` with weight `2/3`), and basis-flip errors at preparation and measurement.
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
    /// The standard uniform circuit-level model: every source at the same physical rate `p`
    /// (Bravyi et al.'s `error_rate`).
    pub fn uniform(p: f64) -> Self {
        Self {
            p_cnot: p,
            p_init: p,
            p_meas: p,
            p_idle: p,
        }
    }
}

/// A built `rounds`-cycle memory-`X` experiment on a BB code: the annotated Clifford circuit plus
/// the rate-free geometry of the circuit-level noise, from which [`Self::circuit_level_mechanisms`]
/// produces the [`ErrorMechanism`]s and [`Self::stim_program`] an equivalent Stim program.
#[derive(Clone, Debug)]
pub struct BBMemoryExperiment {
    /// Circuit + detector / observable definitions.
    pub annotated: AnnotatedCircuit,
    /// Number of syndrome-extraction cycles.
    pub rounds: usize,
    /// Total qubits (data + both ancilla blocks).
    pub num_qubits: usize,
    num_data: usize,
    num_checks: usize,
    /// `(at, control, target)` for every CNOT — site of a two-qubit depolarizing channel.
    cnot_sites: Vec<(usize, u32, u32)>,
    /// `(at, qubit)` for every `X`-basis preparation (data + `X`-ancillas).
    prep_sites: Vec<(usize, u32)>,
    /// `(at, qubit)` for every `X`-basis measurement (data + `X`-ancillas).
    meas_sites: Vec<(usize, u32)>,
    /// `(at, qubit)` for every idle data qubit in a CNOT round.
    idle_sites: Vec<(usize, u32)>,
}

impl BBMemoryExperiment {
    /// The `Z`-error-sector [`ErrorMechanism`]s for circuit-level `noise`. Each CNOT contributes
    /// `Z(c)`, `Z(t)`, `Z(c)Z(t)` at `4/15·p_cnot`; each idle data qubit a `Z` at `2/3·p_idle`;
    /// each `X`-basis prep/measurement a `Z` at `p_init`/`p_meas`. [`build_dem`] propagates these,
    /// drops the ones that flip no `X`-check detector and no observable (e.g. a `Z` on a `Z`-check
    /// ancilla), and merges the rest.
    pub fn circuit_level_mechanisms(&self, noise: CircuitNoise) -> Vec<ErrorMechanism> {
        let mut mechs = Vec::new();
        let z1 = |prob: f64, q: u32, at: usize| ErrorMechanism {
            prob,
            x: vec![],
            z: vec![q],
            at,
        };
        let cnot_w = noise.p_cnot * 4.0 / 15.0;
        for &(at, c, t) in &self.cnot_sites {
            mechs.push(z1(cnot_w, c, at));
            mechs.push(z1(cnot_w, t, at));
            mechs.push(ErrorMechanism {
                prob: cnot_w,
                x: vec![],
                z: vec![c, t],
                at,
            });
        }
        let idle_w = noise.p_idle * 2.0 / 3.0;
        for &(at, q) in &self.idle_sites {
            mechs.push(z1(idle_w, q, at));
        }
        for &(at, q) in &self.prep_sites {
            mechs.push(z1(noise.p_init, q, at));
        }
        for &(at, q) in &self.meas_sites {
            mechs.push(z1(noise.p_meas, q, at));
        }
        mechs
    }

    /// Number of `X`-check detectors (`(rounds + 1)·ℓm`).
    pub fn num_detectors(&self) -> usize {
        self.annotated.detectors.len()
    }

    /// Number of data qubits (`n = 2ℓm`).
    pub fn num_data(&self) -> usize {
        self.num_data
    }

    /// Round index (time coordinate) of every detector, for sliding-window decoding: the round-`r`
    /// difference detectors live at time `r ∈ 0..rounds`; the final readout block at time `rounds`.
    pub fn detector_rounds(&self) -> Vec<usize> {
        let nc = self.num_checks;
        (0..self.annotated.detectors.len())
            .map(|d| (d / nc).min(self.rounds))
            .collect()
    }

    /// Emit an equivalent Stim program with the same `Z`-sector circuit-level noise, so Stim's
    /// `detector_error_model` can be cross-checked against [`build_dem`]. Detectors and observables
    /// are emitted in the same order as [`AnnotatedCircuit`], using `rec[-k]` relative indexing, so
    /// Stim's `D{i}`/`L{i}` match ours edge-for-edge.
    pub fn stim_program(&self, noise: CircuitNoise) -> String {
        let ac = &self.annotated;
        let insts = ac.circuit.instructions();
        // Map each Measure instruction index -> its record number, then to rec[-k].
        let total_recs = insts
            .iter()
            .filter(|i| matches!(i, Instruction::Measure { .. }))
            .count();
        let rel = |rec: usize| format!("rec[-{}]", total_recs - rec);

        // Group rate-free noise sites by the instruction index they precede, so we can splat them
        // in while re-walking the circuit.
        let cnot_w = noise.p_cnot * 4.0 / 15.0;
        let idle_w = noise.p_idle * 2.0 / 3.0;
        let mut emit_before: std::collections::BTreeMap<usize, Vec<String>> =
            std::collections::BTreeMap::new();
        for &(at, c, t) in &self.cnot_sites {
            let e = emit_before.entry(at).or_default();
            e.push(format!("E({cnot_w}) Z{c}"));
            e.push(format!("E({cnot_w}) Z{t}"));
            e.push(format!("E({cnot_w}) Z{c} Z{t}"));
        }
        for &(at, q) in &self.idle_sites {
            emit_before
                .entry(at)
                .or_default()
                .push(format!("E({idle_w}) Z{q}"));
        }
        for &(at, q) in &self.prep_sites {
            emit_before
                .entry(at)
                .or_default()
                .push(format!("E({}) Z{q}", noise.p_init));
        }
        for &(at, q) in &self.meas_sites {
            emit_before
                .entry(at)
                .or_default()
                .push(format!("E({}) Z{q}", noise.p_meas));
        }

        let mut s = String::new();
        for (i, inst) in insts.iter().enumerate() {
            if let Some(errs) = emit_before.get(&i) {
                for e in errs {
                    s.push_str(e);
                    s.push('\n');
                }
            }
            match inst {
                Instruction::Reset(q) => s.push_str(&format!("R {q}\n")),
                Instruction::Gate(gi) => match gi.gate {
                    Gate::H => s.push_str(&format!("H {}\n", gi.qubits[0])),
                    Gate::Cnot => s.push_str(&format!("CX {} {}\n", gi.qubits[0], gi.qubits[1])),
                    _ => {}
                },
                Instruction::Measure { qubit, .. } => s.push_str(&format!("M {qubit}\n")),
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
}

/// A fixed-width GF(2) bit vector over `Vec<u64>` words.
#[derive(Clone, Debug, PartialEq, Eq)]
struct BitVec {
    words: Vec<u64>,
}

impl BitVec {
    fn zeros(nbits: usize) -> Self {
        Self {
            words: vec![0u64; nbits.div_ceil(64)],
        }
    }
    fn from_iter(nbits: usize, set: &[usize]) -> Self {
        let mut v = Self::zeros(nbits);
        for &b in set {
            v.set(b);
        }
        v
    }
    #[inline]
    fn get(&self, i: usize) -> bool {
        (self.words[i / 64] >> (i % 64)) & 1 == 1
    }
    #[inline]
    fn set(&mut self, i: usize) {
        self.words[i / 64] |= 1u64 << (i % 64);
    }
    #[inline]
    fn xor_assign(&mut self, other: &Self) {
        for (a, b) in self.words.iter_mut().zip(&other.words) {
            *a ^= b;
        }
    }
    /// GF(2) inner product (parity of the overlap).
    #[inline]
    fn dot(&self, other: &Self) -> u32 {
        self.words
            .iter()
            .zip(&other.words)
            .fold(0u32, |acc, (a, b)| acc ^ (a & b).count_ones())
            & 1
    }
    /// Highest set bit index, or `None` if zero.
    fn leading(&self) -> Option<usize> {
        for (wi, &w) in self.words.iter().enumerate().rev() {
            if w != 0 {
                return Some(wi * 64 + (63 - w.leading_zeros() as usize));
            }
        }
        None
    }
}

/// Basis of the null space `{v : M v = 0}` over GF(2), `M` given by its rows, `ncols` wide.
fn gf2_kernel(mat: &[BitVec], ncols: usize) -> Vec<BitVec> {
    let mut rows: Vec<BitVec> = mat.to_vec();
    // `pivot_row_of_col[c]` = the row whose pivot column is `c`, after reduction (or MAX).
    let mut pivot_row_of_col = vec![usize::MAX; ncols];
    let mut r = 0usize;
    // `c` is a bit index into each `BitVec` row (not a slice index), so enumerate() does not apply.
    #[allow(clippy::needless_range_loop)]
    for c in 0..ncols {
        if r >= rows.len() {
            break;
        }
        if let Some(pr) = (r..rows.len()).find(|&i| rows[i].get(c)) {
            rows.swap(r, pr);
            for i in 0..rows.len() {
                if i != r && rows[i].get(c) {
                    let pivot = rows[r].clone();
                    rows[i].xor_assign(&pivot);
                }
            }
            pivot_row_of_col[c] = r;
            r += 1;
        }
    }
    // Each free column → one kernel basis vector.
    let mut ker = Vec::new();
    for f in 0..ncols {
        if pivot_row_of_col[f] != usize::MAX {
            continue;
        }
        let mut v = BitVec::zeros(ncols);
        v.set(f);
        for (c, &pr) in pivot_row_of_col.iter().enumerate() {
            if pr != usize::MAX && rows[pr].get(f) {
                v.set(c);
            }
        }
        ker.push(v);
    }
    ker
}

/// Reduce `vectors` modulo `rowspace(span)` and return a basis of the quotient (the new independent
/// directions) — used to peel logical operators out of `ker(H) / rowspace(H')`.
fn quotient_basis(vectors: &[BitVec], span: &[BitVec], ncols: usize) -> Vec<BitVec> {
    let mut echelon = Echelon::new(ncols);
    for s in span {
        echelon.insert(s.clone());
    }
    let mut logicals = Vec::new();
    for v in vectors {
        if let Some(reduced) = echelon.reduce_nonzero(v.clone()) {
            echelon.insert(reduced.clone());
            logicals.push(reduced);
        }
    }
    logicals
}

/// GF(2) row-echelon span keyed by leading set-bit, for membership / reduction queries.
struct Echelon {
    /// `by_leading[i]` = a basis vector whose leading bit is `i` (or `None`).
    by_leading: Vec<Option<BitVec>>,
}

impl Echelon {
    fn new(ncols: usize) -> Self {
        Self {
            by_leading: vec![None; ncols],
        }
    }
    /// Reduce `v` against the span; return the residue if it is independent (nonzero), else `None`.
    fn reduce_nonzero(&self, mut v: BitVec) -> Option<BitVec> {
        while let Some(lead) = v.leading() {
            match &self.by_leading[lead] {
                Some(b) => v.xor_assign(b),
                None => return Some(v),
            }
        }
        None
    }
    /// Insert `v` (already reduced is fine; this re-reduces) into the span.
    fn insert(&mut self, mut v: BitVec) {
        while let Some(lead) = v.leading() {
            match &self.by_leading[lead] {
                Some(b) => v.xor_assign(b),
                None => {
                    self.by_leading[lead] = Some(v);
                    return;
                }
            }
        }
    }
}

/// Replace `lx_raw` by a symplectic **dual** basis of `lz`: returns `lx` with `lx[i]·lz[j] = δ_ij`.
/// Solves `G · lx_new = lx_raw` where `G[i][j] = lx_raw[i]·lz[j]` (invertible for nondegenerate
/// logicals), i.e. `lx_new = G⁻¹ lx_raw`.
fn symplectic_dualize(lx_raw: &[BitVec], lz: &[BitVec], ncols: usize) -> Vec<BitVec> {
    let k = lz.len();
    assert_eq!(lx_raw.len(), k, "logical X/Z counts must match");
    if k == 0 {
        return Vec::new();
    }
    // G[i][j] = lx_raw[i] · lz[j].
    let mut g: Vec<BitVec> = (0..k)
        .map(|i| {
            let mut row = BitVec::zeros(k);
            for (j, lzj) in lz.iter().enumerate() {
                if lx_raw[i].dot(lzj) == 1 {
                    row.set(j);
                }
            }
            row
        })
        .collect();
    // Invert G over GF(2) via Gauss-Jordan on [G | I].
    let mut inv: Vec<BitVec> = (0..k)
        .map(|i| BitVec::from_iter(k, &[i]))
        .collect::<Vec<_>>();
    for col in 0..k {
        let pivot = (col..k)
            .find(|&r| g[r].get(col))
            .expect("logicals nondegenerate ⇒ G invertible");
        g.swap(col, pivot);
        inv.swap(col, pivot);
        for r in 0..k {
            if r != col && g[r].get(col) {
                let (gp, ip) = (g[col].clone(), inv[col].clone());
                g[r].xor_assign(&gp);
                inv[r].xor_assign(&ip);
            }
        }
    }
    // lx_new[i] = Σ_j inv[i][j] · lx_raw[j].
    (0..k)
        .map(|i| {
            let mut v = BitVec::zeros(ncols);
            for (j, lxj) in lx_raw.iter().enumerate() {
                if inv[i].get(j) {
                    v.xor_assign(lxj);
                }
            }
            v
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gross_code_has_expected_parameters() {
        let code = BBCode::gross();
        assert_eq!(code.params(), (12, 6));
        assert_eq!(code.n(), 144, "n = 2ℓm");
        assert_eq!(code.num_checks(), 72, "ℓm X-checks and ℓm Z-checks");
        assert_eq!(code.k(), 12, "gross code encodes 12 logical qubits");
    }

    #[test]
    fn checks_are_weight_six() {
        let code = BBCode::gross();
        for row in code.hx_rows() {
            assert_eq!(row.len(), 6, "every X-check has weight 6");
        }
        for row in code.hz_rows() {
            assert_eq!(row.len(), 6, "every Z-check has weight 6");
        }
        // Each qubit sits in exactly 3 X-checks (column weight 3 ⇒ hyperedges, not matching).
        let mut col_weight = vec![0usize; code.n()];
        for row in code.hx_rows() {
            for &q in row {
                col_weight[q] += 1;
            }
        }
        assert!(col_weight.iter().all(|&w| w == 3));
    }

    #[test]
    fn logicals_are_dual_and_nontrivial() {
        let code = BBCode::gross();
        // Dual basis: lx[i]·lz[j] = δ_ij.
        for i in 0..code.k() {
            for j in 0..code.k() {
                let expect = u32::from(i == j);
                assert_eq!(
                    code.lx[i].dot(&code.lz[j]),
                    expect,
                    "lx[{i}]·lz[{j}] must be δ"
                );
            }
        }
        // Logicals commute with the stabilizers: lz ∈ ker(H_X), lx ∈ ker(H_Z).
        let n = code.n();
        let hx: Vec<BitVec> = code
            .hx_rows()
            .iter()
            .map(|r| BitVec::from_iter(n, r))
            .collect();
        let hz: Vec<BitVec> = code
            .hz_rows()
            .iter()
            .map(|r| BitVec::from_iter(n, r))
            .collect();
        for lz in &code.lz {
            assert!(
                hx.iter().all(|c| c.dot(lz) == 0),
                "logical Z must commute with X-checks"
            );
        }
        for lx in &code.lx {
            assert!(
                hz.iter().all(|c| c.dot(lx) == 0),
                "logical X must commute with Z-checks"
            );
        }
    }

    #[test]
    fn code_capacity_dem_is_well_formed() {
        let code = BBCode::gross();
        let dem = code.code_capacity_dem(0.01);
        assert_eq!(dem.detectors, 72);
        assert_eq!(dem.observables, 12);
        assert_eq!(dem.errors.len(), 144, "one Z-error mechanism per qubit");
        // Each mechanism is a 3-detector hyperedge.
        assert!(dem.errors.iter().all(|e| e.dets.len() == 3));
        // A single qubit's error must flip at least one observable for some qubit (logicals cover
        // the block) and the DEM is non-degenerate overall.
        let total_obs: usize = dem.errors.iter().map(|e| e.obs.len()).sum();
        assert!(
            total_obs > 0,
            "logical observables must be reachable by single-qubit errors"
        );
    }

    /// A smaller BB code `[[72,12,6]]` (ℓ=m=6, same polynomials) also verifies — guards the
    /// construction against hard-coding the gross parameters.
    #[test]
    fn small_bb_code_parameters() {
        let code = BBCode::new(6, 6, &[(3, 0), (0, 1), (0, 2)], &[(0, 3), (1, 0), (2, 0)]);
        assert_eq!(code.n(), 72);
        assert_eq!(code.k(), 12);
    }

    /// The monomial-ordered neighbour lists are the same qubit sets as the (sorted) parity-check
    /// rows — i.e. the schedule drives the *same* stabilisers, just in a controlled order.
    #[test]
    fn schedule_neighbours_match_check_rows() {
        let code = BBCode::gross();
        for c in 0..code.num_checks() {
            let mut xs = code.xchk_nbrs[c].clone();
            xs.sort_unstable();
            assert_eq!(xs, code.hx_rows[c], "X-check {c} neighbours");
            let mut zs = code.zchk_nbrs[c].clone();
            zs.sort_unstable();
            assert_eq!(zs, code.hz_rows[c], "Z-check {c} neighbours");
            assert_eq!(code.xchk_nbrs[c].len(), 6);
            assert_eq!(code.zchk_nbrs[c].len(), 6);
        }
    }

    /// The Bravyi depth-7 property: in every round, every physical qubit is involved in at most one
    /// CNOT — the X-check CNOTs (ancilla→data) and Z-check CNOTs (data→ancilla) never collide. This
    /// is what makes the schedule depth-7 rather than depth-12, and it depends on the monomial order
    /// matching the reference. Also checks each check touches all six neighbours across the rounds.
    #[test]
    fn depth7_schedule_is_conflict_free() {
        for l in [6usize, 12] {
            let code = BBCode::new(l, 6, &[(3, 0), (0, 1), (0, 2)], &[(0, 3), (1, 0), (2, 0)]);
            let lm = code.num_checks();
            let n = code.n();
            let mut x_hits = vec![0usize; lm]; // CNOTs each X-check performs
            let mut z_hits = vec![0usize; lm];
            for t in 0..7 {
                let mut used = std::collections::HashSet::new();
                let mut insert = |q: usize| {
                    assert!(
                        used.insert(q),
                        "ℓ={l} round {t}: qubit {q} used twice (schedule conflict)"
                    );
                };
                #[allow(clippy::needless_range_loop)] // c is the check index, used several ways
                if let Some(d) = SX[t] {
                    for c in 0..lm {
                        insert(n + c); // X-ancilla
                        insert(code.xchk_nbrs[c][d]); // data target
                        x_hits[c] += 1;
                    }
                }
                #[allow(clippy::needless_range_loop)]
                if let Some(d) = SZ[t] {
                    for c in 0..lm {
                        insert(n + lm + c); // Z-ancilla
                        insert(code.zchk_nbrs[c][d]); // data control
                        z_hits[c] += 1;
                    }
                }
            }
            assert!(
                x_hits.iter().all(|&h| h == 6),
                "ℓ={l}: every X-check makes 6 CNOTs"
            );
            assert!(
                z_hits.iter().all(|&h| h == 6),
                "ℓ={l}: every Z-check makes 6 CNOTs"
            );
        }
    }

    /// The memory-X experiment has the expected shape: qubit count `n + 2ℓm`, detector count
    /// `(rounds+1)·ℓm`, `k` observables, and a CNOT count matching the depth-7 schedule (each of the
    /// `2ℓm` checks makes 6 CNOTs per cycle). Determinism of the detectors and full DEM correctness
    /// are gated by the Stim oracle (`tests/bb_circuit_dem_stim_oracle.rs`): the noiseless circuit
    /// here has genuinely random measurements (Z-ancillas on |+⟩^n, transversal X readout), so the
    /// Pauli-frame sampler does not apply — only a full Clifford simulator (Stim) can check it.
    #[test]
    fn memory_x_experiment_shape() {
        let code = BBCode::new(6, 6, &[(3, 0), (0, 1), (0, 2)], &[(0, 3), (1, 0), (2, 0)]);
        let lm = code.num_checks();
        let rounds = 3;
        let exp = code.memory_x_experiment(rounds);
        assert_eq!(exp.num_qubits, code.n() + 2 * lm);
        assert_eq!(exp.num_data(), code.n());
        assert_eq!(exp.num_detectors(), (rounds + 1) * lm);
        assert_eq!(exp.annotated.observables.len(), code.k());
        let cnots = exp
            .annotated
            .circuit
            .instructions()
            .iter()
            .filter(|i| matches!(i, aleph_ir::Instruction::Gate(g) if matches!(g.gate, aleph_core::Gate::Cnot)))
            .count();
        assert_eq!(cnots, rounds * 2 * lm * 6, "6 CNOTs per check per cycle");
    }

    /// The circuit-level DEM is well-formed: right detector/observable counts, all probabilities in
    /// (0,1), it contains genuine hyperedges (weight > 2, since BB checks are weight-6 hypergraphs),
    /// and some mechanism reaches a logical observable.
    #[test]
    fn circuit_level_dem_well_formed() {
        let code = BBCode::gross();
        let rounds = 3;
        let dem = code
            .circuit_level_dem(rounds, CircuitNoise::uniform(0.003))
            .unwrap();
        assert_eq!(dem.detectors, (rounds + 1) * code.num_checks());
        assert_eq!(dem.observables, code.k());
        assert!(!dem.errors.is_empty());
        for e in &dem.errors {
            assert!(e.prob > 0.0 && e.prob < 1.0, "prob {} out of range", e.prob);
        }
        assert!(
            dem.errors.iter().any(|e| e.dets.len() > 2),
            "circuit-level BB DEM must contain hyperedges"
        );
        assert!(
            dem.errors.iter().any(|e| !e.obs.is_empty()),
            "some mechanism must flip a logical observable"
        );
    }
}
