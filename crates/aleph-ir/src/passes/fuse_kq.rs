//! `FuseKq` — greedy fusion of adjacent gates into dense k-qubit blocks.
//! See docs/superpowers/specs/2026-06-03-p2-07-fuse-kq-design.md.

use aleph_core::Complex;
use aleph_core::{Gate, GateMatrix};

/// Embed a gate's matrix into the `k`-qubit block space spanned by sorted
/// `block_qubits` (ascending). `gate_qubits` is the gate's own operand list
/// (MSB-first: `gate_qubits[0]` is the gate-matrix MSB). Returns the
/// row-major `2^k × 2^k` lifted matrix.
///
/// # Invariant
///
/// The caller must ensure `gate.matrix()` succeeds — i.e. the gate has a
/// concrete `GateMatrix` (arity ≤ 3, no symbolic parameters). `UnitaryKq`
/// members must use `lift_dense` directly.
#[allow(dead_code)] // Used by FuseKq block builder (Task 3).
pub(crate) fn lift_to_block(
    gate: &Gate,
    gate_qubits: &[u32],
    block_qubits: &[u32],
) -> Vec<Complex> {
    let gd: Vec<Complex> = match gate
        .matrix()
        .expect("FuseKq lifts only concrete GateMatrix gates (no UnitaryKq, no symbolic/non-finite params); UnitaryKq routes via lift_dense")
    {
        GateMatrix::M2x2(m) => m.iter().flatten().copied().collect(),
        GateMatrix::M4x4(m) => m.iter().flatten().copied().collect(),
        GateMatrix::M8x8(m) => m.iter().flatten().copied().collect(),
    };
    lift_dense(&gd, gate_qubits, block_qubits)
}

/// Embed an already-flattened dense `2^ga × 2^ga` matrix (`ga == gate_qubits.len()`)
/// into the `k`-qubit block space spanned by sorted `block_qubits` (ascending).
///
/// `gate_qubits` is MSB-first: `gate_qubits[0]` is the most significant bit
/// of the gate's row/col index. `block_qubits` is sorted ascending; block-bit
/// position `p` (0 = LSB … k−1 = MSB) corresponds to `block_qubits[k-1-p]`.
///
/// Returns a row-major `2^k × 2^k` matrix.
#[allow(dead_code)] // Used by FuseKq block builder (Task 3).
pub(crate) fn lift_dense(
    gd: &[Complex],
    gate_qubits: &[u32],
    block_qubits: &[u32],
) -> Vec<Complex> {
    let k = block_qubits.len();
    let dim = 1usize << k;
    let ga = gate_qubits.len();
    let gdim = 1usize << ga;
    debug_assert_eq!(gd.len(), gdim * gdim);

    // Map a physical qubit to its block-bit position (0 = LSB, k-1 = MSB).
    // block_qubits is sorted ascending so block_qubits[0] is the smallest qubit
    // index, which by the MSB-first convention is the HIGHEST block bit (k-1).
    let block_bit_of = |q: u32| -> usize {
        let idx = block_qubits
            .iter()
            .position(|&bq| bq == q)
            .expect("gate qubit must lie in block");
        k - 1 - idx
    };

    let mut out = vec![Complex::new(0.0, 0.0); dim * dim];

    for col in 0..dim {
        // Extract the gate-input index from this block column index.
        // gate_qubits[a] occupies gate bit (ga-1-a) (MSB-first).
        let mut gin = 0usize;
        for (a, &gq) in gate_qubits.iter().enumerate() {
            let bb = block_bit_of(gq);
            gin |= ((col >> bb) & 1) << (ga - 1 - a);
        }

        for grow in 0..gdim {
            let amp = gd[grow * gdim + gin];
            if amp == Complex::new(0.0, 0.0) {
                continue;
            }
            // Reconstruct the block row: start from col (identity on non-gate bits),
            // then overwrite the gate bits with the output bits from grow.
            let mut row = col;
            for (a, &gq) in gate_qubits.iter().enumerate() {
                let bb = block_bit_of(gq);
                let outbit = (grow >> (ga - 1 - a)) & 1;
                row = (row & !(1usize << bb)) | (outbit << bb);
            }
            out[row * dim + col] += amp;
        }
    }

    out
}

#[cfg(test)]
mod lift_tests {
    use super::*;
    use aleph_core::{Gate, GateInstance};
    use smallvec::smallvec;

    fn brute_lift(gate: &Gate, gate_qubits: &[u32], block_qubits: &[u32]) -> Vec<Complex> {
        let k = block_qubits.len();
        let dim = 1usize << k;
        let gd = match gate.matrix().unwrap() {
            aleph_core::GateMatrix::M2x2(m) => m.iter().flatten().copied().collect::<Vec<_>>(),
            aleph_core::GateMatrix::M4x4(m) => m.iter().flatten().copied().collect::<Vec<_>>(),
            aleph_core::GateMatrix::M8x8(m) => m.iter().flatten().copied().collect::<Vec<_>>(),
        };
        let ga = gate_qubits.len();
        let gdim = 1usize << ga;
        let block_bit_of = |q: u32| -> usize {
            let idx = block_qubits.iter().position(|&bq| bq == q).unwrap();
            k - 1 - idx
        };
        let mut out = vec![Complex::new(0.0, 0.0); dim * dim];
        for col in 0..dim {
            let mut gin = 0usize;
            for (a, &gq) in gate_qubits.iter().enumerate() {
                let bb = block_bit_of(gq);
                gin |= ((col >> bb) & 1) << (ga - 1 - a);
            }
            for grow in 0..gdim {
                let amp = gd[grow * gdim + gin];
                if amp.norm() < 1e-18 {
                    continue;
                }
                let mut row = col;
                for (a, &gq) in gate_qubits.iter().enumerate() {
                    let bb = block_bit_of(gq);
                    let outbit = (grow >> (ga - 1 - a)) & 1;
                    row = (row & !(1usize << bb)) | (outbit << bb);
                }
                out[row * dim + col] += amp;
            }
        }
        out
    }

    #[test]
    fn lift_1q_into_3q_block_matches_bruteforce() {
        let g = Gate::X;
        let gq = [5u32];
        let bq = [2u32, 5, 7];
        let got = lift_to_block(&g, &gq, &bq);
        let want = brute_lift(&g, &gq, &bq);
        assert_eq!(got.len(), want.len());
        for i in 0..got.len() {
            assert!((got[i] - want[i]).norm() < 1e-12, "entry {i}");
        }
    }

    #[test]
    fn lift_2q_cnot_into_3q_block_matches_bruteforce() {
        let g = Gate::Cnot;
        let gq = [7u32, 2]; // [control, target], reversed/non-adjacent
        let bq = [2u32, 5, 7];
        let got = lift_to_block(&g, &gq, &bq);
        let want = brute_lift(&g, &gq, &bq);
        for i in 0..got.len() {
            assert!((got[i] - want[i]).norm() < 1e-12, "entry {i}");
        }
        let _ = GateInstance::new(Gate::H, smallvec![0u32]);
    }

    #[test]
    fn lift_rz_2q_block_matches_bruteforce() {
        // single-qubit non-trivial gate into a 2-qubit block, low position
        let g = Gate::Rz(0.7.into());
        let gq = [0u32];
        let bq = [0u32, 3];
        let got = lift_to_block(&g, &gq, &bq);
        let want = brute_lift(&g, &gq, &bq);
        for i in 0..got.len() {
            assert!((got[i] - want[i]).norm() < 1e-12, "entry {i}");
        }
    }

    #[test]
    fn lift_cnot_msb_first_is_canonical_cnot() {
        // CNOT, gate_qubits=[0,1] (control=0=MSB, target=1=LSB), block Q=[0,1].
        // Canonical CNOT (q0=MSB): |10>->|11>, |11>->|10>.
        let got = lift_to_block(&Gate::Cnot, &[0u32, 1], &[0u32, 1]);
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        let want: [Complex; 16] = [o, z, z, z, z, o, z, z, z, z, z, o, z, z, o, z];
        for i in 0..16 {
            assert!(
                (got[i] - want[i]).norm() < 1e-12,
                "entry {i}: got {:?} want {:?}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn lift_cnot_reversed_operands_is_reversed_cnot() {
        // CNOT, gate_qubits=[1,0] (control=q1=LSB, target=q0=MSB), block Q=[0,1].
        // Flips q0(MSB) when q1(LSB)=1: |01>->|11>, |11>->|01>.
        let got = lift_to_block(&Gate::Cnot, &[1u32, 0], &[0u32, 1]);
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        let want: [Complex; 16] = [o, z, z, z, z, z, z, o, z, z, o, z, z, o, z, z];
        for i in 0..16 {
            assert!(
                (got[i] - want[i]).norm() < 1e-12,
                "entry {i}: got {:?} want {:?}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn lift_x_middle_qubit_flips_only_that_bit() {
        // X on q5 in block Q=[2,5,7]: q2=bit2(MSB), q5=bit1, q7=bit0(LSB).
        // Lifted 8x8 is the permutation row == col ^ 0b010 (flip bit 1 only).
        let got = lift_to_block(&Gate::X, &[5u32], &[2u32, 5, 7]);
        for col in 0..8usize {
            let expect_row = col ^ 0b010;
            for row in 0..8usize {
                let want = if row == expect_row {
                    Complex::new(1.0, 0.0)
                } else {
                    Complex::new(0.0, 0.0)
                };
                assert!(
                    (got[row * 8 + col] - want).norm() < 1e-12,
                    "row={row} col={col}"
                );
            }
        }
    }
}
