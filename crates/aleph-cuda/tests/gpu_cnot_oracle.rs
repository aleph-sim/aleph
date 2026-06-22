//! P5.9-04: the `apply_cnot` permutation kernel, oracle-equal to the dense path.
//!
//! A plain CNOT routes to `apply_cnot` (swap the target pair where control=1)
//! instead of the dense `apply_kq` 4×4 matvec. This pins the custom path against
//! the unbatched CPU `NaiveSvBackend` at 1e-10 and against the backend's own
//! dense path (`with_custom_2q(false)`), covering both operand orderings
//! (control < target and control > target) and non-adjacent qubits.
//!
//! Gated on `cfg(all(target_os = "linux", feature = "cuda"))`; skips with no GPU.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use aleph_backend::run;
use aleph_core::Complex;
use aleph_cuda::CudaSvBackend;
use aleph_ir::Circuit;
use aleph_oracle::HasAmplitudes;
use aleph_sv::NaiveSvBackend;
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Spread the state first (so every amplitude is non-trivial), then a thicket of
/// CNOTs at mixed distances and both control/target orderings.
fn cnot_thicket(rng: &mut StdRng, n: u32, layers: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.h(q).unwrap();
        c.rx(0.3 + q as f64 * 0.1, q).unwrap();
    }
    for _ in 0..layers {
        for _ in 0..n {
            let a = rng.gen_range(0..n);
            let mut b = rng.gen_range(0..n);
            while b == a {
                b = rng.gen_range(0..n);
            }
            // Randomly flip control/target so both orderings are exercised.
            if rng.gen::<bool>() {
                c.cnot(a, b).unwrap();
            } else {
                c.cnot(b, a).unwrap();
            }
        }
    }
    c
}

fn amps(c: &Circuit, gpu: &mut CudaSvBackend) -> Vec<Complex> {
    HasAmplitudes::amplitudes(&run(gpu, c).expect("gpu run"))
}

#[test]
fn cnot_kernel_matches_cpu_and_dense_path() {
    let mut rng = StdRng::seed_from_u64(0x0590_4c01);
    let n = 11;
    let circ = cnot_thicket(&mut rng, n, 12);

    // CPU per-gate oracle.
    let mut cpu = NaiveSvBackend::with_seed(0);
    let want = HasAmplitudes::amplitudes(&run(&mut cpu, &circ).expect("cpu"));

    // GPU with the custom CNOT kernel (default).
    let mut gpu = match CudaSvBackend::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping cnot oracle: {e}");
            return;
        }
    };
    let got = amps(&circ, &mut gpu);

    // GPU forced onto the dense apply_kq path.
    let mut gpu_dense = CudaSvBackend::with_seed(0)
        .map(|b| b.with_custom_2q(false))
        .expect("gpu");
    let dense = amps(&circ, &mut gpu_dense);

    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).norm() <= 1e-10,
            "cnot vs cpu i={i}: |Δ|={:.2e}",
            (g - w).norm()
        );
    }
    for (i, (g, d)) in got.iter().zip(dense.iter()).enumerate() {
        assert!(
            (g - d).norm() <= 1e-12,
            "cnot vs dense i={i}: |Δ|={:.2e}",
            (g - d).norm()
        );
    }
}

/// Minimal hand-checked truth table: CNOT on |10> → |11> for both orderings,
/// catching any control/target swap in the bit-insertion.
#[test]
fn cnot_basis_states() {
    let mut gpu = match CudaSvBackend::with_seed(0) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping cnot basis: {e}");
            return;
        }
    };
    // control=0, target=1: X on q0 then CNOT(0,1) ⇒ |q0=1,q1=1>.
    // Index convention: amps[i], bit q set ⇔ (i>>q)&1. So q0=1,q1=1 ⇒ i=0b11=3.
    let mut c = Circuit::new(2, 0);
    c.x(0).unwrap();
    c.cnot(0, 1).unwrap();
    let a = HasAmplitudes::amplitudes(&run(&mut gpu, &c).expect("run"));
    assert!(
        (a[3] - Complex::new(1.0, 0.0)).norm() < 1e-12,
        "expected |11>, got {a:?}"
    );

    // control=1 (higher index), target=0: X on q1 then CNOT(1,0) ⇒ q1=1,q0=1 ⇒ i=3.
    let mut c2 = Circuit::new(2, 0);
    c2.x(1).unwrap();
    c2.cnot(1, 0).unwrap();
    let a2 = HasAmplitudes::amplitudes(&run(&mut gpu, &c2).expect("run"));
    assert!(
        (a2[3] - Complex::new(1.0, 0.0)).norm() < 1e-12,
        "expected |11>, got {a2:?}"
    );
}
