//! P2-04: applying any gate under different ChunkPolicy values must
//! produce bit-identical amplitudes. The knobs only change task
//! partitioning, never which amplitude a body writes and never a
//! cross-thread FP reduction — so equality is exact, not within a
//! tolerance. Guards against a future kernel accidentally letting the
//! policy leak into results.

use aleph_core::Complex;
use aleph_sv::kernels::{self, tuning};

fn seeded_state(n: u32) -> Vec<Complex> {
    // Hash each index to a well-spread angle in [0, 2π) so adjacent
    // amplitudes differ substantially — exercises the kernel on varied
    // (not near-identical) data. Magnitudes are irrelevant here: this
    // test checks policy-invariance of the output, not its correctness.
    (0..(1usize << n))
        .map(|k| {
            let h = (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let theta = (h >> 11) as f64 / (1u64 << 53) as f64 * std::f64::consts::TAU;
            Complex::new(theta.cos(), theta.sin())
        })
        .collect()
}

fn h_matrix() -> [[Complex; 2]; 2] {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    [
        [Complex::new(s, 0.0), Complex::new(s, 0.0)],
        [Complex::new(s, 0.0), Complex::new(-s, 0.0)],
    ]
}

/// RAII: forces `policy` for the duration of the guard, restores `None`
/// on drop (panic-safe). The override is a thread-local consulted by
/// `resolve_policy`; `par_blocks` receives the resolved `ChunkPolicy` by
/// value, so the forced policy reaches the rayon workers correctly.
struct PolicyGuard;
impl PolicyGuard {
    fn set(policy: tuning::ChunkPolicy) -> Self {
        tuning::test_override::set(Some(policy));
        PolicyGuard
    }
}
impl Drop for PolicyGuard {
    fn drop(&mut self) {
        tuning::test_override::set(None);
    }
}

const POLICIES: &[tuning::ChunkPolicy] = &[
    tuning::ChunkPolicy {
        min_amps: 0,
        grain: 1,
    },
    tuning::ChunkPolicy {
        min_amps: 0,
        grain: 4096,
    },
    tuning::ChunkPolicy {
        min_amps: usize::MAX,
        grain: 64,
    },
    tuning::ChunkPolicy {
        min_amps: 1 << 18,
        grain: 64,
    },
];

fn assert_invariant(reference: &[Complex], state: &[Complex], label: &str) {
    assert_eq!(reference.len(), state.len());
    for (i, (a, b)) in reference.iter().zip(state).enumerate() {
        assert_eq!(
            a.re.to_bits(),
            b.re.to_bits(),
            "{label}: re mismatch at {i}"
        );
        assert_eq!(
            a.im.to_bits(),
            b.im.to_bits(),
            "{label}: im mismatch at {i}"
        );
    }
}

/// H → `apply_1q` generic path (neither diagonal nor anti-diagonal).
#[test]
fn one_q_generic_h_is_policy_invariant() {
    let n = 14;
    let m = h_matrix();
    for &target in &[0u32, 5, 11] {
        let reference = {
            let mut s = seeded_state(n);
            {
                let _g = PolicyGuard::set(POLICIES[3]);
                kernels::aos::apply_1q(&mut s, target, &[], &m);
            }
            s
        };
        for p in POLICIES {
            let mut s = seeded_state(n);
            {
                let _g = PolicyGuard::set(*p);
                kernels::aos::apply_1q(&mut s, target, &[], &m);
            }
            assert_invariant(&reference, &s, &format!("H target={target} policy={p:?}"));
        }
    }
}

/// CNOT → permutation dispatch (`apply_2q_cnot_*`), NOT the dense kernel.
#[test]
fn cnot_is_policy_invariant() {
    let n = 14;
    let m = {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        [[o, z, z, z], [z, o, z, z], [z, z, z, o], [z, z, o, z]]
    };
    let reference = {
        let mut s = seeded_state(n);
        {
            let _g = PolicyGuard::set(POLICIES[3]);
            kernels::aos::apply_2q(&mut s, [3, 7], &[], &m);
        }
        s
    };
    for p in POLICIES {
        let mut s = seeded_state(n);
        {
            let _g = PolicyGuard::set(*p);
            kernels::aos::apply_2q(&mut s, [3, 7], &[], &m);
        }
        assert_invariant(&reference, &s, &format!("CNOT policy={p:?}"));
    }
}

/// H⊗H: dense 4x4 (no permutation/diagonal fast-path) → exercises
/// the apply_2q dense kernel's resolve_policy call
/// (`apply_2q_dense_scalar` on scalar/aarch64; `apply_2q_avx512` on x86_64).
#[test]
fn dense_2q_is_policy_invariant() {
    let n = 14;
    let h = 0.5_f64;
    let m = [
        [
            Complex::new(h, 0.0),
            Complex::new(h, 0.0),
            Complex::new(h, 0.0),
            Complex::new(h, 0.0),
        ],
        [
            Complex::new(h, 0.0),
            Complex::new(-h, 0.0),
            Complex::new(h, 0.0),
            Complex::new(-h, 0.0),
        ],
        [
            Complex::new(h, 0.0),
            Complex::new(h, 0.0),
            Complex::new(-h, 0.0),
            Complex::new(-h, 0.0),
        ],
        [
            Complex::new(h, 0.0),
            Complex::new(-h, 0.0),
            Complex::new(-h, 0.0),
            Complex::new(h, 0.0),
        ],
    ];
    let reference = {
        let mut s = seeded_state(n);
        {
            let _g = PolicyGuard::set(POLICIES[3]);
            kernels::aos::apply_2q(&mut s, [3, 7], &[], &m);
        }
        s
    };
    for p in POLICIES {
        let mut s = seeded_state(n);
        {
            let _g = PolicyGuard::set(*p);
            kernels::aos::apply_2q(&mut s, [3, 7], &[], &m);
        }
        assert_invariant(&reference, &s, &format!("HxH policy={p:?}"));
    }
}
