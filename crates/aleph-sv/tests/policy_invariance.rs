//! P2-04: applying any gate under different ChunkPolicy values must
//! produce bit-identical amplitudes. The knobs only change task
//! partitioning, never which amplitude a body writes and never a
//! cross-thread FP reduction — so equality is exact, not within a
//! tolerance. Guards against a future kernel accidentally letting the
//! policy leak into results.

use aleph_core::Complex;
use aleph_sv::kernels::{self, tuning};

fn seeded_state(n: u32) -> Vec<Complex> {
    (0..(1usize << n))
        .map(|k| {
            let r = ((k as u64).wrapping_mul(2_654_435_761) as f64) * 1e-19;
            Complex::new(r.sin(), r.cos())
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

fn with_policy<R>(policy: tuning::ChunkPolicy, f: impl FnOnce() -> R) -> R {
    tuning::test_override::set(Some(policy));
    let r = f();
    tuning::test_override::set(None);
    r
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

#[test]
fn one_q_generic_h_is_policy_invariant() {
    let n = 12;
    let m = h_matrix();
    for &target in &[0u32, 5, 11] {
        let reference = {
            let mut s = seeded_state(n);
            with_policy(POLICIES[3], || {
                kernels::aos::apply_1q(&mut s, target, &[], &m)
            });
            s
        };
        for p in POLICIES {
            let mut s = seeded_state(n);
            with_policy(*p, || kernels::aos::apply_1q(&mut s, target, &[], &m));
            assert_invariant(&reference, &s, &format!("H target={target} policy={p:?}"));
        }
    }
}

#[test]
fn cnot_is_policy_invariant() {
    let n = 12;
    let m = {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        [[o, z, z, z], [z, o, z, z], [z, z, z, o], [z, z, o, z]]
    };
    let reference = {
        let mut s = seeded_state(n);
        with_policy(POLICIES[3], || {
            kernels::aos::apply_2q(&mut s, [3, 7], &[], &m)
        });
        s
    };
    for p in POLICIES {
        let mut s = seeded_state(n);
        with_policy(*p, || kernels::aos::apply_2q(&mut s, [3, 7], &[], &m));
        assert_invariant(&reference, &s, &format!("CNOT policy={p:?}"));
    }
}
