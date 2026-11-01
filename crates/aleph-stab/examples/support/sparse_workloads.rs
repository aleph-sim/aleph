use aleph_core::{Pauli, PauliString, PauliSum};
use aleph_ir::Circuit;
use std::error::Error;
pub fn names() -> Vec<String> {
    let mut out = vec![
        "ghz18-clifford".into(),
        "ghz18-rotation".into(),
        "local128".into(),
        "ghz128".into(),
        "cancel16".into(),
    ];
    for k in [0, 2, 4, 8, 12, 16] {
        out.push(format!("product16-k{k}"));
    }
    out
}
pub fn fixture(name: &str) -> Result<(Circuit, PauliSum, Option<f64>), Box<dyn Error>> {
    let (mut c, ps, analytic) = if name.starts_with("ghz") {
        let n = if name == "ghz128" { 128 } else { 18 };
        let mut c = Circuit::new(n, 0);
        c.h(0)?;
        for q in 1..n {
            c.cnot(0, q)?;
        }
        let rotated = name != "ghz18-clifford";
        if rotated {
            c.rz(0.37, 0)?;
        }
        (
            c,
            (0..n).map(|q| (q, Pauli::X)).collect(),
            Some(if rotated { 0.37f64.cos() } else { 1. }),
        )
    } else if name == "local128" {
        let mut c = Circuit::new(128, 0);
        for layer in 0..3 {
            for q in 0..128 {
                c.ry(0.17 + (q as f64) * 0.002, q)?;
            }
            for q in ((layer % 2)..127).step_by(2) {
                c.cnot(q, q + 1)?;
            }
        }
        (c, vec![(64, Pauli::Z)], None)
    } else if name == "cancel16" {
        let mut c = Circuit::new(16, 0);
        c.h(0)?;
        for _ in 0..32 {
            c.rz(0.31, 0)?;
            c.rz(-0.31, 0)?;
        }
        (c, vec![(0, Pauli::X)], Some(1.))
    } else if let Some(k) = name.strip_prefix("product16-k") {
        let k: u32 = k.parse()?;
        if k > 16 {
            return Err("k > 16".into());
        }
        let mut c = Circuit::new(16, 0);
        for q in 0..k {
            c.ry(0.37, q)?;
        }
        (
            c,
            (0..16).map(|q| (q, Pauli::Z)).collect(),
            Some(0.37f64.cos().powi(k as i32)),
        )
    } else {
        return Err("unknown fixture".into());
    };
    // Retain circuit identity metadata in fixture construction rather than changing execution.
    c = c.with_name(name);
    Ok((
        c,
        PauliSum {
            terms: vec![PauliString::new(1., ps)?],
        },
        analytic,
    ))
}
