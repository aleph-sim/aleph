//! P3-10 NN brickwork: CNOT-Rz-CNOT interactions and Rx mixers.
use aleph_core::{Pauli, PauliString, PauliSum};
use aleph_ir::Circuit;
pub fn brickwork(n: u32, layers: u32, offset: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.h(q).unwrap();
    }
    for layer in 0..layers {
        for q in 0..n - 1 {
            if (q + offset) % 2 != layer % 2 {
                continue;
            }
            c.cnot(q, q + 1).unwrap();
            c.rz(0.3 + 0.05 * f64::from(q + offset), q + 1).unwrap();
            c.cnot(q, q + 1).unwrap();
        }
        for q in 0..n {
            c.rx(0.4 + 0.03 * f64::from(layer), q).unwrap();
        }
    }
    c
}
pub fn global(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.h(0).unwrap();
    for q in 0..n - 1 {
        c.cnot(q, q + 1).unwrap();
    }
    c
}
pub fn observable(q: u32) -> PauliSum {
    PauliSum {
        terms: vec![PauliString::new(1., vec![(q, Pauli::X)]).unwrap()],
    }
}
