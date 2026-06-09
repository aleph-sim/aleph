//! Dump the Stim program for one surface-code cycle to stdout. Used to
//! regenerate the committed timing corpus under
//! `scripts/surface_code/circuits/surface_d{d}.stim`.
//!
//!   cargo run -q -p aleph-benches --bin surface_dump -- 5 > surface_d5.stim

fn main() {
    let d: usize = std::env::args()
        .nth(1)
        .expect("usage: surface_dump <distance>")
        .parse()
        .expect("distance must be a positive odd integer");
    print!("{}", aleph_benches::surface_code_stim_program(d));
}
