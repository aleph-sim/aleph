//! Q6-02 (sim) — reference vectors for the repetition-code Union-Find decoder RTL.
//!
//! The repetition (bit-flip) code of distance `D`: `D` data qubits, `D-1` parity checks
//! `c_i = e_i ⊕ e_{i+1}`. A data error `e_j` lights checks `{j-1, j}` (a bulk edge), or a single
//! check at the two ends (boundary edges) — a 1-D matching graph, the simplest non-trivial case for
//! Union-Find. The logical observable is `X` on data qubit 0 (a valid logical-X representative): the
//! `j=0` error carries it.
//!
//! For every one of the `2^(D-1)` syndromes we run the Rust [`UnionFindDecoder`] and emit its
//! predicted logical flip. The RTL (`hw/uf_rep_decoder.sv`) decodes the same graph and the Verilator
//! testbench checks (a) its correction reproduces the syndrome and (b) its logical flip matches this
//! oracle — verifying a *real* decoder datapath in hardware, not a lookup table. This 1-D core is the
//! stepping stone to the full 2-D surface-code Union-Find datapath (Q6-02 proper).
//!
//! Output (one entry per syndrome index, LSB = check 0), for Verilog `$readmemb`: a single bit = the
//! predicted flip of logical observable 0. Header comments carry `D`.
//!
//! Usage: `cargo run --release -p aleph-qec --example qec_rep_uf_vectors > hw/rep_uf_vectors.mem`

use aleph_qec::{DemError, DetectorErrorModel, Syndrome, UnionFindDecoder};

const D: usize = 7; // data qubits; D-1 = 6 checks → 64 syndromes (exhaustive)

fn main() {
    let dets = D - 1;
    // One error mechanism per data qubit. e_j lights checks {j-1, j} ∩ [0,dets); e_0 carries logical.
    let mut errors = Vec::with_capacity(D);
    for j in 0..D {
        let mut d = Vec::new();
        if j > 0 {
            d.push((j - 1) as u32);
        }
        if j < dets {
            d.push(j as u32);
        }
        let obs = if j == 0 { vec![0u32] } else { vec![] };
        errors.push(DemError::new(0.05, d, obs));
    }
    let dem = DetectorErrorModel {
        detectors: dets,
        observables: 1,
        errors,
    };
    let decoder = UnionFindDecoder::new_weighted(&dem).expect("uf decoder");

    println!(
        "// repetition-code distance-{D} Union-Find decoder reference (logical flip per syndrome)"
    );
    println!("// D={D} detectors={dets} entries={}", 1usize << dets);
    println!("// line i (0-indexed) = predicted flip of logical observable 0 for syndrome i (LSB=check0)");
    for idx in 0u32..(1u32 << dets) {
        let bits: Vec<bool> = (0..dets).map(|b| (idx >> b) & 1 == 1).collect();
        let syn = Syndrome::from_bits(&bits);
        let corr = aleph_qec::Decoder::decode(&decoder, &syn);
        let flip = corr.observable_flips.first().copied().unwrap_or(false);
        println!("{}", u8::from(flip));
    }
}
