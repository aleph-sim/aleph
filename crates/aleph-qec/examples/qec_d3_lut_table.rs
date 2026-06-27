//! Q6-01 (sim) — emit the distance-3 rotated surface-code memory-Z decoder as a lookup table.
//!
//! For the smallest interesting code, a full syndrome → logical-correction table fits in 2^D
//! entries (D = detector count). This table is the **oracle** for the sim-first FPGA decoder skeleton
//! (`hw/`): the SystemVerilog ROM implements it, and the Verilator testbench checks the RTL against
//! it for every syndrome. Larger codes cannot be tabulated (exponential) — they get the Union-Find
//! RTL in Q6-02 — but this establishes the whole host-model ↔ RTL ↔ simulation flow end to end with a
//! genuinely correct, decoder-derived table.
//!
//! Output: one line per syndrome index `0..2^D` (LSB = detector 0), each `0` or `1` = the predicted
//! flip of logical observable 0, suitable for Verilog `$readmemb`. The header lines (comments) carry
//! the detector count so the build is self-describing.
//!
//! Usage: `cargo run --release -p aleph-qec --example qec_d3_lut_table > hw/surface_d3_lut.mem`

use aleph_qec::{build_dem, SurfaceCode, Syndrome, UnionFindDecoder};

fn main() {
    // d=3, single round, phenomenological noise → a graph-like memory-Z DEM the Union-Find decoder
    // consumes. The noise rates only set edge weights; the resulting LUT is the decoder's behaviour.
    let exp = SurfaceCode::new(3).memory_z_experiment(1);
    let dem =
        build_dem(&exp.annotated, &exp.phenomenological_mechanisms(0.01, 0.01)).expect("build dem");
    let d = dem.detectors;
    assert!(
        d <= 16,
        "LUT only practical for small detector counts (got {d})"
    );
    let decoder = UnionFindDecoder::new_weighted(&dem).expect("uf decoder");

    println!("// distance-3 rotated surface-code memory-Z decoder LUT (Union-Find, weighted)");
    println!(
        "// detectors={d} observables={} entries={}",
        dem.observables,
        1usize << d
    );
    println!(
        "// line i (0-indexed) = predicted flip of logical observable 0 for syndrome i (LSB=D0)"
    );
    for idx in 0u32..(1u32 << d) {
        let bits: Vec<bool> = (0..d).map(|b| (idx >> b) & 1 == 1).collect();
        let syn = Syndrome::from_bits(&bits);
        let corr = aleph_qec::Decoder::decode(&decoder, &syn);
        let flip = corr.observable_flips.first().copied().unwrap_or(false);
        println!("{}", u8::from(flip));
    }
}
