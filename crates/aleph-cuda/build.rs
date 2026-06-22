//! Build script for `aleph-cuda`.
//!
//! Its only job is to link NVIDIA cuStateVec when the `cuquantum` feature is on
//! (P5-03). For every other configuration — the default build, the `cuda`-only
//! build, macOS, the CUDA-less CI runner — it is a no-op, so nothing here forces
//! a CUDA or cuQuantum install to be present.
//!
//! cuStateVec ships as `libcustatevec.so`. On Ubuntu's cuquantum packages it
//! lands in the default linker search path (`/usr/lib/x86_64-linux-gnu`); a
//! custom install is found via `CUQUANTUM_ROOT` (the path NVIDIA's tarball/conda
//! distributions set). `cudarc` still loads `libcuda` dynamically, so we only
//! add the cuStateVec library here.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUQUANTUM_ROOT");

    // Only do anything when the cuStateVec backend is actually compiled in.
    if std::env::var_os("CARGO_FEATURE_CUQUANTUM").is_none() {
        return;
    }

    // cuStateVec is Linux + NVIDIA only; never try to link it elsewhere.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }

    if let Some(root) = std::env::var_os("CUQUANTUM_ROOT") {
        let root = root.to_string_lossy();
        // NVIDIA's distributions use `lib/`; some package layouts use `lib64/`.
        println!("cargo:rustc-link-search=native={root}/lib");
        println!("cargo:rustc-link-search=native={root}/lib64");
    }
    // Distro package location (Ubuntu cuquantum). Harmless if already on the path.
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");

    println!("cargo:rustc-link-lib=dylib=custatevec");
}
