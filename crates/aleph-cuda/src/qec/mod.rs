//! GPU Union-Find / cluster-growth decoder (Q3-01).
//!
//! [`CudaUnionFind`] ports the CPU [`UnionFindDecoder`](aleph_qec::UnionFindDecoder) to CUDA. It
//! uploads the decoder's flattened matching graph once (device-resident, shared, read-only) and
//! decodes a **batch** of syndromes with one GPU thread per shot, each thread running the full
//! serial Delfosse-Nickerson decode (growth + peeling) on its own syndrome. The result is
//! **bit-identical** to the CPU decoder — there is no cross-thread interaction to diverge on — so
//! the CPU decoder is a direct oracle (see `tests/qec_uf_oracle.rs`). Throughput comes from running
//! thousands of independent shots concurrently, which is exactly the Monte-Carlo decode workload.
//!
//! The host consumes an [`aleph_qec::DecoderGraph`] (the CPU decoder's own arrays), so the GPU and
//! CPU decode the identical graph layout, edge ordering and growth mode.

mod uf;

pub use uf::{mask_to_flips, CudaUnionFind};
