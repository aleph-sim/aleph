//! End-to-end Metal compute dispatch: runtime-compile the smoke kernel, run it
//! on the GPU over a unified-memory buffer, and verify the result host-side.
//! Skips (passes) when no Metal device is available so headless CI stays green.

#![cfg(all(target_os = "macos", feature = "metal"))]

use aleph_metal::{DeviceBuffer, Error, MetalContext};
use metal::MTLSize;

const SMOKE_SRC: &str = include_str!("../src/shaders/smoke.metal");

#[test]
fn smoke_double_runs_on_gpu() {
    let ctx = match MetalContext::new() {
        Ok(c) => c,
        Err(Error::NoDevice) => {
            eprintln!("skipping GPU smoke test: no Metal device available");
            return;
        }
        Err(e) => panic!("unexpected Metal init error: {e}"),
    };

    let pipeline = ctx
        .make_compute_pipeline(SMOKE_SRC, "smoke_double")
        .expect("smoke pipeline builds");

    let input: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let buf = DeviceBuffer::from_slice(&ctx, &input);

    let cmd = ctx.queue().new_command_buffer();
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(buf.metal_buffer()), 0);

    let n = input.len() as u64;
    // non-uniform dispatch_threads still requires a threadgroup size no larger
    // than max_total_threads_per_threadgroup; clamping to n also avoids an
    // oversized group when the element count is tiny.
    let tg = pipeline.max_total_threads_per_threadgroup().min(n);
    encoder.dispatch_threads(MTLSize::new(n, 1, 1), MTLSize::new(tg, 1, 1));
    encoder.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    for (i, &v) in buf.as_slice().iter().enumerate() {
        // Exact: f32 doubling is bit-exact.
        assert_eq!(v, 2.0 * input[i], "element {i} not doubled");
    }
}
