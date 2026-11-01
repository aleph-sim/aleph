//! Compare the previous byte-batch-plus-cast output with direct CUDA-Q float64 output.
//! RAYON_NUM_THREADS=1 cargo bench -p aleph-py --bench error_output
use aleph_py::qec_core::{
    decode_errors_rows, decode_errors_rows_f64_into, AnyDecoder, DecoderParams,
};
use aleph_qec::{build_dem, SurfaceCode};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("cudaq_error_output");
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(3));
    for distance in [5, 9] {
        let experiment = SurfaceCode::new(distance).memory_z_experiment(distance);
        let dem = build_dem(
            &experiment.annotated,
            &experiment.phenomenological_mechanisms(0.003, 0.003),
        )
        .unwrap();
        let decoder = AnyDecoder::build(&dem, "mwpm", &DecoderParams::default()).unwrap();
        let shots = 256;
        // Seeded Bernoulli errors give valid sparse syndromes, rather than arbitrary dense rows.
        let mut state = 516u64;
        let mut bits = vec![0u8; shots * dem.detectors];
        for row in bits.chunks_mut(dem.detectors) {
            for error in &dem.errors {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if (state >> 11) as f64 / ((1u64 << 53) as f64) < error.prob {
                    for &d in &error.dets {
                        row[d as usize] ^= 1;
                    }
                }
            }
        }
        let previous = |out: &mut [f64]| {
            let (bytes, conv) =
                decode_errors_rows(&decoder, &bits, shots, dem.detectors, dem.errors.len());
            for (dst, bit) in out.iter_mut().zip(bytes) {
                *dst = f64::from(bit);
            }
            conv
        };
        let direct = |out: &mut [f64]| {
            decode_errors_rows_f64_into(
                &decoder,
                &bits,
                shots,
                dem.detectors,
                dem.errors.len(),
                out,
            )
        };
        let mut reference = vec![0.; shots * dem.errors.len()];
        let mut output = reference.clone();
        assert_eq!(previous(&mut reference), direct(&mut output));
        assert_eq!(reference, output);
        // Both arms reuse the same float64 output allocation: isolate writing/decoding,
        // without pretending the Rust allocator models NumPy's allocation policy.
        group.throughput(Throughput::Elements(shots as u64));
        group.bench_function(BenchmarkId::new("byte_then_cast", distance), |b| {
            b.iter(|| black_box(previous(&mut output)))
        });
        group.bench_function(BenchmarkId::new("direct_f64_into", distance), |b| {
            b.iter(|| black_box(direct(&mut output)))
        });
    }
    group.finish();
}
criterion_group!(benches, bench);
criterion_main!(benches);
