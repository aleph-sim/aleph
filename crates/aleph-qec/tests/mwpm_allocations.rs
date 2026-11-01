//! Count allocations on the calling thread only; setup and test-runner activity are excluded.
use aleph_qec::{build_dem, Correction, Decoder, MwpmDecoder, SurfaceCode, Syndrome};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountingAllocator;
thread_local! {
    static COUNT: Cell<Option<usize>> = const { Cell::new(None) };
}

fn allocated() {
    let _ = COUNT.try_with(|c| {
        if let Some(n) = c.get() {
            c.set(Some(n + 1));
        }
    });
}

// SAFETY: This test-only allocator forwards every pointer and layout unchanged to System.
// The thread-local counter contains no heap storage and never calls the allocator itself.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        allocated();
        // SAFETY: GlobalAlloc's caller provides the valid layout forwarded to System.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        allocated();
        // SAFETY: Same forwarding contract as alloc.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: Pointer/layout originate from the System allocation being freed.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        allocated();
        // SAFETY: Forward the caller's live System allocation and requested size unchanged.
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn reusable_output_resizes_and_clears_previous_observables() {
    use aleph_qec::DetectorErrorModel;
    let dem = DetectorErrorModel::parse("error(0.1) D0 L1").unwrap();
    let decoder = MwpmDecoder::new(&dem).unwrap();
    let mut output = Correction::new(vec![true; 8]);
    decoder.decode_into(&Syndrome::new(1, vec![0]), &mut output);
    assert_eq!(output.observable_flips, [false, true]);
    decoder.decode_into(&Syndrome::default(), &mut output);
    assert_eq!(output.observable_flips, [false, false]);
    let empty = MwpmDecoder::new(&DetectorErrorModel::default()).unwrap();
    empty.decode_into(&Syndrome::default(), &mut output);
    assert!(output.observable_flips.is_empty());
    decoder.decode_into(&Syndrome::new(1, vec![0]), &mut output);
    assert_eq!(output.observable_flips, [false, true]);
}

fn count(f: impl FnOnce()) -> usize {
    struct Stop;
    impl Drop for Stop {
        fn drop(&mut self) {
            COUNT.with(|c| c.set(None));
        }
    }
    COUNT.with(|c| c.set(Some(0)));
    let _stop = Stop;
    f();
    COUNT.with(|c| c.get().unwrap_or(0))
}

#[test]
fn d11_repeated_workload_allocates_only_owned_outputs() {
    let exp = SurfaceCode::new(11).memory_z_experiment(11);
    let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(0.03, 0.03)).unwrap();
    let decoder = MwpmDecoder::new(&dem).unwrap();
    // Same SplitMix64 corpus as the Criterion benchmark, including every d=11 benchmark shot.
    let mut syndromes = Vec::new();
    for shot in 0..512u64 {
        let mut z = (0x5EED ^ 11u64).wrapping_add(shot.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut bits = vec![false; dem.detectors];
        for e in &dem.errors {
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            if ((z >> 11) as f64 / (1u64 << 53) as f64) < e.prob {
                for &d in &e.dets {
                    bits[d as usize] ^= true;
                }
            }
        }
        syndromes.push(Syndrome::from_bits(&bits));
    }
    // Empty and malformed public fields must not leak a previous shot's state/output.
    syndromes.push(Syndrome::default());
    syndromes.push(Syndrome {
        detectors: dem.detectors,
        fired: vec![8, 2, 8, u32::MAX],
    });
    let expected: Vec<_> = syndromes.iter().map(|s| decoder.decode(s)).collect();
    let estimates: Vec<_> = syndromes.iter().map(|s| decoder.decode_errors(s)).collect();
    let mut output = Correction::none(0);
    for _ in 0..2 {
        for s in &syndromes {
            decoder.decode_into(s, &mut output);
        }
    }
    let into_allocs = count(|| {
        for _ in 0..3 {
            for (s, expected) in syndromes.iter().zip(&expected) {
                decoder.decode_into(s, &mut output);
                assert_eq!(&output, expected);
            }
        }
    });
    let owned_allocs = count(|| {
        for (s, expected) in syndromes.iter().zip(&expected) {
            assert_eq!(&decoder.decode(s), expected);
        }
    });
    let estimate_allocs = count(|| {
        for (s, expected) in syndromes.iter().zip(&estimates) {
            assert_eq!(&decoder.decode_errors(s), expected);
        }
    });
    eprintln!("shots={} decode_into_allocs={into_allocs} decode_allocs={owned_allocs} decode_errors_allocs={estimate_allocs}", syndromes.len());
    assert_eq!(
        into_allocs, 0,
        "matcher/input/output reuse must allocate nothing after warmup"
    );
    assert_eq!(
        owned_allocs,
        syndromes.len(),
        "only the returned Correction allocates"
    );
    assert_eq!(
        estimate_allocs,
        syndromes.len(),
        "only the returned error estimate allocates"
    );
}
