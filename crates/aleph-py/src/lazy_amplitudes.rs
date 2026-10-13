//! Serialized, retryable GPU materialization with a detached host cache.
use aleph_core::Complex;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

static MATERIALIZATIONS: AtomicU64 = AtomicU64::new(0);
static RESIDENT_BYTES: AtomicU64 = AtomicU64::new(0);
static CACHED_BYTES: AtomicU64 = AtomicU64::new(0);

enum Storage<S> {
    Device(S),
    Host(Arc<Vec<Complex>>),
}

/// The mutex requires only `S: Send`, not `Sync`; no unchecked GPU trait impls.
/// Copies run without the GIL, serializing concurrent reads of the same result.
pub(crate) struct LazyAmplitudes<S> {
    storage: Mutex<Storage<S>>,
    device_bytes: u64,
}

impl<S> LazyAmplitudes<S> {
    pub(crate) fn new(state: S, device_bytes: u64) -> Self {
        RESIDENT_BYTES.fetch_add(device_bytes, Ordering::Relaxed);
        Self {
            storage: Mutex::new(Storage::Device(state)),
            device_bytes,
        }
    }

    /// A failed copy retains the device state for retry. A successful copy drops
    /// it and caches the host amplitudes; callers copy the shared cache into independent NumPy arrays.
    pub(crate) fn materialize(
        &self,
        copy: impl FnOnce(&S) -> Result<Vec<Complex>, String>,
    ) -> Result<Arc<Vec<Complex>>, String> {
        let mut storage = self
            .storage
            .lock()
            .map_err(|_| "amplitude cache lock poisoned".to_string())?;
        if let Storage::Device(state) = &*storage {
            let host = copy(state)?;
            CACHED_BYTES.fetch_add(
                (host.len() * std::mem::size_of::<Complex>()) as u64,
                Ordering::Relaxed,
            );
            *storage = Storage::Host(Arc::new(host));
            RESIDENT_BYTES.fetch_sub(self.device_bytes, Ordering::Relaxed);
            MATERIALIZATIONS.fetch_add(1, Ordering::Relaxed);
        }
        match &*storage {
            Storage::Host(host) => Ok(Arc::clone(host)),
            Storage::Device(_) => unreachable!("successful materialization stores host amplitudes"),
        }
    }
}

impl<S> Drop for LazyAmplitudes<S> {
    fn drop(&mut self) {
        // Poisoning does not change resource ownership; still account for cleanup.
        let storage = self.storage.get_mut().unwrap_or_else(|e| e.into_inner());
        match storage {
            Storage::Device(_) => {
                RESIDENT_BYTES.fetch_sub(self.device_bytes, Ordering::Relaxed);
            }
            Storage::Host(host) => {
                CACHED_BYTES.fetch_sub(
                    (host.len() * std::mem::size_of::<Complex>()) as u64,
                    Ordering::Relaxed,
                );
            }
        }
    }
}

/// Process-wide diagnostics for Python-owned results (logical buffer bytes,
/// excluding allocator pools, scratch, NumPy arrays and transient copy buffers).
pub(crate) fn stats() -> (u64, u64, u64) {
    (
        MATERIALIZATIONS.load(Ordering::Relaxed),
        RESIDENT_BYTES.load(Ordering::Relaxed),
        CACHED_BYTES.load(Ordering::Relaxed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    struct State(Arc<AtomicU64>);
    impl Drop for State {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn failed_copy_retries_and_success_releases_device() {
        let drops = Arc::new(AtomicU64::new(0));
        let lazy = LazyAmplitudes::new(State(drops.clone()), 8);
        assert!(lazy.materialize(|_| Err("copy failed".into())).is_err());
        assert_eq!(drops.load(Ordering::Relaxed), 0);
        let first = lazy
            .materialize(|_| Ok(vec![Complex::new(1.0, 0.0)]))
            .unwrap();
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        let mut caller_copy = (*first).clone();
        caller_copy[0] = Complex::new(0.0, 0.0);
        assert_eq!(
            lazy.materialize(|_| panic!("cached read copied device"))
                .unwrap()[0],
            Complex::new(1.0, 0.0)
        );
        drop(lazy);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn counts_only_drop_releases_device() {
        let drops = Arc::new(AtomicU64::new(0));
        drop(LazyAmplitudes::new(State(drops.clone()), 8));
        assert_eq!(drops.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn concurrent_access_copies_once() {
        let lazy = Arc::new(LazyAmplitudes::new((), 0));
        let copies = Arc::new(AtomicU64::new(0));
        let threads: Vec<_> = (0..4)
            .map(|_| {
                let lazy = lazy.clone();
                let copies = copies.clone();
                std::thread::spawn(move || {
                    lazy.materialize(|_| {
                        copies.fetch_add(1, Ordering::Relaxed);
                        Ok(vec![Complex::new(1.0, 0.0)])
                    })
                    .unwrap()
                })
            })
            .collect();
        for thread in threads {
            assert_eq!(thread.join().unwrap()[0], Complex::new(1.0, 0.0));
        }
        assert_eq!(copies.load(Ordering::Relaxed), 1);
        let _ = stats();
    }
}
