//! [`PyMatchingOracle`] — an external minimum-weight-perfect-matching decoder that shells out
//! to [PyMatching](https://github.com/oscarhiggott/PyMatching).
//!
//! It exists so the Q0-04 harness produces a *real* threshold before our own MWPM decoder lands
//! in Q1: PyMatching is the reference MWPM implementation everyone benchmarks against. It is a
//! placeholder in the sense that it is not part of the shipped decoder set — it is a trusted
//! oracle for validating the harness and, later, our native decoders.
//!
//! # How it works
//!
//! Our [`DetectorErrorModel`] already emits Stim-compatible `.dem` text
//! ([`DetectorErrorModel::to_dem_string`]). On each [`decode_batch`](Decoder::decode_batch) we
//! spawn one Python process that builds `pymatching.Matching.from_detector_error_model(...)` and
//! calls its vectorised `decode_batch` over the whole shot set — one process per batch, not per
//! shot. The DEM and the packed syndrome matrix go in over stdin; the packed prediction matrix
//! comes back over stdout. PyMatching's matching is deterministic, so a fixed shot set decodes
//! identically every time.
//!
//! # Requirements
//!
//! A Python with `numpy`, `stim`, and `pymatching`. The interpreter is taken from the
//! `PYMATCHING_PYTHON` env var, falling back to `STIM_PYTHON`, then `python3`. Tests that use it
//! are `#[ignore]`d so the default `cargo test` stays hermetic.

use std::io::Write;
use std::process::{Command, Stdio};

use crate::decoder::Decoder;
use crate::dem::DetectorErrorModel;
use crate::error::{Error, Result};
use crate::syndrome::{Correction, Syndrome};

/// The Python driver: read DEM + packed syndromes from stdin, write packed predictions to
/// stdout. Kept inline so the oracle has no on-disk dependency.
const DRIVER: &str = r#"
import sys, numpy as np, stim, pymatching

buf = sys.stdin.buffer.read()
off = 0
def u64():
    global off
    v = int.from_bytes(buf[off:off+8], "little"); off += 8; return v

dem_len = u64()
dem_text = buf[off:off+dem_len].decode("utf-8"); off += dem_len
shots = u64()
ndet = u64()
nobs = u64()

synd = np.frombuffer(buf[off:off+shots*ndet], dtype=np.uint8)
synd = synd.reshape(shots, ndet) if shots else np.zeros((0, ndet), dtype=np.uint8)

m = pymatching.Matching.from_detector_error_model(stim.DetectorErrorModel(dem_text))
pred = m.decode_batch(synd) if shots else np.zeros((0, nobs), dtype=np.uint8)
pred = np.ascontiguousarray(pred, dtype=np.uint8).reshape(shots, nobs)
sys.stdout.buffer.write(pred.tobytes())
"#;

/// An MWPM decoder backed by an external PyMatching process.
///
/// Construct it once from a [`DetectorErrorModel`]; it caches the DEM text and the detector /
/// observable counts and re-uses them for every batch.
#[derive(Clone, Debug)]
pub struct PyMatchingOracle {
    dem_text: String,
    detectors: usize,
    observables: usize,
}

impl PyMatchingOracle {
    /// Build an oracle for `dem`. Cheap — it only serialises the DEM; the Python process is not
    /// spawned until [`decode_batch`](Decoder::decode_batch) runs.
    pub fn new(dem: &DetectorErrorModel) -> Self {
        PyMatchingOracle {
            dem_text: dem.to_dem_string(),
            detectors: dem.detectors,
            observables: dem.observables,
        }
    }

    /// The interpreter to run, from `PYMATCHING_PYTHON`, then `STIM_PYTHON`, then `python3`.
    fn python() -> String {
        std::env::var("PYMATCHING_PYTHON")
            .or_else(|_| std::env::var("STIM_PYTHON"))
            .unwrap_or_else(|_| "python3".to_string())
    }

    /// Spawn PyMatching once over the whole batch. Separate from the trait method so the
    /// `Result` is visible to callers that want to handle oracle failures explicitly.
    pub fn decode_all(&self, syndromes: &[Syndrome]) -> Result<Vec<Correction>> {
        let shots = syndromes.len();
        if shots == 0 {
            return Ok(Vec::new());
        }

        // Pack the request: dem_len | dem | shots | ndet | nobs | (shots × ndet) syndrome bits.
        let mut input = Vec::with_capacity(40 + self.dem_text.len() + shots * self.detectors);
        input.extend_from_slice(&(self.dem_text.len() as u64).to_le_bytes());
        input.extend_from_slice(self.dem_text.as_bytes());
        input.extend_from_slice(&(shots as u64).to_le_bytes());
        input.extend_from_slice(&(self.detectors as u64).to_le_bytes());
        input.extend_from_slice(&(self.observables as u64).to_le_bytes());
        let mut row = vec![0u8; self.detectors];
        for s in syndromes {
            row.iter_mut().for_each(|b| *b = 0);
            for &d in &s.fired {
                if (d as usize) < self.detectors {
                    row[d as usize] = 1;
                }
            }
            input.extend_from_slice(&row);
        }

        let python = Self::python();
        let mut child = Command::new(&python)
            .args(["-c", DRIVER])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| Error::Oracle(format!("spawning `{python}`: {e}")))?;

        // Write the request on a thread so a large request can't deadlock against a child that
        // starts writing stdout before it has drained stdin.
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| Error::Oracle("child stdin unavailable".into()))?;
        let writer =
            std::thread::spawn(move || stdin.write_all(&input).and_then(|_| stdin.flush()));

        let output = child
            .wait_with_output()
            .map_err(|e| Error::Oracle(format!("waiting for `{python}`: {e}")))?;
        writer
            .join()
            .map_err(|_| Error::Oracle("stdin writer thread panicked".into()))?
            .map_err(|e| Error::Oracle(format!("writing to `{python}` stdin: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(Error::Oracle(format!(
                "`{python}` exited with {}: {}",
                output.status,
                stderr.trim()
            )));
        }

        let expected = shots * self.observables;
        if output.stdout.len() != expected {
            return Err(Error::Oracle(format!(
                "expected {expected} prediction bytes ({shots}×{}), got {}",
                self.observables,
                output.stdout.len()
            )));
        }

        Ok(output
            .stdout
            .chunks_exact(self.observables.max(1))
            .take(shots)
            .map(|row| Correction::new(row.iter().map(|&b| b != 0).collect()))
            .collect())
    }
}

impl Decoder for PyMatchingOracle {
    /// Single-syndrome convenience. The harness uses [`decode_batch`](Decoder::decode_batch);
    /// this exists only for the trait contract and, on a subprocess failure it cannot report,
    /// falls back to predicting no flip.
    fn decode(&self, syndrome: &Syndrome) -> Correction {
        match self.decode_all(std::slice::from_ref(syndrome)) {
            Ok(mut v) if !v.is_empty() => v.remove(0),
            _ => Correction::none(self.observables),
        }
    }

    fn decode_batch(&self, syndromes: &[Syndrome]) -> Result<Vec<Correction>> {
        self.decode_all(syndromes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dem::DemError;

    #[test]
    fn empty_batch_needs_no_subprocess() {
        let dem = DetectorErrorModel {
            detectors: 2,
            observables: 1,
            errors: vec![DemError::new(0.1, vec![0, 1], vec![0])],
        };
        let oracle = PyMatchingOracle::new(&dem);
        // No Python required for an empty batch.
        assert_eq!(oracle.decode_all(&[]).unwrap(), Vec::new());
    }

    #[test]
    fn caches_dem_and_counts() {
        let dem = DetectorErrorModel {
            detectors: 5,
            observables: 2,
            errors: vec![DemError::new(0.1, vec![0, 3], vec![1])],
        };
        let oracle = PyMatchingOracle::new(&dem);
        assert_eq!(oracle.detectors, 5);
        assert_eq!(oracle.observables, 2);
        assert!(oracle.dem_text.contains("error(0.1) D0 D3 L1"));
    }
}
