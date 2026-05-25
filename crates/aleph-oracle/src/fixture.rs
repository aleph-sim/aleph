//! Fixture JSON schema + loader. Format is defined in
//! `docs/superpowers/specs/2026-05-24-p0-10-oracle-qiskit-design.md` §4.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::OracleError;

/// Schema version this build of the harness understands.
pub const SCHEMA_VERSION: u32 = 1;

/// One fixture JSON file, deserialized as-is.
#[derive(Debug, Clone, Deserialize)]
pub struct Fixture {
    pub schema_version: u32,
    pub name: String,
    pub num_qubits: u32,
    pub qasm_path: String,
    pub qiskit_version: String,
    pub aer_version: String,
    pub generated_at: String,
    pub shots: u64,
    pub rng_seed: u64,
    pub statevector: StateVectorFixture,
    pub counts: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StateVectorFixture {
    pub endianness: String,
    pub amplitudes: Vec<(f64, f64)>,
}

/// Load a fixture from disk and validate the schema version + endianness.
pub fn load_fixture(path: &Path) -> Result<Fixture, OracleError> {
    let bytes = std::fs::read(path).map_err(|source| OracleError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let fx: Fixture = serde_json::from_slice(&bytes)?;
    if fx.schema_version != SCHEMA_VERSION {
        return Err(OracleError::SchemaVersion {
            name: fx.name,
            found: fx.schema_version,
            expected: SCHEMA_VERSION,
        });
    }
    if fx.statevector.endianness != "little" {
        return Err(OracleError::UnsupportedEndianness {
            name: fx.name,
            endianness: fx.statevector.endianness,
        });
    }
    Ok(fx)
}

/// Load a `.qasm` source file (UTF-8) from disk.
pub fn load_qasm(path: &Path) -> Result<String, OracleError> {
    std::fs::read_to_string(path).map_err(|source| OracleError::Io {
        path: path.to_path_buf(),
        source,
    })
}

/// Workspace-root-relative resolver: given a crate-relative tail like
/// `"oracle/fixtures/ghz_10.json"`, return an absolute path rooted at
/// the workspace root. Brittle to crate-tree moves; centralized so a
/// future restructuring touches one function.
pub fn workspace_path(tail: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // CARGO_MANIFEST_DIR points at crates/aleph-oracle; pop twice.
    p.pop();
    p.pop();
    p.push(tail);
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_fixture_json() -> String {
        r#"{
            "schema_version": 1,
            "name": "synthetic",
            "num_qubits": 1,
            "qasm_path": "circuits/synthetic.qasm",
            "qiskit_version": "test",
            "aer_version": "test",
            "generated_at": "1970-01-01T00:00:00Z",
            "shots": 0,
            "rng_seed": 0,
            "statevector": {
                "endianness": "little",
                "amplitudes": [[1.0, 0.0], [0.0, 0.0]]
            },
            "counts": {}
        }"#
        .to_string()
    }

    /// Tiny stdlib-only temp directory helper, RAII-cleaned on drop.
    struct TestTempDir {
        path: PathBuf,
    }
    impl TestTempDir {
        fn new() -> Self {
            let mut p = std::env::temp_dir();
            p.push(format!(
                "aleph-oracle-test-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir(&p).unwrap();
            Self { path: p }
        }
    }
    impl Drop for TestTempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn deserializes_valid_fixture() {
        let fx: Fixture = serde_json::from_str(&synth_fixture_json()).unwrap();
        assert_eq!(fx.name, "synthetic");
        assert_eq!(fx.statevector.amplitudes.len(), 2);
        assert_eq!(fx.statevector.amplitudes[0], (1.0, 0.0));
    }

    #[test]
    fn load_fixture_round_trip() {
        let tmp = TestTempDir::new();
        let path = tmp.path.join("fx.json");
        std::fs::write(&path, synth_fixture_json()).unwrap();
        let fx = load_fixture(&path).unwrap();
        assert_eq!(fx.name, "synthetic");
    }

    #[test]
    fn load_fixture_rejects_bad_schema_version() {
        let tmp = TestTempDir::new();
        let json =
            synth_fixture_json().replace("\"schema_version\": 1", "\"schema_version\": 999");
        let path = tmp.path.join("fx.json");
        std::fs::write(&path, json).unwrap();
        let err = load_fixture(&path).unwrap_err();
        match err {
            OracleError::SchemaVersion {
                found: 999,
                expected: 1,
                ..
            } => {}
            other => panic!("expected SchemaVersion error, got {other:?}"),
        }
    }

    #[test]
    fn load_fixture_rejects_big_endian() {
        let tmp = TestTempDir::new();
        let json = synth_fixture_json().replace("\"little\"", "\"big\"");
        let path = tmp.path.join("fx.json");
        std::fs::write(&path, json).unwrap();
        let err = load_fixture(&path).unwrap_err();
        match err {
            OracleError::UnsupportedEndianness { endianness, .. } => {
                assert_eq!(endianness, "big");
            }
            other => panic!("expected UnsupportedEndianness, got {other:?}"),
        }
    }

    use proptest::prelude::*;

    /// Returns the size of one ulp at `x` (the spacing between `x` and
    /// the next representable `f64`). For subnormals and zero this
    /// degenerates to `f64::MIN_POSITIVE`.
    fn one_ulp(x: f64) -> f64 {
        let bits = x.to_bits();
        let next = f64::from_bits(bits.wrapping_add(1));
        (next - x).abs().max(f64::MIN_POSITIVE)
    }

    proptest! {
        /// Every finite f64 round-trips through `serde_json` to within
        /// **1 ulp**. `serde_json` is not bit-exact (it drifts ≤ 1 ulp
        /// for some inputs); see spec §10.5. For amplitudes in a
        /// normalized state vector this is ≤ `2.22e-16`, six orders
        /// of magnitude below `STATE_TOLERANCE = 1e-10`.
        #[test]
        fn f64_pair_round_trips_through_serde_json_within_1_ulp(
            re in proptest::num::f64::ANY,
            im in proptest::num::f64::ANY,
        ) {
            prop_assume!(re.is_finite() && im.is_finite());
            let s = serde_json::to_string(&(re, im)).unwrap();
            let (re2, im2): (f64, f64) = serde_json::from_str(&s).unwrap();
            let re_err = (re - re2).abs();
            let im_err = (im - im2).abs();
            prop_assert!(
                re_err <= one_ulp(re),
                "re drift > 1 ulp: re={re:e}, re2={re2:e}, err={re_err:e}, ulp={:e}",
                one_ulp(re)
            );
            prop_assert!(
                im_err <= one_ulp(im),
                "im drift > 1 ulp: im={im:e}, im2={im2:e}, err={im_err:e}, ulp={:e}",
                one_ulp(im)
            );
        }
    }
}
