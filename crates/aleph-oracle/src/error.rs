//! Errors raised by the oracle harness itself (file I/O, schema
//! drift, dimension mismatch). Correctness-failure diagnostics are
//! panics inside `#[test]`, not `OracleError` values.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum OracleError {
    #[error("fixture I/O at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("fixture {name} has unsupported schema_version {found} (expected {expected})")]
    SchemaVersion {
        name: String,
        found: u32,
        expected: u32,
    },

    #[error(
        "fixture {name} qubit-count mismatch: fixture says {fixture}, parsed circuit has {circuit}"
    )]
    QubitMismatch {
        name: String,
        fixture: u32,
        circuit: u32,
    },

    #[error(
        "fixture {name} dimension mismatch: fixture has {fixture} amplitudes, state has {state}"
    )]
    DimensionMismatch {
        name: String,
        fixture: usize,
        state: usize,
    },

    #[error("fixture {name} declares endianness {endianness:?}, only \"little\" is supported")]
    UnsupportedEndianness { name: String, endianness: String },

    #[error("malformed fixture JSON: {0}")]
    Json(#[from] serde_json::Error),

    #[error("backend error: {0}")]
    Backend(#[from] aleph_backend::BackendError),

    #[error("parse error: {0}")]
    Parse(#[from] aleph_parser::ParseError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_version_message_contains_fields() {
        let e = OracleError::SchemaVersion {
            name: "ghz_3".into(),
            found: 7,
            expected: 1,
        };
        let s = format!("{e}");
        assert!(s.contains("ghz_3"));
        assert!(s.contains("7"));
        assert!(s.contains("1"));
    }

    #[test]
    fn dimension_mismatch_message_contains_fields() {
        let e = OracleError::DimensionMismatch {
            name: "kernel_cx".into(),
            fixture: 4,
            state: 8,
        };
        let s = format!("{e}");
        assert!(s.contains("kernel_cx"));
        assert!(s.contains("4"));
        assert!(s.contains("8"));
    }
}
