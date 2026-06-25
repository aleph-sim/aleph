//! Crate-local error type. Per CLAUDE.md, library code never `panic!`s on input and never
//! uses `unwrap`/`expect` outside tests — fallible operations return [`Result`].

/// Errors produced by `aleph-qec`.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A Detector Error Model text line could not be parsed.
    #[error("DEM parse error at line {line}: {msg}")]
    DemParse {
        /// 1-based line number in the input.
        line: usize,
        /// Human-readable reason.
        msg: String,
    },

    /// A DEM instruction that the Q0-01 flat-DEM subset does not yet support
    /// (`repeat` blocks and `shift_detectors`). Full support lands with Q0-03, when we
    /// consume Stim-emitted DEMs directly.
    #[error("unsupported DEM instruction at line {line}: `{what}` (Q0-01 supports flat error/detector/logical_observable only)")]
    UnsupportedDem {
        /// 1-based line number in the input.
        line: usize,
        /// The instruction keyword that is not supported.
        what: String,
    },

    /// Error propagation through the stabilizer engine failed (e.g. a
    /// non-Clifford gate in a circuit handed to the DEM builder).
    #[error("stabilizer propagation failed: {0}")]
    Propagation(String),

    /// An external decoder oracle (e.g. the PyMatching subprocess) failed to run
    /// or returned malformed output.
    #[error("decoder oracle failed: {0}")]
    Oracle(String),
}

impl From<aleph_stab::StabError> for Error {
    fn from(e: aleph_stab::StabError) -> Self {
        Error::Propagation(e.to_string())
    }
}

/// Convenience alias for results in this crate.
pub type Result<T> = std::result::Result<T, Error>;
