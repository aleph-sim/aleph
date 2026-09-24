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

    /// Error propagation through the stabilizer engine failed (e.g. a
    /// non-Clifford gate in a circuit handed to the DEM builder).
    #[error("stabilizer propagation failed: {0}")]
    Propagation(String),

    /// An external decoder oracle (e.g. the PyMatching subprocess) failed to run
    /// or returned malformed output.
    #[error("decoder oracle failed: {0}")]
    Oracle(String),

    /// A DEM error mechanism flips more than two detectors (a hyperedge), so the model is not
    /// graph-like and cannot be turned into a matching graph. Matching decoders (MWPM, UF)
    /// need a graph-like DEM; hyperedges require decomposition or a hypergraph decoder (qLDPC,
    /// Q5). The surface-code memory DEMs from Q0-03 are graph-like by construction.
    #[error("non-graphlike DEM: an error mechanism flips {dets} detectors (>2); matching needs a graph-like DEM")]
    NonGraphlike {
        /// Number of detectors the offending mechanism flips.
        dets: usize,
    },

    /// A check-matrix (`H`/`O`/priors) input to
    /// [`DetectorErrorModel::from_check_matrices`](crate::DetectorErrorModel::from_check_matrices)
    /// is malformed (length mismatch, bad index, duplicate index, bad probability).
    #[error("invalid check matrices: {0}")]
    CheckMatrix(String),

    /// More logical observables than the decoders' `u64` observable masks can hold.
    #[error("aleph decoders support at most 64 logical observables, got {observables}")]
    TooManyObservables {
        /// Number of observables requested.
        observables: usize,
    },
}

impl From<aleph_stab::StabError> for Error {
    fn from(e: aleph_stab::StabError) -> Self {
        Error::Propagation(e.to_string())
    }
}

/// Convenience alias for results in this crate.
pub type Result<T> = std::result::Result<T, Error>;
