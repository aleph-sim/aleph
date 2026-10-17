//! Crate-local error type. Per CLAUDE.md, library code never `panic!`s on input and never
//! uses `unwrap`/`expect` outside tests — fallible operations return [`Result`].

/// Errors produced by `aleph-qec`.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Invalid or numerically unsupported rare-event plan or experiment.
    #[error("rare-event experiment: {0}")]
    RareEvent(String),

    /// Invalid, incompatible or inconsistent failure-witness data.
    #[error("failure witness: {0}")]
    Witness(String),

    /// Matching decoders require finite per-mechanism priors in [0, 0.5].
    #[error("matching prior at column {column} must be finite and in [0, 0.5], got {prob}")]
    InvalidMatchingPrior {
        /// Zero-based column in the DEM used to build the graph (split columns if split first).
        column: usize,
        /// Rejected input probability, retained without clamping.
        prob: f64,
    },

    /// The experiment's sampling/decode batch limit must be positive and fit in u64.
    #[error("DEM experiment batch size must be positive and fit in u64")]
    InvalidExperimentBatchSize,

    /// Experiment accounting would exceed the u64 shot-count representation.
    #[error("DEM experiment shot/error count overflow")]
    ExperimentCountOverflow,

    /// A decoder returned a wrong batch length or observable width.
    #[error("decoder contract violation at absolute shot {shot}: {message}")]
    DecoderContract {
        /// Zero-based offending shot, or start of a batch with a wrong prediction count.
        shot: u64,
        /// Expected and actual output dimensions.
        message: String,
    },

    /// A Bernoulli tally has more failures than trials.
    #[error("invalid Bernoulli tally: {errors} errors exceeds {shots} shots")]
    InvalidTally {
        /// Number of trials.
        shots: u64,
        /// Number of failures.
        errors: u64,
    },

    /// Numerical evaluation of a confidence bound could not be validated.
    #[error(
        "binomial interval could not be evaluated reliably for {shots} shots and {errors} errors"
    )]
    IntervalNumerics {
        /// Number of trials.
        shots: u64,
        /// Number of failures.
        errors: u64,
    },
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
