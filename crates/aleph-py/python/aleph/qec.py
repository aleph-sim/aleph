"""QEC decoding: stim-format Detector Error Models and aleph's decoders.

    import aleph.qec as qec
    dem = qec.DetectorErrorModel(stim_circuit.detector_error_model(decompose_errors=True))
    predictions = qec.Decoder(dem, "relay-bp-osd").decode_batch(detection_events)

``decoder_names()`` lists the available decoders. stim is not required.
"""
import numpy as np

from . import _native

DetectorErrorModel = _native.qec.DetectorErrorModel
Decoder = _native.qec.Decoder
decoder_names = _native.qec.decoder_names
gross_code_dem = _native.qec.gross_code_dem


def _csc(m, what):
    """(indptr, indices) int64 of the non-zero pattern of a dense or scipy.sparse 2-D matrix."""
    try:
        import scipy.sparse as sp
        if sp.issparse(m):
            c = sp.csc_matrix(m)
            c.sum_duplicates()
            c.eliminate_zeros()
            return c.shape, c.indptr.astype(np.int64), c.indices.astype(np.int64)
    except ImportError:
        pass
    a = np.asarray(m)
    if a.ndim != 2:
        raise ValueError(f"{what}: expected a 2-D matrix, got shape {a.shape}")
    rows, cols = np.nonzero(a.T)          # sorted by column, then row
    indptr = np.searchsorted(rows, np.arange(a.shape[1] + 1)).astype(np.int64)
    return a.shape, indptr, cols.astype(np.int64)


def dem_from_matrices(H, O=None, error_rate_vec=None):
    """Build a DetectorErrorModel from a parity-check matrix ``H`` (detectors x errors), an
    optional observable matrix ``O`` (observables x errors) and one prior per column.

    ``H``/``O`` may be dense arrays (any dtype; non-zero means 1) or ``scipy.sparse`` matrices
    (never densified). Column ``j`` becomes one error mechanism. This is the input a
    ``cudaq_qec`` decoder is constructed from.
    """
    if error_rate_vec is None:
        raise ValueError("error_rate_vec is required (one probability per column of H)")
    (d, e), h_ptr, h_idx = _csc(H, "H")
    if O is None:
        n_obs, o_ptr, o_idx = 0, np.zeros(e + 1, dtype=np.int64), np.zeros(0, dtype=np.int64)
    else:
        (n_obs, e_o), o_ptr, o_idx = _csc(O, "O")
        if e_o != e:
            raise ValueError(f"O has {e_o} columns but H has {e}")
    probs = np.ascontiguousarray(np.asarray(error_rate_vec, dtype=np.float64).ravel())
    if probs.shape[0] != e:
        raise ValueError(f"error_rate_vec has {probs.shape[0]} entries but H has {e} columns")
    return _native.qec._dem_from_csc(int(d), int(n_obs), h_ptr, h_idx, o_ptr, o_idx, probs)


__all__ = ["DetectorErrorModel", "Decoder", "decoder_names", "dem_from_matrices", "gross_code_dem"]
