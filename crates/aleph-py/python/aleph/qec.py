"""QEC decoding: stim-format Detector Error Models and aleph's decoders.

    import aleph.qec as qec
    dem = qec.DetectorErrorModel(stim_circuit.detector_error_model(decompose_errors=True))
    predictions = qec.Decoder(dem, "relay-bp-osd").decode_batch(detection_events)

``decoder_names()`` lists the available decoders. stim is not required. One rule to know up
front: ``Decoder.decode_batch_errors`` raises for the matching decoders (``mwpm``,
``union-find``, ``union-find-weighted``) on a DEM with ``^``-decomposed mechanisms — split the
parts first with ``dem_to_matrices`` + ``dem_from_matrices`` (BP-family decoders have no such
restriction).
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
            # `copy=True`: sp.csc_matrix(m) returns `m` itself when it is already CSC, so
            # sum_duplicates()/eliminate_zeros() below would otherwise canonicalise the
            # caller's own matrix in place.
            c = sp.csc_matrix(m, copy=True)
            c.sum_duplicates()
            c.eliminate_zeros()
            return c.shape, c.indptr.astype(np.int64), c.indices.astype(np.int64)
    except ImportError:
        pass
    a = np.asarray(m)
    if a.ndim != 2:
        raise ValueError(f"{what}: expected a 2-D matrix, got shape {a.shape}")
    col_idx, row_idx = np.nonzero(a.T)    # sorted by column, then row
    indptr = np.searchsorted(col_idx, np.arange(a.shape[1] + 1)).astype(np.int64)
    return a.shape, indptr, row_idx.astype(np.int64)


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


def dem_to_matrices(dem):
    """Stim DEM (``stim.DetectorErrorModel``, DEM text, or ``aleph.qec.DetectorErrorModel``) ->
    ``(H, O, error_rate_vec)`` with every ``^``-separated part of a mechanism as its own column,
    so a matching decoder (which needs a graph-like, one-part-per-column model) can take the
    result as ``H``. ``H`` is ``uint8 (detectors, E)``, ``O`` is ``uint8 (observables, E)``,
    ``error_rate_vec`` is ``float64 (E,)``. Pure Python/numpy — no ``stim`` or ``cudaq_qec``
    dependency; re-exported as ``aleph.cudaq.dem_to_matrices`` for the cudaq plugin path.
    """
    adem = dem if isinstance(dem, DetectorErrorModel) else DetectorErrorModel(dem)
    cols = []  # (prob, dets, obs)
    for line in adem.to_dem_string().splitlines():
        s = line.strip()
        if not s.startswith("error("):
            continue
        head, _, targets = s.partition(")")
        p = float(head[len("error("):])
        for part in targets.split("^"):
            dets = [int(t[1:]) for t in part.split() if t[0] == "D"]
            obs = [int(t[1:]) for t in part.split() if t[0] == "L"]
            cols.append((p, dets, obs))
    D, L, E = adem.num_detectors, adem.num_observables, len(cols)
    H = np.zeros((D, E), dtype=np.uint8)
    O = np.zeros((L, E), dtype=np.uint8)
    rates = np.empty(E, dtype=np.float64)
    for j, (p, dets, obs) in enumerate(cols):
        # Fancy-index augmented assignment (`H[dets, j] ^= 1`) gathers, XORs once, then
        # scatters — a target repeated within one `^`-part would silently NOT cancel (it
        # would just be set), unlike Stim's actual semantics (repeat = no flip). The
        # unbuffered ufunc form accumulates in place over duplicate indices.
        np.bitwise_xor.at(H[:, j], dets, 1)
        np.bitwise_xor.at(O[:, j], obs, 1)
        rates[j] = p
    return H, O, rates


__all__ = [
    "DetectorErrorModel", "Decoder", "decoder_names", "dem_from_matrices", "dem_to_matrices",
    "gross_code_dem",
]
