"""aleph decoders as CUDA-Q QEC (``cudaq_qec``) decoders.

    pip install "aleph-sim[cudaq]"

    import cudaq_qec as qec
    import aleph.cudaq                      # registers aleph-mwpm, aleph-relay-bp-osd, ...
    dec = qec.get_decoder("aleph-relay-bp-osd", H, O=O, error_rate_vec=rates, osd_order=12)
    res = dec.decode_batch(syndromes)       # res.result: (shots, E) float64, > 0.5 == error

Every decoder takes the parity-check matrix ``H`` (dense or scipy.sparse), the observable
matrix ``O`` and one prior per column (``error_rate_vec``, or a scalar ``error_rate``), plus
the same keyword parameters as ``aleph.qec.Decoder``. Results are per-column error estimates,
so ``(O @ (res.result > 0.5).T) % 2`` gives the predicted observable flips, exactly as with
cudaq's own decoders. Matching decoders need a graph-like ``H`` (every column has at most two
ones): cudaq keeps ``^``-decomposed hyperedges of a Stim DEM as one column, so pass
``dem_to_matrices(dem)`` instead of the DEM string for ``aleph-mwpm`` / ``aleph-union-find*``.
Matching decoders (``aleph-mwpm``, ``aleph-union-find``, ``aleph-union-find-weighted``) also
require every prior (``error_rate_vec`` entry, or the scalar ``error_rate``) to be <= 0.5
(matching on a negative-weight edge is undefined here); BP-family decoders (``aleph-bp``,
``aleph-bp-osd``, ``aleph-relay-bp``, ``aleph-relay-bp-osd``) have no such restriction.
"""
import cudaq_qec as _qec  # the only cudaq import in aleph; ImportError means "install the extra"
import numpy as np

import aleph.qec as _aq
from aleph.qec import dem_to_matrices  # re-exported: aleph.cudaq.dem_to_matrices (no cudaq dep)

NAMES = tuple(_aq.decoder_names())

_MATCHING = {"mwpm", "union-find", "union-find-weighted"}

__all__ = ["NAMES", "DECODERS", "dem_to_matrices"]


def _rates(n, error_rate_vec, error_rate):
    if error_rate_vec is not None and error_rate is not None:
        raise ValueError("pass either error_rate_vec or error_rate, not both")
    if error_rate_vec is not None:
        r = np.asarray(error_rate_vec, dtype=np.float64).ravel()
        if r.shape[0] != n:
            raise ValueError(f"error_rate_vec has {r.shape[0]} entries but H has {n} columns")
        bad = np.flatnonzero(~np.isfinite(r))
        if bad.size:
            raise ValueError(f"error_rate_vec[{int(bad[0])}] = {r[bad[0]]!r} is not finite")
        return r
    if error_rate is not None:
        v = float(error_rate)
        if not np.isfinite(v):
            raise ValueError(f"error_rate must be a finite number, got {v!r}")
        return np.full(n, v)
    raise ValueError("aleph decoders need a prior per column: pass error_rate_vec=[...] "
                     "(cudaq fills it in from a DEM string) or a scalar error_rate=")


def _to_bits(syndromes, width):
    """float syndromes (rows) -> C-contiguous uint8 (shots, width); values > 0.5 count as fired."""
    a = np.asarray(syndromes, dtype=np.float64)
    if a.size == 0:
        return np.zeros((0, width), dtype=np.uint8)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != width:
        raise ValueError(f"syndrome width {a.shape[-1] if a.ndim else 0} != {width} detectors")
    return np.ascontiguousarray(a > 0.5).astype(np.uint8)


def _make(name):
    @_qec.decoder(f"aleph-{name}")
    class AlephDecoder:
        def __init__(self, H, O=None, error_rate_vec=None, error_rate=None, **params):
            _qec.Decoder.__init__(self, H)
            n = H.shape[1]
            rates = _rates(n, error_rate_vec, error_rate)
            if name in _MATCHING and rates.size:
                j = int(np.argmax(rates))
                if rates[j] > 0.5:
                    raise ValueError(
                        f"aleph-{name}: priors must be <= 0.5 for a matching decoder "
                        f"(max {rates[j]:.3g} at column {j}); clamp or use a BP-family decoder")
            try:
                dem = _aq.dem_from_matrices(H, O, rates)
                self._inner = _aq.Decoder(dem, name, **params)
            except ValueError as e:
                if "non-graphlike" in str(e):
                    raise ValueError(
                        f"aleph-{name} needs a graph-like H (every column with at most two "
                        f"ones): {e}. Pass the decomposed DEM as H (aleph.cudaq.dem_to_matrices), "
                        "not as a DEM string.") from None
                raise
            self._width = H.shape[0]

        def decode(self, syndrome):
            a = np.asarray(syndrome, dtype=np.float64)
            if a.size != self._width:
                raise ValueError(f"syndrome width {a.size} != {self._width} detectors")
            ehat, conv = self._inner.decode_batch_errors(_to_bits(a.reshape(1, -1), self._width))
            r = _qec.DecoderResult()
            r.converged = bool(conv[0])
            r.result = ehat[0].astype(np.float64).tolist()
            r.opt_results = None
            return r

        def decode_batch(self, syndromes):
            ehat, conv = self._inner.decode_batch_errors(_to_bits(syndromes, self._width))
            # `ehat` is already `(0, E)` for an empty batch (Step-0 probe: cudaq's own
            # `(O @ err.T)` breaks on a `(0, 0)` result), so it is returned as-is rather than
            # overridden to `(0, 0)`.
            return _qec.BatchDecoderResult(result=ehat.astype(np.float64), converged=conv,
                                           opt_results=None, batch_opt_results=None)

    AlephDecoder.__name__ = AlephDecoder.__qualname__ = f"Aleph_{name.replace('-', '_')}"
    return AlephDecoder


DECODERS = {name: _make(name) for name in NAMES}
