//! `aleph.qec` bindings: [`DetectorErrorModel`] parsing and the DEM decoders of `aleph-qec`,
//! with numpy batch decoding that releases the GIL (see `qec_core`).
// pyo3 0.22 proc-macro expansion emits trivial PyErr->PyErr .into() calls.
#![allow(clippy::useless_conversion)]

use crate::qec_core::{self, AnyDecoder, DecoderParams, DECODER_NAMES};
use aleph_qec::DetectorErrorModel;
use numpy::{
    PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn value_err(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

/// A stim-format Detector Error Model (`repeat`/`shift_detectors`/`^` supported).
#[pyclass(name = "DetectorErrorModel", module = "aleph.qec", frozen)]
pub struct PyDem {
    inner: DetectorErrorModel,
}

#[pymethods]
impl PyDem {
    /// Parse DEM text; any object whose `str()` is DEM text (e.g. `stim.DetectorErrorModel`) works.
    #[new]
    fn new(model: &Bound<'_, PyAny>) -> PyResult<Self> {
        let text: String = match model.extract::<String>() {
            Ok(s) => s,
            Err(_) => model.str()?.extract()?,
        };
        DetectorErrorModel::parse(&text)
            .map(|inner| PyDem { inner })
            .map_err(|e| value_err(e.to_string()))
    }

    /// Read and parse a `.dem` file.
    #[staticmethod]
    fn from_file(path: std::path::PathBuf) -> PyResult<Self> {
        let text = std::fs::read_to_string(&path)
            .map_err(|e| PyOSError::new_err(format!("read {}: {e}", path.display())))?;
        DetectorErrorModel::parse(&text)
            .map(|inner| PyDem { inner })
            .map_err(|e| value_err(e.to_string()))
    }

    #[getter]
    fn num_detectors(&self) -> usize {
        self.inner.detectors
    }
    #[getter]
    fn num_observables(&self) -> usize {
        self.inner.observables
    }
    #[getter]
    fn num_errors(&self) -> usize {
        self.inner.errors.len()
    }
    /// The flat model as DEM text (`repeat` blocks unrolled).
    fn to_dem_string(&self) -> String {
        self.inner.to_dem_string()
    }
    fn __repr__(&self) -> String {
        format!(
            "DetectorErrorModel(detectors={}, observables={}, errors={})",
            self.inner.detectors,
            self.inner.observables,
            self.inner.errors.len()
        )
    }
}

/// Names accepted by `Decoder(dem, name)`.
#[pyfunction]
fn decoder_names() -> Vec<&'static str> {
    DECODER_NAMES.iter().map(|(n, _)| *n).collect()
}

fn parse_params(name: &str, kw: Option<&Bound<'_, PyDict>>) -> PyResult<DecoderParams> {
    let mut p = DecoderParams::default();
    let Some(kw) = kw else { return Ok(p) };
    let allowed = DECODER_NAMES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, a)| *a)
        .unwrap_or(&[]);
    for (k, v) in kw.iter() {
        let k: String = k.extract()?;
        if !allowed.contains(&k.as_str()) {
            return Err(value_err(format!(
                "unknown parameter `{k}` for decoder `{name}`; valid: [{}]",
                allowed.join(", ")
            )));
        }
        match k.as_str() {
            "max_iter" => p.max_iter = Some(v.extract()?),
            "alpha" => p.alpha = Some(v.extract()?),
            "osd_order" => p.osd_order = Some(v.extract()?),
            "legs" => p.legs = Some(v.extract()?),
            "gamma_min" => p.gamma_min = Some(v.extract()?),
            "gamma_max" => p.gamma_max = Some(v.extract()?),
            "seed" => p.seed = Some(v.extract()?),
            // `allowed` (from DECODER_NAMES) only ever lists the keys matched above, so this
            // arm is unreachable in practice — but library code doesn't get to assert that via
            // `unreachable!` (CLAUDE.md), so it's a normal, reportable error instead.
            _ => {
                return Err(value_err(format!(
                    "internal error: unhandled parameter `{k}`"
                )))
            }
        }
    }
    Ok(p)
}

/// Flatten a bool or uint8 numpy array to 0/1 bytes, checking the trailing dimension.
fn to_bytes(
    a: &Bound<'_, PyAny>,
    ndim: usize,
    cols: usize,
    what: &str,
) -> PyResult<(usize, Vec<u8>)> {
    let (shape, data): (Vec<usize>, Vec<u8>) =
        if let Ok(arr) = a.extract::<PyReadonlyArrayDyn<'_, bool>>() {
            (
                arr.shape().to_vec(),
                arr.as_array().iter().map(|&b| b as u8).collect(),
            )
        } else if let Ok(arr) = a.extract::<PyReadonlyArrayDyn<'_, u8>>() {
            (
                arr.shape().to_vec(),
                arr.as_array().iter().copied().collect(),
            )
        } else {
            return Err(value_err(format!(
                "{what}: expected a numpy bool or uint8 array"
            )));
        };
    if shape.len() != ndim || shape[ndim - 1] != cols {
        return Err(value_err(format!(
            "{what}: expected {ndim}-D array with last dimension {cols}, got shape {shape:?}"
        )));
    }
    let rows = if ndim == 1 { 1 } else { shape[0] };
    Ok((rows, data))
}

/// A decoder built once for a DEM and reused across shots.
#[pyclass(name = "Decoder", module = "aleph.qec", frozen)]
pub struct PyDecoder {
    dec: AnyDecoder,
    #[pyo3(get)]
    name: String,
    detectors: usize,
    observables: usize,
    errors: usize,
    /// Whether any mechanism of the source DEM has `^`-decomposed parts (see
    /// `MatchingEdge::column`'s docs in `aleph-qec`). Guards `decode_batch_errors` for the
    /// matching decoders, whose per-column estimate is exact only when every mechanism is a
    /// single part.
    has_components: bool,
}

#[pymethods]
impl PyDecoder {
    #[new]
    #[pyo3(signature = (dem, name, **params))]
    fn new(
        dem: PyRef<'_, PyDem>,
        name: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let p = parse_params(name, params)?;
        let dec = AnyDecoder::build(&dem.inner, name, &p).map_err(value_err)?;
        let has_components = dem.inner.errors.iter().any(|e| !e.components.is_empty());
        Ok(PyDecoder {
            dec,
            name: name.to_string(),
            detectors: dem.inner.detectors,
            observables: dem.inner.observables,
            errors: dem.inner.errors.len(),
            has_components,
        })
    }

    /// Number of error mechanisms (DEM columns) in a `decode_batch_errors` row.
    #[getter]
    fn num_errors(&self) -> usize {
        self.errors
    }

    /// Decode one shot: `[num_detectors]` bool/uint8 -> `[num_observables]` bool.
    fn decode<'py>(
        &self,
        py: Python<'py>,
        dets: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let (_, bits) = to_bytes(dets, 1, self.detectors, "decode")?;
        let out = qec_core::decode_rows(self.dec.get(), &bits, 1, self.detectors, self.observables);
        Ok(PyArray1::from_vec_bound(
            py,
            out.into_iter().map(|b| b != 0).collect(),
        ))
    }

    /// Decode `[shots, num_detectors]` bool/uint8 -> `[shots, num_observables]` bool (GIL released).
    fn decode_batch<'py>(
        &self,
        py: Python<'py>,
        dets: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<bool>>> {
        let (shots, bits) = to_bytes(dets, 2, self.detectors, "decode_batch")?;
        let (d, o) = (self.detectors, self.observables);
        let dec = self.dec.get();
        let out = py.allow_threads(|| qec_core::decode_rows(dec, &bits, shots, d, o));
        let flat = PyArray1::from_vec_bound(py, out.into_iter().map(|b| b != 0).collect());
        flat.reshape([shots, o])
    }

    /// stim b8 rows `[shots, ceil(D/8)]` uint8 -> packed predictions `[shots, ceil(O/8)]` uint8.
    fn decode_batch_bit_packed<'py>(
        &self,
        py: Python<'py>,
        packed: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let ib = qec_core::packed_len(self.detectors);
        let arr = packed
            .extract::<PyReadonlyArrayDyn<'_, u8>>()
            .map_err(|_| value_err("decode_batch_bit_packed: expected a numpy uint8 array"))?;
        let (shots, bytes) = to_bytes(arr.as_any(), 2, ib, "decode_batch_bit_packed")?;
        let (d, o) = (self.detectors, self.observables);
        let dec = self.dec.get();
        let out = py.allow_threads(|| qec_core::decode_packed(dec, &bytes, shots, d, o));
        PyArray1::from_vec_bound(py, out).reshape([shots, qec_core::packed_len(o)])
    }

    /// Decode `[shots, num_detectors]` bool/uint8 -> `(errors uint8 [shots, num_errors],
    /// converged bool [shots])`: the per-column error estimate every decoder can produce (what a
    /// cudaq-qec decoder returns). GIL released; shots decode in parallel.
    ///
    /// Raises `ValueError` for `mwpm`/`union-find`/`union-find-weighted` on a DEM with
    /// `^`-decomposed mechanisms: their per-column estimate is per mechanism, not per part, so
    /// `H ê = s` would not hold (split the parts first, e.g. via `aleph.cudaq.dem_to_matrices`).
    #[allow(clippy::type_complexity)]
    fn decode_batch_errors<'py>(
        &self,
        py: Python<'py>,
        dets: &Bound<'py, PyAny>,
    ) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray1<bool>>)> {
        if self.has_components
            && matches!(
                self.name.as_str(),
                "mwpm" | "union-find" | "union-find-weighted"
            )
        {
            return Err(value_err(
                "decode_batch_errors: this DEM has ^-decomposed mechanisms; a matching \
                 decoder's per-column estimate needs one column per part — rebuild the model \
                 with aleph.cudaq.dem_to_matrices(dem) + aleph.qec.dem_from_matrices(H, O, rates)",
            ));
        }
        let (shots, bits) = to_bytes(dets, 2, self.detectors, "decode_batch_errors")?;
        let (d, e) = (self.detectors, self.errors);
        let dec = &self.dec;
        let (out, conv) =
            py.allow_threads(|| qec_core::decode_errors_rows(dec, &bits, shots, d, e));
        let errors = PyArray1::from_vec_bound(py, out).reshape([shots, e])?;
        Ok((errors, PyArray1::from_vec_bound(py, conv)))
    }

    fn __repr__(&self) -> String {
        format!(
            "Decoder({:?}, detectors={}, observables={})",
            self.name, self.detectors, self.observables
        )
    }
}

/// CSC column lists → DEM (see `aleph.qec.dem_from_matrices` for the user-facing wrapper).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn _dem_from_csc(
    detectors: usize,
    observables: usize,
    h_indptr: PyReadonlyArray1<'_, i64>,
    h_indices: PyReadonlyArray1<'_, i64>,
    o_indptr: PyReadonlyArray1<'_, i64>,
    o_indices: PyReadonlyArray1<'_, i64>,
    probs: PyReadonlyArray1<'_, f64>,
) -> PyResult<PyDem> {
    fn cols(indptr: &[i64], indices: &[i64], what: &str) -> PyResult<Vec<Vec<u32>>> {
        let mut out = Vec::with_capacity(indptr.len().saturating_sub(1));
        for w in indptr.windows(2) {
            let (a, b) = (w[0], w[1]);
            if a < 0 || b < a || b as usize > indices.len() {
                return Err(value_err(format!("{what}: malformed CSC indptr")));
            }
            let col: Result<Vec<u32>, _> = indices[a as usize..b as usize]
                .iter()
                .map(|&i| {
                    u32::try_from(i).map_err(|_| value_err(format!("{what}: negative index {i}")))
                })
                .collect();
            out.push(col?);
        }
        Ok(out)
    }
    let h = cols(h_indptr.as_slice()?, h_indices.as_slice()?, "H")?;
    let o = cols(o_indptr.as_slice()?, o_indices.as_slice()?, "O")?;
    DetectorErrorModel::from_check_matrices(detectors, observables, &h, &o, probs.as_slice()?)
        .map(|inner| PyDem { inner })
        .map_err(|e| value_err(e.to_string()))
}

/// The gross [[144,12,12]] bivariate-bicycle code's circuit-level DEM (`Z` sector of a memory-X
/// experiment, depth-7 syndrome extraction) under uniform circuit noise `p`; the model behind
/// `docs/perf/qec-q5-circuit-dem.md`.
#[pyfunction]
fn gross_code_dem(rounds: usize, p: f64) -> PyResult<PyDem> {
    if rounds == 0 {
        return Err(value_err("rounds must be >= 1"));
    }
    if !p.is_finite() || !(0.0..1.0).contains(&p) {
        return Err(value_err(format!(
            "p must be a finite number in [0, 1), got {p}"
        )));
    }
    aleph_qec::BBCode::gross()
        .circuit_level_dem(rounds, aleph_qec::CircuitNoise::uniform(p))
        .map(|inner| PyDem { inner })
        .map_err(|e| value_err(e.to_string()))
}

/// Register the `qec` submodule on the native module.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "qec")?;
    m.add_class::<PyDem>()?;
    m.add_class::<PyDecoder>()?;
    m.add_function(wrap_pyfunction!(decoder_names, &m)?)?;
    m.add_function(wrap_pyfunction!(_dem_from_csc, &m)?)?;
    m.add_function(wrap_pyfunction!(gross_code_dem, &m)?)?;
    parent.add_submodule(&m)
}
