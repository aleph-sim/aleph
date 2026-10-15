# Native per-component DEM columns (#518)

Matching decoders build one edge for each `^` part, while an unsplit DEM's error
columns refer to whole mechanisms. Several edges can therefore point at the same
column even when that column's detector/observable support differs from each edge.
A per-mechanism `decode_errors` vector cannot represent those independent edge choices.

`DetectorErrorModel::split_components()` returns an owned decoder model with one
column per part. Mechanisms retain declaration order; parts retain their order
within each mechanism. Each part keeps its parent's probability, with duplicate
detector/observable targets cancelled independently within that part. Empty and
purely logical parts remain columns. Declared detector/observable counts remain
unchanged; undecomposed mechanisms remain unchanged. The source is untouched and
the operation is idempotent.

For `error(0.1) D0 L0 ^ D1 L1`, one original column becomes two columns:

| Split column | Detectors | Observables | Prior |
|---:|---|---|---:|
| 0 | D0 | L0 | 0.1 |
| 1 | D1 | L1 | 0.1 |

A correction to syndrome D0 can now mark `[1, 0]`, which projects to D0/L0, instead
of marking the single original column and incorrectly projecting to D0+D1/L0+L1.

## Rust

```rust
use aleph_qec::{DetectorErrorModel, MwpmDecoder, Syndrome, UnionFindDecoder};
let dem = DetectorErrorModel::parse("error(0.1) D0 L0 ^ D1 L1")?;
let split = dem.split_components();
let decoder = MwpmDecoder::new_with_split_components(&dem)?;
assert_eq!(decoder.decode_errors(&Syndrome::new(2, vec![0])), vec![1, 0]);
// Error entries index split.errors, not dem.errors.
let uf = UnionFindDecoder::new_with_split_components(&dem)?.weighted(true);
// Equivalently, construct any matching decoder against &split directly.
```

The ordinary constructors retain their original mechanism-column contract.
An individual hyperedge component still raises `NonGraphlike`; splitting does not
turn arbitrary hypergraphs into matching graphs.

## Python

```python
import aleph.qec as qec

dem = qec.DetectorErrorModel("error(0.1) D0 L0 ^ D1 L1")
split = dem.split_components()
decoder = qec.Decoder(split, "mwpm")  # also union-find / union-find-weighted
# ehat, converged = decoder.decode_batch_errors(detection_events)
# ehat.shape[1] == split.num_errors
```

The guard on unsplit matching `decode_batch_errors` remains and now points at this
native API. `dem_to_matrices` / `dem_from_matrices` remain available; a dense matrix
round trip is no longer required to construct a decoder with per-part columns.
BP-family decoders continue to use parity-reduced original mechanism columns.

## Noise semantics and remaining matching limits

Keep the **original** DEM for sampling experiments and use the split model only for
decoder construction/projection. Stim's [`^` separator](https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md#separator-target)
suggests a decomposition of one mechanism; its parts fire together. Sampling the
split columns as independent mechanisms loses that correlation and is not an
equivalent noise model. The Rust regression tests include a deterministic example:
D0 alone is possible in the split model but impossible under `error(p) D0 ^ D1`.

Per-part column identity fixes this representation problem. Existing matching
limitations still apply: infeasible syndromes, negative edge weights (#515), and
MWPM equal-weight path retrace ties. In particular, this API does not promise that
MWPM `O·ehat` always equals its separate `decode()` observable prediction on genuine
ties. Union-Find uses the same peeled edges for both outputs. No hot decoder or
sampling algorithm, precision policy or tie-breaking is changed here; no speedup
is claimed.

## Validation

- 50 focused Rust tests/examples **per platform** (Apple Silicon and Linux x86-64): 21 existing DEM tests, 19 matching-graph tests,
  seven new split/component/constructor tests (including a randomized parity
  property) and three rustdoc examples.
- 13 Python API/Stim tests **per platform**: native-vs-matrix projection checks, source immutability,
  empty/no-component models, repeated targets, multiple observables, exact column
  counts, empty batches and retained actionable guard errors. No oracle tests in
  this targeted run were skipped.
- Real Stim surface-code shots: distances 3/5, seed 518, physical error probability
  0.003, 2,000 shots per distance and three decoders (12,000 decoded shot cases
  per platform; 24,000 total).
  Native and matrix-round-trip error estimates agree exactly. Detector projections,
  observable projections and original-vs-split ordinary decoder predictions have
  zero disagreements in this campaign. The observed zero does not remove the
  general MWPM tie limitation above.
- Workspace all-target clippy, Python-feature all-target clippy, rustfmt and diff
  checks pass. Review was manual. The entire workspace runtime suite is left to CI.

The [Apple Silicon audit CSV](../perf/data/q1-518-projection-audit.csv),
[Linux audit CSV](../perf/data/q1-518-projection-audit-linux.csv) and
[validation record](../perf/data/q1-518-validation.json) accompany this change.
The campaign records correctness counts rather than CPU timings. Apple Silicon
and Linux validation environments are listed in the validation record.

```sh
cargo test -p aleph-qec --lib dem::tests
cargo test -p aleph-qec --lib matching::tests
cargo test -p aleph-qec --test split_components
cargo test -p aleph-qec --doc
cargo clippy --workspace --all-targets -- -D warnings
# Set PYO3_PYTHON to the test interpreter when needed:
cargo clippy -p aleph-py --features python --all-targets -- -D warnings
maturin develop --manifest-path crates/aleph-py/Cargo.toml --release --features python
PYTHONPATH=scripts/python python -m unittest -v test_qec.TestApi \
  test_qec.TestStimDems.test_native_split_matches_matrix_roundtrip_on_real_stim_shots \
  test_qec.TestStimDems.test_matching_decode_batch_errors_rejects_decomposed_dem
python scripts/python/audit_qec_split_components.py
```
