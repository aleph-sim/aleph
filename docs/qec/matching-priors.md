# Matching prior policy (#515)

MWPM, Union-Find and weighted Union-Find require each source mechanism's prior to
be finite and in **[0, 0.5]**. Rust, native Python and the CUDA-Q adapter reject
unsupported priors at construction. Inputs are never clamped: doing so silently
changes the noise model. DEM parsing and matrix construction retain their [0, 1]
domain, as do the four BP-family decoder constructors.

| Prior | Matching construction |
| --- | --- |
| 0 or negative zero | Accepted; no edge, original column numbering retained |
| Positive, including the smallest positive `f64` | Accepted |
| 0.5 | Accepted; zero-weight edge |
| Greater than 0.5, including 1 | Rejected |
| Negative, NaN or infinity | Rejected |

Validation runs before dropping empty/purely logical columns or merging parallel
edges. For example, two 0.6 source priors are rejected even though their XOR
probability would be 0.48. It also applies to `^` components. Split constructors
report columns of the **split** input model; ordinary constructors report original
DEM columns. Indices are zero-based.

Rust returns `Error::InvalidMatchingPrior { column, prob }`, retaining the rejected
value. Python raises `ValueError`; for example:

```python
from aleph import qec

dem = qec.DetectorErrorModel("error(0.1) D0\nerror(0.6) D1 L0")
qec.Decoder(dem, "mwpm")
# ValueError: matching prior at column 1 must be finite and in [0, 0.5], got 0.6
```

`aleph.cudaq` delegates matching range validation to this same native constructor.
Its existing finite-input and matrix validation remain in place. Tests compare
the adapter's range-error message directly with the native Python message.

## Why reject, and what changed for tiny priors?

Previously Rust accepted priors above 0.5, creating negative log-odds weights.
Sparse Blossom's retrace cannot use these edges and its flooder assumes
nonnegative weights. Rust also silently discarded probability 1 and non-finite
mechanisms while the CUDA-Q adapter rejected high priors. Explicit rejection
avoids a partially decoded model and makes the supported domain consistent.
Supporting high matching priors through a transformed model would require a
separate design; clamping is not a substitute.

Boundary testing also reproduced an infinite weight for the smallest positive
`f64`: `(1-p)/p` overflows before taking its logarithm. Such a weight can overflow
the sparse graph's integer scaling. When the ratio overflows, weight calculation
now uses `ln(1-p) - ln(p)`, which stays finite. Ordinary finite ratios retain the
original expression; regression tests check bit-identical weights at representative
priors. Decoder algorithms, integer scale and tie-breaking are unchanged.

This does not resolve infeasible syndromes or genuine MWPM retrace ties described
in [component splitting](split-components.md). No throughput improvement is claimed.

## Validation (2026-10-16)

Baseline: `5a671ae26b2eb4235d024403952c2f27988bddfb`; validated code:
`ef9c93b723c9942c7434b0aa9001f06fcfadfe9c`. Manual self-review found no
blocking issues. These are targeted correctness checks, not a full workspace test
run; the PR's CI provides the full platform matrix.

| Environment | Rust tests | Python tests | Additional checks |
| --- | ---: | ---: | --- |
| Local macOS, Python 3.12 | 65 | 15 | Workspace/all-target Clippy, Python-feature Clippy, rustfmt |
| openwebgui Linux, Python 3.12.3, real CUDA-Q QEC SDK | 65 | 33 | QEC/Python all-target Clippy; no Python skips |

Each Rust total includes seven prior regressions in **both debug and release**,
21 DEM tests, 19 matching tests, seven split-component tests and four doctests.
Cases cover invalid ranges/NaN/infinities, all matching constructors, unchanged BP
construction, zero and half priors, subnormal weights, split indices and exact
syndrome/observable projection on independently constrained columns.

The Linux Python run includes 18 actual CUDA-Q registration/adapter tests, using
an isolated source/native-module overlay over the existing `cqvenv`; the installed
environment and shared services were left intact. Ten source hashes were verified
against the local patch. An additional Stim audit decoded 2,000 shots for each of
three matching decoders at distances 3 and 5 (12,000 cases): all native/reference,
syndrome and observable comparisons agree. This audit uses the existing matrix
roundtrip reference, not an independent optimal-matching oracle.

Evidence: [validation metadata](../perf/data/q1-515-validation.json),
[Stim audit](../perf/data/q1-515-projection-audit.csv),
[source hashes](../perf/data/q1-515-source-sha256.txt), and
[targeted validation output](../perf/data/q1-515-validation.txt).

Reproduce after building the Python extension with `maturin develop --release
--features python` from `crates/aleph-py`:

```bash
cargo test -p aleph-qec --test matching_priors
cargo test --release -p aleph-qec --test matching_priors
cargo test -p aleph-qec --lib dem::tests
cargo test -p aleph-qec --lib matching::tests
cargo test -p aleph-qec --test split_components
cargo test -p aleph-qec --doc
PYTHONPATH=scripts/python python -m unittest test_qec.TestApi \
  test_qec.TestStimDems.test_native_split_matches_matrix_roundtrip_on_real_stim_shots \
  test_qec.TestStimDems.test_matching_decode_batch_errors_rejects_decomposed_dem \
  test_cudaq.TestRegistration
python scripts/python/audit_qec_split_components.py
```

The CUDA-Q class requires its SDK and SciPy; without them unittest may skip it.
The recorded Linux run explicitly required the SDK and rejected any skipped test.
