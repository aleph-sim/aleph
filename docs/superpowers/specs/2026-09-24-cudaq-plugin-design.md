# CUDA-Q QEC decoder plugin (`aleph.cudaq`) — design (Track O, task F6)

**Status:** draft for review. **Date:** 2026-09-24.
**Scope:** expose aleph's QEC decoders (relay-BP family, BP/OSD, Sparse Blossom MWPM, union-find)
as decoders inside NVIDIA CUDA-Q QEC (`cudaq-qec`), so an NVQLink/CUDA-Q integrator can A/B them
against NVIDIA's own decoders on their own hardware with one `pip install` and one `import`.
Python plugin now; the native C++ `.so` plugin is a follow-up issue (§10).

## 1. Intent (what, for whom, success)

**From `docs/qec/open-silicon-program.md` § Track O, F6:** "NVQLink is where every real-time QEC
integrator already is; a plugin lets them A/B our decoder against the GPU path on their own
hardware. Using the competitor's platform as our distribution channel is the cheapest reach
available."

**Who:** a QEC integrator already using `cudaq_qec` (Python) who wants to try aleph's relay-BP /
MWPM without leaving their harness.

**Success:**

1. `pip install aleph-sim[cudaq]` then `import aleph.cudaq` registers seven decoders in
   `cudaq_qec`; `cudaq_qec.get_decoder("aleph-relay-bp-osd", H, O=O, error_rate_vec=r)` works
   and passes cudaq's own canonical logical-error-rate loop unchanged (result is an error
   estimate per column of `H`, `O @ result % 2` gives the observable flips).
2. An A/B script (`scripts/python/ab_cudaq.py`) produces an honest table of logical error rate
   (with confidence intervals) and shots/s for **gross qLDPC** (aleph relay-BP+OSD vs
   `nv-qldpc-decoder` in relay mode) and **surface code** (aleph MWPM vs cudaq `pymatching` vs
   `nv-fusion-decoder`), recorded in `docs/perf/f6-cudaq-plugin.md`.
3. Documented in `crates/aleph-py/README.md` with a ten-line example in the style of cudaq's
   `circuit_level_noise.py`.

**Decisions taken with the user (2026-09-24):** Python plugin now, C++ later (issue);
packaged as a submodule of the existing `aleph-sim` wheel, not a second PyPI package;
A/B covers both gross qLDPC and surface code with LER and throughput.

## 2. Verified facts about `cudaq-qec` 0.8.0 (the target)

Measured on the GPU box (`openwebgui.splynx.com`, CUDA 13.0, driver 580, RTX 4000 SFF Ada,
Ubuntu 24.04, Python 3.12) with `cudaq-qec-cu13 0.8.0` + `cuda-quantum-cu13 0.16.0`, and read
from the `NVIDIA/cudaqx` sources at tag `0.8.0`.

- **Registration is by import.** `@cudaq_qec.decoder("name")` on a class re-creates it with base
  `cudaq_qec.Decoder` and registers `name`. Built-in Python plugins live in the namespace package
  `cudaq_qec.plugins.decoders`; third-party ones simply need to be imported. No env var, no
  entry point, nothing to copy into the cudaq tree.
- **Constructor contract.** `get_decoder(name, H, **kw)` calls `cls(H, **kw)`. `H` is whatever
  the caller passed: a dense `np.uint8` `[D × E]` array or any `scipy.sparse` matrix. If the caller
  passed a Stim DEM *string*, cudaq parses it first and calls the plugin with the dense `H` plus
  two injected kwargs: `O` (`np.uint8 [num_obs × E]`) and `error_rate_vec` (`float64 [E]`).
  cudaq's DEM parser keeps `^`-decomposed hyperedges as one column (a column may have three or
  more ones), so a matching decoder cannot be fed a DEM string through cudaq. The plugin must call
  `cudaq_qec.Decoder.__init__(self, H)`; that sets `get_block_size()` = E and
  `get_syndrome_size()` = D.
- **Decode contract.** `decode(syndrome: list[float])` — one float per detector (a probability;
  in practice hard 0.0/1.0) — returns `DecoderResult` with `converged: bool`, `result: list[float]`
  of length E (an error estimate per column; `> 0.5` means "flipped"), `opt_results: dict | None`.
  `decode_batch(list[list[float]])` is optional; if overridden it **must** return
  `cudaq_qec.BatchDecoderResult(result=float64 2-D `[shots × E]`, converged=bool 1-D, opt_results=None,
  batch_opt_results=None)`; returning a list is a `TypeError` (0.7+). The default `decode_batch`
  loops `decode` per shot in Python.
- **cudaq's canonical LER loop** (`docs/sphinx/examples/qec/python/circuit_level_noise.py`):
  `err = decoder.decode_batch(syndromes).result > 0.5`, `pred = (O @ err.T) % 2`, compare with the
  observed logical flips. Our results therefore *must* be per-column error estimates, not
  observable flips.
- **NVIDIA's decoders (0.8.0):** `nv-qldpc-decoder` (GPU BP / relay-BP / BP+OSD; params include
  `max_iterations`, `use_osd`, `osd_method`, `osd_order`, `bp_method` (3 = min-sum + dynamic
  memory, needed for relay), `composition` (1 = sequential relay), `srelay_config{pre_iter,
  num_sets, stopping_criterion, stop_nconv}`, `gamma_dist`, `error_rate_vec`, `n_threads`,
  `proc_float`), `nv-fusion-decoder` (multithreaded MWPM fusion + sparse blossom, new in 0.8),
  `pymatching`, `chromobius`, `single_error_lut`, `multi_error_lut`, `sliding_window`,
  `tensor_network_decoder`. The `nv-qldpc-decoder` and `nv-fusion-decoder` shared objects are
  closed source under NVIDIA's SLA; everything a plugin links or imports is Apache-2.0.
- **cudaq's built-in codes** are `surface_code`, `repetition`, `steane`. There is no
  bivariate-bicycle code, so the gross-code DEM must come from aleph.
- **C++ plugins** are `.so` files dlopen'ed from `<dir of libcudaq-qec-decoders.so>/decoder-plugins/`;
  the wheel ships no headers, and `main` has already changed the plugin ABI relative to 0.8.0.
  This is why the native plugin is deferred (§10).
- `aleph-sim 0.3.0` (manylinux x86_64) installs and imports in the same venv as `cudaq-qec-cu13`.

## 3. Architecture

```
user code                                 aleph-sim wheel
─────────                                 ──────────────────────────────────────────────
import cudaq_qec as qec                   python/aleph/cudaq.py          (new, pure Python)
import aleph.cudaq          ──registers──▶  @qec.decoder("aleph-<name>") × 7
dec = qec.get_decoder(                      class _AlephDecoder(H, **kw):
  "aleph-relay-bp-osd",                       dem = aleph.qec.DetectorErrorModel.from_matrices(H, O, rates)
   H, O=O, error_rate_vec=r,                  inner = aleph.qec.Decoder(dem, "<name>", **params)
   osd_order=12)                            decode(syn)        → threshold → inner.decode_batch_errors → DecoderResult
dec.decode_batch(syndromes)                 decode_batch(syns) → one Rust batch, GIL released → BatchDecoderResult
                                          _native (pyo3, crates/aleph-py/src/qec.rs)
                                            DetectorErrorModel.from_matrices(H, O, error_rate_vec)   (new)
                                            Decoder.decode_batch_errors(dets) -> (uint8 [S×E], bool [S]) (new)
                                            Decoder.num_errors                                            (new)
                                            gross_code_dem(rounds, p) -> DetectorErrorModel               (new)
                                          aleph-qec (Rust)
                                            DetectorErrorModel::from_check_matrices(...)                   (new)
                                            AnyDecoder::decode_errors(&self, &Syndrome) -> (Vec<u8>, bool) (new, aleph-py)
                                            MwpmDecoder::decode_errors → Sparse Blossom matched pairs + path retrace (new)
                                            UnionFindDecoder::decode_edges (existing) → edge → column
                                            Bp/Osd/Relay: existing ehat outputs
```

Layering rule (CLAUDE.md "backend-agnostic IR" analogue): `aleph-qec` knows nothing about cudaq.
`aleph.cudaq` is the only file that imports `cudaq_qec`, and it is imported lazily — `import aleph`
and `import aleph.qec` never touch cudaq.

## 4. Rust changes (`crates/aleph-qec`)

### 4.1 `DetectorErrorModel::from_check_matrices`

```rust
impl DetectorErrorModel {
    /// Build a DEM from column lists: `h_cols[j]` = detectors of mechanism `j`,
    /// `o_cols[j]` = observables it flips, `probs[j]` = its probability.
    pub fn from_check_matrices(
        detectors: usize, observables: usize,
        h_cols: &[Vec<u32>], o_cols: &[Vec<u32>], probs: &[f64],
    ) -> Result<Self>;
}
```

Errors (new `Error` variants, thiserror): length mismatch between the three slices; a detector or
observable index out of range; `observables > 64` (closes the Rust half of #512 — the u64 masks in
every decoder); a non-finite probability (ADR 0006: explicit `is_finite()` reject before any
comparison). `p ≤ 0` columns are kept (they are legal DEM lines; decoders already skip them).
Each column becomes one `DemError::new(prob, dets, obs)` with no `^` components.

### 4.2 Per-column error output — `decode_errors`

The `Decoder` trait is unchanged (it returns observable flips). Each concrete decoder gains a
method returning the error estimate over DEM columns plus a convergence flag:

| decoder | source of `ehat` | `converged` |
|---|---|---|
| `BpDecoder` | `decode_bp_soft().ehat` (exists) | BP converged |
| `OsdDecoder` | `decode_osd_ehat()` (exists) | `true` when OSD ran (it satisfies `H ê = s` by construction), else BP converged |
| `RelayBpDecoder` | `decode_soft().ehat` (exists) | a valid solution was found |
| `RelayBpOsdDecoder` | relay ehat, or OSD ehat if relay failed | as OSD |
| `UnionFindDecoder` | `decode_edges()` (exists) → edge → column | always `true` |
| `MwpmDecoder` | **new** `decode_errors` (§4.3) | always `true` |

`crates/aleph-py/src/qec_core.rs::AnyDecoder` gets `decode_errors(&self, &Syndrome) -> (Vec<u8>, bool)`
dispatching to those; the batch driver mirrors `decode_batch` (rayon `par_chunks`, `Sync`).

**Edge → column map.** `MatchingGraph::from_dem` merges parallel edges keyed by
`(a, b, observables)` and drops undetectable columns. It now also records, per edge, the
**representative column**: the index of the first DEM error (and, for `^`-decomposed mechanisms,
the first *part*) that produced that key. `MatchingEdge` gains `pub column: u32`. Because the key
includes the observable set, the representative column's `O` row equals the edge's observables,
so `O · ê` over representative columns reproduces the edge's observable flips exactly. For
`H + O + rates` input (no `^`), every column has ≤ 2 detectors, so parts are trivially single.

### 4.3 Sparse Blossom: matched pairs + path retrace

Today `sparse_blossom::matcher::resolve()` folds the matching into `(observable mask, weight)`;
the growth state (`NodeState { source, dist, obs, … }`) records *which defect reached a node* and
the observable parity from it, but not the edge path, and blossoms are resolved through collision
edges (`CEdge { from, to, obs, weight }`) that likewise carry no edge identity. Recording
predecessor edges through blossom formation and shattering would touch the hot path for every
shot to serve a rarely-used output.

Instead, the plugin path does what PyMatching's `decode_to_edges` does: **resolve to matched
defect pairs, then retrace each pair's shortest path.**

- `resolve()` gains a variant `resolve_pairs(&self, out: &mut Vec<(NodeId, NodeId)>)` that emits
  one `(u, v)` per top-level and blossom-internal leaf match (`descend` already recurses to the
  leaf regions, whose `source` is the defect) and `(u, BOUNDARY)` for boundary matches. Weight and
  observable mask are still returned by `resolve()`; `decode` is untouched.
- `MwpmDecoder::decode_errors(&self, syndrome) -> Vec<u8>` runs a **bounded Dijkstra** per pair
  on the compiled CSR graph (doubled integer weights, so the path length is exact and equal to the
  match edge's weight; the search is cut off at that distance). Deterministic tie-breaking: lowest
  `(distance, node index)` first. The path's edges are XORed into `ehat` via the edge → column
  map. For boundary matches the target is any boundary edge; the search stops at the first node
  with a boundary edge whose distance plus that edge equals the match weight.
- Cost is `O(Σ pairs × local search)`, comparable to the matching itself; it is paid only by
  `decode_errors`, never by `decode`. Per-thread scratch reuses the existing thread-local state
  cache.

**Invariants (tests, §7):** for every shot, `H · ê = s` (mod 2), `Σ w(ê) = matching weight`
returned by `decode`, and `O · ê = obs mask` returned by `decode` **except on genuine ties**
(equal-weight paths with different observables; the same tie phenomenon the Sparse Blossom PR
calibrated at ~21 % of shots at d = 11, p = 0.06). The test bounds the disagreement rate by the
existing `decode_local` tie sentinel. The plugin derives *nothing* from `decode`'s mask: cudaq
sees `ehat` only, so it is self-consistent.

### 4.4 Gross-code DEM export

`BBCode::gross().circuit_level_dem(rounds, CircuitNoise::uniform(p))` exists in Rust and is the DEM
behind `docs/perf/qec-q5-circuit-dem.md`. It is exposed to Python unchanged (§5.3) so the A/B
harness feeds byte-identical models to both sides.

## 5. pyo3 changes (`crates/aleph-py/src/qec.rs`)

### 5.1 `DetectorErrorModel.from_matrices(H, O=None, error_rate_vec=…)` (staticmethod)

Accepts `H` and `O` as dense `np.uint8` arrays or any `scipy.sparse` matrix (converted on the
Python side to CSC column lists before crossing into Rust: `indptr/indices` of `H.tocsc()`; no
dense allocation for sparse input). `O=None` ⇒ zero observables. Raises `ValueError` mapped from
the §4.1 errors.

### 5.2 `Decoder.decode_batch_errors(dets) -> (errors, converged)`

`dets`: bool/uint8 `[shots × D]` (same coercion as `decode_batch`). Returns
`(np.uint8 [shots × E], np.bool_ [shots])`. Releases the GIL and runs the rayon batch driver like
`decode_batch`. Plus a read-only `Decoder.num_errors` property (E) and
`DetectorErrorModel.num_errors` (exists).

### 5.3 `aleph.qec.gross_code_dem(rounds: int, p: float) -> DetectorErrorModel`

Uniform circuit noise `p` (CNOT depolarizing, init, measurement, idle all at `p`, i.e.
`CircuitNoise::uniform(p)`), `rounds` measurement rounds. Documented as "the model from
`docs/perf/qec-q5-circuit-dem.md`".

## 6. The plugin module `crates/aleph-py/python/aleph/cudaq.py`

Pure Python, ~150 lines, no new runtime dependency on the base wheel. `pyproject.toml` gains the
extra `cudaq = ["cudaq-qec>=0.8,<0.9", "scipy>=1.10"]` (the `cudaq-qec` meta-package selects
the cu12/cu13 wheel at install time; `scipy` is what cudaq hands us for sparse `H`).

```python
import cudaq_qec as _qec          # ImportError here is the only way this module can fail to import
import numpy as np
import aleph.qec as _aq

NAMES = ("mwpm", "union-find", "union-find-weighted", "bp", "bp-osd", "relay-bp", "relay-bp-osd")

def _make(name):
    @_qec.decoder(f"aleph-{name}")
    class AlephDecoder:
        def __init__(self, H, O=None, error_rate_vec=None, error_rate=None, **params):
            _qec.Decoder.__init__(self, H)
            rates = _rates(H.shape[1], error_rate_vec, error_rate)   # ValueError if neither given
            self._dem = _aq.DetectorErrorModel.from_matrices(H, O, rates)
            self._inner = _aq.Decoder(self._dem, name, **params)      # unknown kwargs → its ValueError
        def decode(self, syndrome):
            err, conv = self._inner.decode_batch_errors(_to_bits([syndrome]))
            r = _qec.DecoderResult(); r.converged = bool(conv[0]); r.result = err[0].astype(np.float64).tolist(); r.opt_results = None
            return r
        def decode_batch(self, syndromes):
            err, conv = self._inner.decode_batch_errors(_to_bits(syndromes))
            return _qec.BatchDecoderResult(result=err.astype(np.float64), converged=conv, opt_results=None, batch_opt_results=None)
    return AlephDecoder

DECODERS = {name: _make(name) for name in NAMES}
```

Rules:

- `_to_bits` thresholds the float syndromes at `> 0.5` (cudaq's own convention for its results)
  into a C-contiguous `uint8 [shots × D]`; a wrong width raises `ValueError` naming D.
- Parameter names and defaults are exactly those of `aleph.qec.Decoder` (`legs`, `alpha`,
  `gamma_min`, `gamma_max`, `seed`, `osd_order`, `max_iter`), so the two entry points cannot
  drift; the plugin adds only `error_rate` (scalar fallback for `error_rate_vec`).
- Matching decoders (`mwpm`, `union-find*`) raise `ValueError("column j has k detectors; matching
  decoders need a graphlike H — pass the decomposed DEM as H, not as a DEM string")` when a column
  has > 2 ones (mapped from `Error::NonGraphlike`).
- `converged` semantics follow the §4.2 table; `opt_results` is `None` (nothing extra to report
  in v1).
- The module exposes `aleph.cudaq.NAMES` and `aleph.cudaq.dem_to_matrices(stim_dem) -> (H, O, rates)`:
  a helper that turns a Stim DEM (or `aleph.qec.DetectorErrorModel`) into graphlike matrices by
  splitting `^` parts into separate columns — the way to feed a matching decoder through cudaq
  (§2, third bullet). It is also what the harness uses for the surface-code rows.

## 7. Testing

**Rust (`crates/aleph-qec`)**

- `from_check_matrices`: unit tests for every error variant; round-trip
  `parse(to_dem_string(from_check_matrices(...)))` on a d = 3 surface DEM.
- `decode_errors` invariants (§4.3) as `proptest` on the existing random-graph generator
  (≤ 12 nodes) for MWPM and UF, plus a release-mode differential on the phenomenological
  d ∈ {3, 5, 7, 9, 11} × p ∈ {0.01, 0.03, 0.06} shots already used by the Sparse Blossom PR:
  `H ê = s` and `Σ w = weight` on every shot; `O ê` disagreement rate ≤ the `decode_local` tie
  sentinel. Same invariants for BP/OSD/relay on the gross code-capacity DEM at p = 0.01
  (converged shots only for `H ê = s`).
- `MatchingEdge::column`: each edge's `O` row equals its observables (unit test on a hand DEM with
  parallel and `^` mechanisms).

**Python (`scripts/python/test_cudaq.py`, unittest, discovered by CI like `test_qec.py`)**

- `skipUnless(HAVE_CUDAQ)` at module level — CI's self-hosted runner has no CUDA and the cudaq
  wheel is not installed there, so in CI this file skips; it is run for real on the GPU box
  before the PR is opened (the PR body quotes the output).
- Registration: all seven names resolve through `cudaq_qec.get_decoder`.
- Contract: on a stim surface d = 3 DEM (decomposed, via `dem_to_matrices`), `decode` returns
  `len(result) == get_block_size()`; `decode_batch` returns a `BatchDecoderResult` of shape
  `[shots × E]`; `(O @ result.T) % 2` equals `aleph.qec.Decoder(...).decode_batch(...)` for the
  BP family on converged shots and for MWPM up to ties (weight check via the Rust side is not
  reachable from Python, so the Python test only asserts `H ê = s`).
- DEM-string path: `get_decoder("aleph-relay-bp", str(dem))` works (cudaq injects `O` and
  `error_rate_vec`); `get_decoder("aleph-mwpm", str(dem))` raises the graphlike `ValueError`.
- Missing rates: `get_decoder("aleph-bp", H)` raises `ValueError`.
- `decode_batch_errors`/`from_matrices` also get direct tests in `test_qec.py` (they do not need
  cudaq).

**Not tested in CI:** anything requiring a GPU. The harness is a script, not a test.

## 8. A/B harness `scripts/python/ab_cudaq.py` and the perf record

Runs on the GPU box only (`/root/cqvenv`: cudaq-qec-cu13 0.8.0, stim 1.16, aleph-sim from the PR
wheel via `maturin develop --release`). Box must be idle (CLAUDE.md § Performance: check
`uptime`, `pgrep`). Output: a Markdown table to stdout, pasted into `docs/perf/f6-cudaq-plugin.md`
with the exact command lines, package versions and box description.

**Workload G (gross qLDPC).** `aleph.qec.gross_code_dem(rounds=12, p)` for
p ∈ {0.001, 0.002, 0.003}; syndromes and observable flips sampled with stim from the DEM text
(`stim.DetectorErrorModel(dem.to_dem_string()).compile_sampler(seed=…)`), the same arrays for
every decoder; ≥ 10 000 shots per point, more (up to 10⁵) where the LER is < 10⁻³ so that every
cell has ≥ 20 logical errors or an explicit upper bound.

| decoder | configuration |
|---|---|
| `aleph-relay-bp-osd` | defaults (4 legs, α = 0.875, γ ∈ [−0.3, 0.9]) + `osd_order=12`, the `docs/perf/qec-q5-circuit-dem.md` settings |
| `aleph-relay-bp` | defaults, no OSD (the ASIC-class configuration, flag-only policy) |
| `nv-qldpc-decoder` relay | `bp_method=3, composition=1, use_sparsity=True, srelay_config` at NVIDIA's documented defaults, `use_osd=True, osd_order=12, osd_method=1`, `error_rate_vec` |
| `nv-qldpc-decoder` relay, no OSD | same without OSD |

**Workload S (surface code).** stim `surface_code:rotated_memory_x`, d ∈ {5, 9}, rounds = d,
uniform p = 0.003 (the `bench_qec.py` cases), `decompose_errors=True`, graphlike `H, O, rates`
via `dem_to_matrices`; 20 000 / 5 000 shots.

| decoder | configuration |
|---|---|
| `aleph-mwpm` | — |
| `aleph-union-find-weighted` | — |
| cudaq `pymatching` | `error_rate_vec` |
| `nv-fusion-decoder` | `error_rate_vec` plus its defaults (`num_threads`, `block_leaf_size`, `fusion_strategy` — the whole 0.8.0 schema); the values actually used are recorded verbatim, `num_threads` = 20 to match aleph's all-core row |

**Metrics per cell:** logical error rate = shots where `O ê ≠ observed flips` (any observable) /
shots, with a 95 % Wilson interval; throughput = shots/s of `decode_batch` on the full batch
after a 200-shot warm-up, best of 3. aleph rows are measured twice: all 20 cores, and
`RAYON_NUM_THREADS=1` in a child process (as `bench_qec.py` does). NVIDIA rows run on the GPU.

**Honesty rules for the write-up:** relay-BP parameterisations are *not* equivalent between the
two implementations (different γ-schedules, stopping rules, message precision); the table says
so, lists both parameter sets verbatim, and does not tune either side against the other beyond
the documented defaults. Any cell where a decoder errors out or fails to converge on > 1 % of
shots is reported as such rather than dropped. Throughput compares a CPU decoder with a GPU one;
the write-up states the hardware on both sides and does not claim a "winner" on throughput,
only on LER at equal inputs.

## 9. Documentation

- `crates/aleph-py/README.md`: new section "Using aleph decoders from CUDA-Q QEC" — install line,
  a ten-line example mirroring cudaq's `circuit_level_noise.py` with `aleph-relay-bp-osd`
  swapped in, the seven names, the graphlike-H rule for matching decoders, and a pointer to the
  perf record.
- `CHANGELOG.md` entry under Unreleased.
- `docs/perf/f6-cudaq-plugin.md`: the §8 table and method.
- `docs/qec/open-silicon-program.md` § Track O: tick F6 with the date and a one-line result.

## 10. Out of scope → follow-up issues

1. **Native C++ plugin** (`libaleph-cudaq.so` in `decoder-plugins/`, Rust C ABI + thin C++ shim)
   for the realtime / NVQLink path. Blocked on cudaqx's plugin ABI settling (0.8.0 → `main`
   already differs); the Rust surface this PR adds (`from_check_matrices`, `decode_errors`) is the
   ABI the shim will call.
2. **`iters_per_leg` / early-exit knob for the f64 `RelayBpDecoder`** (today only
   `FixedRelayBp::with_budget` has it), so the A/B can match the ASIC's 6 × 10 schedule exactly.
3. **Python guard for > 64 observables** in `aleph-py` becomes redundant once §4.1 lands; #512
   is closed by this PR's Rust guard and the remaining item (docs note) is done here too.

## 11. Risks and mitigations

- **cudaq-qec API churn** (0.7 changed `decode_batch`'s return type; `main` changes the C++ ABI).
  Mitigation: pin `cudaq-qec>=0.8,<0.9` in the extra; the Python contract is small and tested
  by `test_cudaq.py`, which is re-run on the box at each cudaq bump.
- **Tie disagreement between `decode` and `decode_errors` for MWPM** is inherent (equal-weight
  paths). Mitigation: the plugin uses `ehat` only; the tests bound the rate; the README says
  results agree with `aleph.qec.Decoder` up to ties.
- **`from_matrices` on huge dense `H`** (cudaq hands a dense `uint8` for DEM-string input;
  gross r = 12 is 936 × 8784 ≈ 8 MB — fine). Sparse input never densifies.
- **Fairness of the A/B**: addressed by the §8 honesty rules; the deliverable is the plugin, the
  table is evidence, not a claim of superiority.

## 12. Acceptance criteria (for the PR)

1. `pip install aleph-sim[cudaq]` + `import aleph.cudaq` registers the seven decoders;
   `scripts/python/test_cudaq.py` passes on the GPU box (output in the PR body) and skips in CI.
2. `DetectorErrorModel.from_matrices`, `Decoder.decode_batch_errors`, `gross_code_dem` are
   documented and unit-tested; the §4.3 invariants hold on the differential shot set.
3. `docs/perf/f6-cudaq-plugin.md` contains the G and S tables with the §8 metrics, package
   versions and box description; the README example runs as written.
4. Two follow-up issues (§10 items 1–2) are filed and linked from the perf record.
