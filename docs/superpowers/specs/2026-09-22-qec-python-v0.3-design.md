# QEC decoders in Python + v0.3.0 release — design

**Date:** 2026-09-22 · **Branch:** `qec-python-v0.3` · **Status:** approved in brainstorming, awaiting spec review

## Why

`aleph-qec` holds MWPM, Union-Find, BP, BP-OSD, relay-BP and relay-BP-OSD decoders, but none of them
is reachable from Python — `crates/aleph-py` exposes only circuits, energy, noise and `run`. The QEC
community works in Python on top of stim + sinter, so today the decoders exist only as numbers in
`docs/perf/`. Separately, the last PyPI release (v0.2.0, 2026-06-12) predates 244 commits on `main`
(Phases 5.9–5.11 GPU work, FP32, host paging, noise v1).

Goal: `pip install aleph-sim[sinter]` → an aleph decoder in a `sinter.collect` run in one line, and a
v0.3.0 release that ships it together with everything accumulated since v0.2.0.

## Decisions taken in brainstorming

- **All DEM-constructed decoders** are exposed: `mwpm`, `union-find`, `union-find-weighted`, `bp`,
  `bp-osd`, `relay-bp`, `relay-bp-osd`.
- **Python floor lowered** from `abi3-py312` to **`abi3-py310`**.
- **Packaging approach A:** mixed maturin project — Rust extension renamed to `aleph._native`, a thin
  pure-Python package `python/aleph/` on top; the sinter adapter is Python.

## Out of scope

Windowed decoders (`SlidingWindowDecoder`, `ParallelWindowDecoder`, `SlidingWindowBp`), GPU decoders,
fixed-point `FixedRelayBp*`, Sparse Blossom (#331), CUDA/Metal wheels (remain build-from-source).

---

## 1. DEM parser completeness (`crates/aleph-qec/src/dem.rs`)

Real stim DEMs from `circuit.detector_error_model(decompose_errors=True)` use three features the
parser does not handle correctly today.

### 1.1 `repeat N { … }` and `shift_detectors`

Unrolled at parse time into the flat model the decoders consume. Semantics follow stim:

- `repeat N { body }` executes the body `N` times; blocks nest.
- `shift_detectors(c…) k` adds `k` to the running detector offset; every subsequent `D<i>` target
  (in `error` and `detector`) means `D<i + offset>`. Coordinate shifts are parsed and ignored
  (coords are ignored throughout).
- The offset is global state that persists across block iterations and after the block ends.
- `detector_separator` and any other unknown instruction remain a `DemParse` error naming the line.
- `N = 0` is legal (empty expansion). Unrolled size is bounded only by memory; a malformed/unclosed
  brace is a `DemParse` error.

`Error::UnsupportedDem` becomes unused for these two instructions; keep the variant only if something
else still returns it, otherwise remove it.

### 1.2 Preserve `^` decomposition

Today `error(p) D0 D1 ^ D2 D3` is merged into one 4-detector mechanism. `MatchingGraph::from_dem` then
returns `NonGraphlike`, so MWPM and Union-Find reject every decomposed circuit-level DEM — exactly the
DEMs PyMatching users generate.

Change: `DemError` gains
`pub components: Vec<(Vec<u32>, Vec<u32>)>` — the `^`-separated `(dets, obs)` parts, **empty when the
line had no `^`**. `dets`/`obs` keep their current meaning (the merged, symmetric-difference view), so
BP-family decoders are unaffected. `DemError::new` leaves `components` empty; a new
`DemError::with_components` builds a decomposed one. The 8 struct-literal sites in the workspace get
`components: Vec::new()`.

`MatchingGraph::from_dem`: for a mechanism with non-empty `components`, add **each component** as its
own edge with the mechanism's probability (PyMatching's treatment of decomposed errors); a component
with ≥3 detectors is still `NonGraphlike`. Mechanisms without components behave exactly as today.

`to_dem_string` emits components with ` ^ ` so `parse → to_dem_string → parse` round-trips.

### 1.3 Tests

- Unit: `repeat` (flat, nested, `N=0`), `shift_detectors` inside and outside repeats, unclosed brace,
  `^` component preservation, matching graph from a decomposed error.
- Proptest: round-trip `parse(to_dem_string(dem)) == dem` over random DEMs with components.
- Differential vs stim (Python test, §3): for stim-generated circuits, our parse of the compressed
  DEM equals our parse of `dem.flattened()`.

## 2. Python API

### 2.1 Package layout

```
crates/aleph-py/
├── pyproject.toml           # module-name = "aleph._native", python-source = "python",
│                            # optional-dependencies: sinter = ["stim", "sinter"]
├── python/aleph/
│   ├── __init__.py          # from ._native import *; __version__ — `import aleph` unchanged
│   ├── qec.py               # re-exports DetectorErrorModel, Decoder from _native.qec
│   └── sinter.py            # sinter adapters
└── src/qec.rs               # new pyo3 bindings
```

The `#[pymodule]` is renamed `_native`; every existing symbol stays importable as `aleph.<name>`.
pyo3 feature `abi3-py312` → `abi3-py310`.

### 2.2 Classes

```python
import aleph.qec as qec

dem = qec.DetectorErrorModel(text)            # str(stim.DetectorErrorModel) works
dem = qec.DetectorErrorModel.from_file(path)
dem.num_detectors, dem.num_observables, dem.num_errors

dec = qec.Decoder(dem, "relay-bp", **params)
dec.decode(dets)                     # 1-D bool/uint8 array [D]          -> bool [O]
dec.decode_batch(dets)               # 2-D bool/uint8 array [shots, D]   -> bool [shots, O]
dec.decode_batch_bit_packed(packed)  # uint8 [shots, ceil(D/8)]          -> uint8 [shots, ceil(O/8)]
```

`DetectorErrorModel` also accepts a `stim.DetectorErrorModel` object directly (converted with `str()`),
so stim is never imported by aleph itself.

Bit-packed layout is stim's: little-endian bit order within each byte (detector `d` is bit `d % 8`
of byte `d // 8`), rows are shots.

**Decoder names and parameters** (defaults = the Rust `new` defaults):

| name | Rust type | kwargs |
|---|---|---|
| `mwpm` | `MwpmDecoder` | — |
| `union-find` | `UnionFindDecoder::new` | — |
| `union-find-weighted` | `UnionFindDecoder::new_weighted` | — |
| `bp` | `BpDecoder` | `max_iter`, `alpha` |
| `bp-osd` | `OsdDecoder` | `max_iter`, `alpha`, `osd_order` |
| `relay-bp` | `RelayBpDecoder` | `legs`, `alpha`, `gamma_min`, `gamma_max`, `seed` |
| `relay-bp-osd` | `RelayBpOsdDecoder` | relay-bp kwargs + `osd_order` |

Unknown name or unknown kwarg → `ValueError` listing the valid ones. Non-graphlike DEM for a matching
decoder → `ValueError` carrying the Rust error text.

### 2.3 Execution

Batch methods: validate shape (`D` must equal `dem.num_detectors`, else `ValueError`), copy input into
Rust, **release the GIL**, decode shots in parallel with rayon (all listed decoders are `Sync`), then
build the output array. Bit (un)packing is done in Rust. Internally the enum
`AnyDecoder { Mwpm(..), UnionFind(..), … }` dispatches to `Decoder::decode`.

### 2.4 sinter adapter (`aleph/sinter.py`)

```python
import sinter, aleph.sinter
sinter.collect(tasks=..., decoders=["aleph-relay-bp"],
               custom_decoders=aleph.sinter.decoders(), num_workers=8)
```

- `AlephSinterDecoder(sinter.Decoder)` holds only `name: str` and `params: dict` — picklable by
  construction, which sinter's multiprocessing requires.
- `compile_decoder_for_dem(*, dem)` → `CompiledAlephDecoder(sinter.CompiledDecoder)` wrapping a
  `qec.Decoder`; `decode_shots_bit_packed(*, bit_packed_detection_event_data)` forwards to
  `decode_batch_bit_packed`.
- `decoders(params=None)` returns `{"aleph-<name>": AlephSinterDecoder(name, **params.get(name, {}))}` for
  every name in the table (`params` is a dict keyed by decoder name, since names contain `-`).
- `import aleph.sinter` without sinter installed raises `ImportError` pointing at
  `pip install aleph-sim[sinter]`; `import aleph` / `aleph.qec` never import sinter or stim.

## 3. Testing

- **Rust:** §1.3 unit + proptests; `cargo test -p aleph-qec`.
- **Python differential** (`scripts/python/test_qec.py`, runs whenever stim is installed): on stim
  surface-code memory circuits (d=3, 5; `decompose_errors=True`) and a stim `color_code:memory_xyz`
  DEM (hyperedges, `decompose_errors=False`; matching decoders must raise `ValueError` on it),
  `decode_batch` equals `decode` shot-by-shot and equals `decode_batch_bit_packed` after packing;
  results are deterministic across two runs; parse of compressed DEM equals parse of `flattened()`.
- **Oracle quality** (same file, skipped unless pymatching present): on the *same* 20 000 sampled shots
  of a d=5 surface code at p=0.005, `aleph` `mwpm` logical-error count vs `pymatching.decode_batch`
  count — `|a − p| ≤ max(5, 0.1·p)` (both are exact MWPM; differences come only from ties).
  `bp-osd` vs `ldpc`'s sinter BP-OSD on the color code if `ldpc` imports: aleph errors
  `≤ 1.5·ldpc + 10`; otherwise skipped with a message.
- **Pickle / multiprocessing:** round-trip every adapter; one `sinter.collect(num_workers=2)` run.
- **Existing suite:** `scripts/python/test_aleph.py` passes unchanged (proves the `_native` rename is
  transparent).
- **CI:** `test-python` job installs `stim sinter pymatching` and runs on a matrix of Python 3.10 and
  3.12.

## 4. v0.3.0 release

- Workspace version `0.2.0` → `0.3.0`.
- `CHANGELOG.md` (new) summarising v0.2.0 → v0.3.0 from `docs/perf/` reports: GPU Aer-parity
  (5.9), tiled fused block / host paging / FP32 (5.10), 5.11 items, noise v1, `aleph.qec`.
- README: a QEC section with a ≤10-line sinter example and one table — decode throughput
  (shots/s) of each aleph decoder vs pymatching / ldpc on the test DEMs, measured on a stated
  machine, reported honestly including where aleph is slower.
- `pyproject.toml`: classifiers for 3.10–3.13, description mentions QEC decoders.
- Wheels unchanged in kind (x86_64 manylinux, arm64 macOS, CPU-only). Release via the existing
  `release.yml`: TestPyPI dry-run → tag `v0.3.0` → PyPI. **Tag push and PyPI publish happen only after
  explicit user confirmation.**

## Acceptance criteria

1. A stim `surface_code:rotated_memory_x` DEM with `decompose_errors=True` decodes with every listed
   decoder (matching decoders included) without flattening on the user side.
2. `sinter.collect(..., custom_decoders=aleph.sinter.decoders(), num_workers≥2)` completes.
3. `aleph-mwpm` LER matches pymatching within CI on d=5.
4. Existing Python tests unchanged and green; CI green on Python 3.10 and 3.12; clippy/fmt clean.
5. v0.3.0 wheels install and pass the smoke test on a clean 3.10 venv.
