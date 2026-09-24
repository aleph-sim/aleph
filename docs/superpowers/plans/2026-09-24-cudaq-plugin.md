# CUDA-Q QEC decoder plugin (`aleph.cudaq`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register aleph's seven QEC decoders inside NVIDIA `cudaq-qec` 0.8 via a pure-Python plugin module `aleph.cudaq`, backed by new Rust/pyo3 surface that returns a per-column error estimate, and publish an honest A/B against NVIDIA's decoders.

**Architecture:** `aleph-qec` gains `DetectorErrorModel::from_check_matrices` and a per-decoder `decode_errors` (Sparse Blossom resolves to matched defect pairs and retraces each pair's shortest path; UF maps peeled edges to columns; BP family already has `ehat`). `aleph-py` exposes `dem_from_matrices`, `Decoder.decode_batch_errors`, `gross_code_dem`. `aleph/cudaq.py` is the only file that imports `cudaq_qec`; it adapts `H, O, error_rate_vec` → DEM and float syndromes → `ehat`.

**Tech Stack:** Rust 2021 (≥ 1.89), pyo3 0.22 + rust-numpy 0.22 (abi3-py310), maturin, Python ≥ 3.10, numpy, scipy (extra), `cudaq-qec` 0.8 (extra), stim (tests/harness). GPU box `openwebgui.splynx.com` (`/root/cqvenv`).

**Spec:** `docs/superpowers/specs/2026-09-24-cudaq-plugin-design.md`

**Deviation from the spec, decided while planning:** the spec names `DetectorErrorModel.from_matrices(...)` (staticmethod). The pyo3 class is `frozen` and scipy/dense normalisation is far simpler in Python, so the public entry point is the module function `aleph.qec.dem_from_matrices(H, O=None, error_rate_vec=None)` over a native `_dem_from_csc(...)`. Everything else in the spec is implemented as written.

## Global Constraints

- Rust edition 2021, MSRV 1.89; `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --check` must pass (CI runs the **beta** clippy too — run `cargo +beta clippy` or `cargo +nightly clippy` before pushing).
- No `unwrap()`/`expect()`/`panic!` in library code (tests are fine). Errors via the crate `Error` enum (thiserror). Every `>`/`<`/`.max`/`.clamp` on a float is preceded by an explicit `is_finite()` reject (ADR 0006).
- `aleph-qec` never imports or mentions `cudaq`. `aleph/cudaq.py` is the only file that imports `cudaq_qec`, and `import aleph` / `import aleph.qec` must keep working without it.
- Public API of `MwpmDecoder`, `Decoder` trait and `aleph.qec.Decoder` unchanged; everything new is additive.
- Every decoder packs observables into `u64`: `observables > 64` is an error at construction.
- Plugin decoder names: `aleph-mwpm`, `aleph-union-find`, `aleph-union-find-weighted`, `aleph-bp`, `aleph-bp-osd`, `aleph-relay-bp`, `aleph-relay-bp-osd`. Plugin kwargs = `aleph.qec.Decoder` kwargs (`max_iter`, `alpha`, `osd_order`, `legs`, `gamma_min`, `gamma_max`, `seed`) plus `error_rate` (scalar) and the cudaq-injected `O`, `error_rate_vec`.
- Extra in `pyproject.toml`: `cudaq = ["cudaq-qec>=0.8,<0.9", "scipy>=1.10"]`.
- Bench box must be idle before measuring (`uptime` load ≈ 0, `pgrep -af "cargo bench|bencher run|Runner.Worker|python"`).
- Commits: small, `[F6] <area>: <what>`; end with the session's Co-Authored-By / Claude-Session trailer lines.
- Branch: `f6-cudaq-plugin` in the main checkout (no worktrees).

## Review Focus

1. **Soft syndromes** (floats like 0.3 / 0.7 instead of 0/1): the plugin must threshold at `> 0.5` and give the same answer as the hard syndrome. — Task 6 test `test_soft_syndrome_thresholds`.
2. **Non-uint8 / sparse `H`** (int64 dense, scipy COO/CSR with explicit zeros or duplicates): `dem_from_matrices` must treat every non-zero as 1 and never densify sparse input. — Task 5 test `test_dem_from_matrices_accepts_dense_int64_and_sparse`.
3. **Empty batch** (`decode_batch([])`, zero shots): must return `result.shape == (0, E)`, `converged.shape == (0,)`, not raise. — Task 5 `test_decode_batch_errors_empty_batch`, Task 6 `test_decode_batch_empty`.
4. **Bad probabilities** (NaN, negative, > 1) and **duplicate indices in a column**: `from_check_matrices` must reject with a `ValueError` naming the column, never produce a DEM. — Task 1 tests `rejects_non_finite_probability`, `rejects_duplicate_index`.
5. **Hyperedge columns handed to a matching decoder** (cudaq keeps `^` parts as one column): `aleph-mwpm`/`aleph-union-find*` must raise a clear `ValueError` telling the user to pass the decomposed `H` (via `dem_to_matrices`). — Task 6 `test_matching_decoder_rejects_hyperedge_column`.

---

### Task 1: `DetectorErrorModel::from_check_matrices` (Rust)

**Files:**
- Modify: `crates/aleph-qec/src/error.rs` (add two variants after `NonGraphlike`)
- Modify: `crates/aleph-qec/src/dem.rs` (new constructor + tests in the existing `mod tests`)

**Interfaces:**
- Produces: `DetectorErrorModel::from_check_matrices(detectors: usize, observables: usize, h_cols: &[Vec<u32>], o_cols: &[Vec<u32>], probs: &[f64]) -> Result<DetectorErrorModel>`; `Error::CheckMatrix(String)`, `Error::TooManyObservables { observables: usize }`.

- [ ] **Step 1: Write the failing tests** — append inside `mod tests` in `crates/aleph-qec/src/dem.rs`:

```rust
    #[test]
    fn from_check_matrices_builds_one_mechanism_per_column() {
        let dem = DetectorErrorModel::from_check_matrices(
            3,
            1,
            &[vec![0], vec![1, 0], vec![2, 1], vec![2]],
            &[vec![0], vec![], vec![], vec![]],
            &[0.1, 0.2, 0.3, 0.4],
        )
        .unwrap();
        assert_eq!((dem.detectors, dem.observables, dem.errors.len()), (3, 1, 4));
        assert_eq!(dem.errors[1], DemError::new(0.2, vec![0, 1], vec![]));
        assert_eq!(dem.errors[0].obs, vec![0]);
        // Round-trips through the text form.
        assert_eq!(DetectorErrorModel::parse(&dem.to_dem_string()).unwrap(), dem);
    }

    #[test]
    fn from_check_matrices_rejects_length_mismatch() {
        let e = DetectorErrorModel::from_check_matrices(1, 0, &[vec![0]], &[], &[0.1]).unwrap_err();
        assert!(matches!(e, Error::CheckMatrix(_)), "{e}");
    }

    #[test]
    fn rejects_non_finite_probability() {
        for p in [f64::NAN, f64::INFINITY, -0.1, 1.5] {
            let e = DetectorErrorModel::from_check_matrices(1, 0, &[vec![0]], &[vec![]], &[p])
                .unwrap_err();
            assert!(e.to_string().contains("column 0"), "{p}: {e}");
        }
        // 0 and 1 are legal DEM probabilities.
        assert!(DetectorErrorModel::from_check_matrices(1, 0, &[vec![0]], &[vec![]], &[0.0]).is_ok());
        assert!(DetectorErrorModel::from_check_matrices(1, 0, &[vec![0]], &[vec![]], &[1.0]).is_ok());
    }

    #[test]
    fn rejects_duplicate_index() {
        let e = DetectorErrorModel::from_check_matrices(2, 0, &[vec![1, 1]], &[vec![]], &[0.1])
            .unwrap_err();
        assert!(e.to_string().contains("duplicate"), "{e}");
    }

    #[test]
    fn rejects_out_of_range_index() {
        let e = DetectorErrorModel::from_check_matrices(2, 1, &[vec![2]], &[vec![]], &[0.1]).unwrap_err();
        assert!(e.to_string().contains("detector 2"), "{e}");
        let e = DetectorErrorModel::from_check_matrices(2, 1, &[vec![0]], &[vec![1]], &[0.1]).unwrap_err();
        assert!(e.to_string().contains("observable 1"), "{e}");
    }

    #[test]
    fn rejects_more_than_64_observables() {
        let e = DetectorErrorModel::from_check_matrices(1, 65, &[vec![0]], &[vec![]], &[0.1]).unwrap_err();
        assert!(matches!(e, Error::TooManyObservables { observables: 65 }), "{e}");
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p aleph-qec dem::tests::from_check -- --nocapture 2>&1 | tail -5`
Expected: compile error "no function or associated item named `from_check_matrices`".

- [ ] **Step 3: Add the error variants** — in `crates/aleph-qec/src/error.rs`, after `NonGraphlike { .. },`:

```rust
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
```

- [ ] **Step 4: Implement the constructor** — in `crates/aleph-qec/src/dem.rs`, inside `impl DetectorErrorModel`, after `parse`:

```rust
    /// Build a model from column lists of a parity-check matrix `H` (`detectors × E`), an
    /// observable matrix `O` (`observables × E`) and a per-column prior: column `j` becomes one
    /// single-part mechanism with probability `probs[j]`, detectors `h_cols[j]` and observables
    /// `o_cols[j]`. This is the input a `cudaq-qec` decoder receives (`H`, `O`, `error_rate_vec`).
    ///
    /// # Errors
    /// [`Error::CheckMatrix`] if the three slices differ in length, an index is out of range or
    /// repeated within a column, or a probability is not finite or not in `[0, 1]`;
    /// [`Error::TooManyObservables`] if `observables > 64`.
    pub fn from_check_matrices(
        detectors: usize,
        observables: usize,
        h_cols: &[Vec<u32>],
        o_cols: &[Vec<u32>],
        probs: &[f64],
    ) -> Result<Self> {
        if h_cols.len() != probs.len() || o_cols.len() != probs.len() {
            return Err(Error::CheckMatrix(format!(
                "column count mismatch: H has {} columns, O has {}, error_rate_vec has {}",
                h_cols.len(),
                o_cols.len(),
                probs.len()
            )));
        }
        if observables > 64 {
            return Err(Error::TooManyObservables { observables });
        }
        let mut errors = Vec::with_capacity(probs.len());
        for (j, ((h, o), &p)) in h_cols.iter().zip(o_cols).zip(probs).enumerate() {
            // Explicit is_finite first: NaN passes every range comparison (ADR 0006).
            if !p.is_finite() || !(0.0..=1.0).contains(&p) {
                return Err(Error::CheckMatrix(format!(
                    "column {j}: probability {p} is not a finite number in [0, 1]"
                )));
            }
            check_column(j, "detector", h, detectors)?;
            check_column(j, "observable", o, observables)?;
            errors.push(DemError::new(p, h.clone(), o.clone()));
        }
        Ok(DetectorErrorModel {
            detectors,
            observables,
            errors,
        })
    }
```

and a free function next to `parity_reduce`:

```rust
/// `from_check_matrices` helper: every index in `idx` is `< limit` and appears once.
fn check_column(col: usize, what: &str, idx: &[u32], limit: usize) -> Result<()> {
    let mut sorted = idx.to_vec();
    sorted.sort_unstable();
    if let Some(w) = sorted.windows(2).find(|w| w[0] == w[1]) {
        return Err(Error::CheckMatrix(format!(
            "column {col}: duplicate {what} index {}",
            w[0]
        )));
    }
    if let Some(&i) = sorted.last() {
        if i as usize >= limit {
            return Err(Error::CheckMatrix(format!(
                "column {col}: {what} {i} out of range (model has {limit})"
            )));
        }
    }
    Ok(())
}
```

- [ ] **Step 5: Run the tests**

Run: `cargo test -p aleph-qec dem::tests 2>&1 | tail -5`
Expected: all `dem::tests` pass, including the six new ones.

- [ ] **Step 6: Lint and commit**

```bash
cargo clippy -p aleph-qec --all-targets -- -D warnings && cargo fmt
git add crates/aleph-qec/src/error.rs crates/aleph-qec/src/dem.rs
git commit -m "[F6] aleph-qec: DetectorErrorModel::from_check_matrices

One mechanism per column of (H, O, error_rate_vec) — the input a cudaq-qec
decoder receives. Rejects bad/duplicate indices, non-finite or out-of-range
probabilities, and > 64 observables (the u64 mask limit, #512)."
```

---

### Task 2: `MatchingEdge::column` and `UnionFindDecoder::decode_errors`

**Files:**
- Modify: `crates/aleph-qec/src/matching.rs` (struct field, `from_dem`, `num_columns`, tests)
- Modify: `crates/aleph-qec/src/union_find.rs` (`edge_col`, `num_columns`, `decode_errors`, tests)

**Interfaces:**
- Produces: `MatchingEdge { .., pub column: u32 }` — index into `dem.errors` of the first mechanism (and first `^` part) that produced this edge's `(a, b, observables)` key; `MatchingGraph::num_columns(&self) -> usize` (= `dem.errors.len()`); `UnionFindDecoder::decode_errors(&self, &Syndrome) -> Vec<u8>` of length `num_columns`.

- [ ] **Step 1: Write failing tests** — in `crates/aleph-qec/src/matching.rs` `mod tests`:

```rust
    #[test]
    fn edge_column_is_the_first_mechanism_with_that_key() {
        // Column 0 and 2 are parallel (same endpoints, same observables) → one edge, column 0.
        // Column 1 has the same endpoints but a different observable → its own edge, column 1.
        // Column 3 is a `^` mechanism: part 0 → boundary edge D2 (column 3), part 1 → edge D0 D1 L0
        //   which is parallel to column 1 → merged into column 1's edge.
        let dem = DetectorErrorModel::parse(
            "error(0.1) D0 D1\nerror(0.1) D0 D1 L0\nerror(0.2) D0 D1\nerror(0.05) D2 ^ D0 D1 L0\n",
        )
        .unwrap();
        let g = MatchingGraph::from_dem(&dem).unwrap();
        assert_eq!(g.num_columns(), 4);
        let cols: Vec<(NodeId, NodeId, Vec<u32>, u32)> = g
            .edges()
            .iter()
            .map(|e| (e.a, e.b, e.observables.clone(), e.column))
            .collect();
        assert_eq!(
            cols,
            vec![(0, 1, vec![], 0), (0, 1, vec![0], 1), (2, 3, vec![], 3)]
        );
        // The representative column's observable set equals the edge's.
        for e in g.edges() {
            assert_eq!(dem.errors[e.column as usize].components.is_empty() as u8 | 1, 1);
        }
    }
```

and in `crates/aleph-qec/src/union_find.rs` `mod tests`:

```rust
    #[test]
    fn decode_errors_satisfies_syndrome_and_reproduces_flips() {
        use crate::{build_dem, SurfaceCode};
        let exp = SurfaceCode::new(5).memory_z_experiment(5);
        let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(0.03, 0.03)).unwrap();
        let dec = UnionFindDecoder::new_weighted(&dem).unwrap();
        let mut rng = 0x1234_5678u64;
        for _ in 0..300 {
            let mut det = vec![false; dem.detectors];
            for e in &dem.errors {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                if (rng >> 40) as f64 / (1u64 << 24) as f64 < e.prob {
                    for &d in &e.dets {
                        det[d as usize] ^= true;
                    }
                }
            }
            let s = Syndrome::from_bits(&det);
            let ehat = dec.decode_errors(&s);
            assert_eq!(ehat.len(), dem.errors.len());
            // H ê = s and O ê = the decoder's own correction.
            let mut hs = vec![false; dem.detectors];
            let mut os = vec![false; dem.observables];
            for (j, &b) in ehat.iter().enumerate() {
                if b == 1 {
                    for &d in &dem.errors[j].dets {
                        hs[d as usize] ^= true;
                    }
                    for &o in &dem.errors[j].obs {
                        os[o as usize] ^= true;
                    }
                }
            }
            assert_eq!(hs, det, "H ê != s");
            assert_eq!(os, dec.decode(&s).observable_flips, "O ê != correction");
        }
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p aleph-qec edge_column decode_errors_satisfies 2>&1 | tail -5`
Expected: compile errors (`column` field, `num_columns`, `decode_errors` missing).

- [ ] **Step 3: Add the field and bookkeeping to `matching.rs`**

In `pub struct MatchingEdge`, after `observables`:

```rust
    /// Index into the source DEM's `errors` of the first mechanism (first `^` part) that produced
    /// this edge's `(a, b, observables)` key — the *representative column* a per-column error
    /// estimate marks when this edge is in the correction. Its observable set equals
    /// `observables` by construction.
    pub column: u32,
```

In `pub struct MatchingGraph`, after `num_observables`:

```rust
    /// Number of mechanisms (columns) in the source DEM; `MatchingEdge::column < num_columns`.
    num_columns: usize,
```

In `from_dem`: change `order` to carry the column and thread the DEM index:

```rust
        let mut order: Vec<((NodeId, NodeId, Vec<u32>), u32)> = Vec::new();

        for (j, e) in dem.errors.iter().enumerate() {
            // ... (unchanged filtering / parts loop) ...
                let key = (a, b, obs);
                if let Some(p) = merged.get_mut(&key) {
                    *p = xor_combine(*p, e.prob);
                } else {
                    merged.insert(key.clone(), e.prob);
                    order.push((key, j as u32));
                }
```

and in the edge-building loop:

```rust
        for (key, column) in order {
            let prob = merged[&key];
            let (a, b, observables) = key;
            let idx = edges.len();
            adjacency[a].push(idx);
            adjacency[b].push(idx);
            edges.push(MatchingEdge {
                a,
                b,
                prob,
                weight: edge_weight(prob),
                observables,
                column,
            });
        }

        Ok(MatchingGraph {
            num_detectors: dem.detectors,
            num_observables: dem.observables,
            num_columns: dem.errors.len(),
            edges,
            adjacency,
        })
```

Add the accessor after `num_observables()`:

```rust
    /// Number of mechanisms (columns) in the DEM this graph was built from.
    pub fn num_columns(&self) -> usize {
        self.num_columns
    }
```

Then `grep -rn "MatchingEdge {" crates/ --include='*.rs'` and add `column: <index>` to every struct literal in tests (use the edge's position in the literal list).

- [ ] **Step 4: Add `decode_errors` to `union_find.rs`**

Fields (after `edge_len`):

```rust
    /// Representative DEM column of each edge (`MatchingEdge::column`).
    edge_col: Vec<u32>,
    /// Number of DEM columns (`decode_errors` output length).
    num_columns: usize,
```

In `from_graph`, alongside `edge_a`/`edge_b`:

```rust
        let edge_col: Vec<u32> = edges.iter().map(|e| e.column).collect();
```

and in the struct literal at the end of `from_graph`: `edge_col, num_columns: graph.num_columns(),`.

Method, after `decode_edges`:

```rust
    /// Decode `syndrome` and return the error estimate over DEM columns: `ehat[j] == 1` ⇔ the
    /// peeled correction contains the edge whose representative column is `j`. Satisfies
    /// `H ê = s` on every detector with an incident edge, and `O ê` equals
    /// [`decode`](Decoder::decode)'s flips.
    pub fn decode_errors(&self, syndrome: &Syndrome) -> Vec<u8> {
        let (_, edges) = self.decode_edges(syndrome);
        let mut ehat = vec![0u8; self.num_columns];
        for e in edges {
            ehat[self.edge_col[e] as usize] ^= 1;
        }
        ehat
    }
```

- [ ] **Step 5: Run the whole crate's tests (the field change touches many tests)**

Run: `cargo test -p aleph-qec 2>&1 | tail -5`
Expected: all pass.

- [ ] **Step 6: Lint and commit**

```bash
cargo clippy -p aleph-qec --all-targets -- -D warnings && cargo fmt
git add crates/aleph-qec/src/matching.rs crates/aleph-qec/src/union_find.rs
git commit -m "[F6] aleph-qec: MatchingEdge::column + UnionFindDecoder::decode_errors

Each matching edge remembers the first DEM column that produced its
(endpoints, observables) key, so a per-column error estimate can be built
from a set of matched edges. Union-Find exposes it via decode_errors."
```

---

### Task 3: Sparse Blossom matched pairs + path retrace → `MwpmDecoder::decode_errors`

**Files:**
- Modify: `crates/aleph-qec/src/sparse_blossom/graph.rs` (column arrays)
- Modify: `crates/aleph-qec/src/sparse_blossom/matcher.rs` (`resolve_with`)
- Modify: `crates/aleph-qec/src/sparse_blossom/state.rs` (retrace scratch)
- Create: `crates/aleph-qec/src/sparse_blossom/retrace.rs`
- Modify: `crates/aleph-qec/src/sparse_blossom/mod.rs` (`with_state`, `run_edges`, `decode_errors`, tests)
- Modify: `crates/aleph-qec/src/mwpm.rs` (`decode_errors`, tests)

**Interfaces:**
- Consumes: `MatchingEdge::column`, `MatchingGraph::num_columns()` (Task 2).
- Produces: `MwpmDecoder::decode_errors(&self, &Syndrome) -> Vec<u8>` (length `num_columns`); crate-private `SparseMatcher::decode_errors(&self, defects: &[u32], ehat: &mut [u8]) -> (u64, i64)`.

Background for the implementer: a `CEdge { from, to, obs, weight }` produced by a collision is *defect-to-defect*: `from`/`to` are the two regions' source defects (`flooder.rs:139-144`, `to == BOUNDARY` for a boundary hit) and `weight` is the full path length in **doubled** integer units. By primal-dual tightness that length equals the shortest-path distance between the two defects, so a Dijkstra from `from` bounded at `weight` always reaches `to` (or a boundary edge) at exactly `weight`.

- [ ] **Step 1: Write the failing unit test** — in `crates/aleph-qec/src/sparse_blossom/mod.rs` `mod tests`:

```rust
    #[test]
    fn decode_errors_marks_the_matched_paths() {
        // Chain 0-1-2-3 (columns 0,1,2), boundary at 3 (column 3 via from_int_edges numbering:
        // boundary columns follow the edge columns).
        let g = CompiledGraph::from_int_edges(
            4,
            &[(0, 1, 2, 1), (1, 2, 2, 2), (2, 3, 2, 4)],
            &[(3, 5, 8)],
        );
        let m = SparseMatcher::new(g);
        let mut ehat = vec![0u8; 4];
        let (obs, w) = m.decode_errors(&[0, 2], &mut ehat);
        assert_eq!((obs, w), (3, 4));
        assert_eq!(ehat, vec![1, 1, 0, 0]);
        let mut ehat = vec![0u8; 4];
        let (obs, w) = m.decode_errors(&[2], &mut ehat);
        assert_eq!((obs, w), (4 ^ 8, 7));
        assert_eq!(ehat, vec![0, 0, 1, 1]);
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p aleph-qec decode_errors_marks 2>&1 | tail -3`
Expected: compile error — `decode_errors` not found.

- [ ] **Step 3: Carry columns through `CompiledGraph`** (`graph.rs`)

Add fields to `CompiledGraph`:

```rust
    /// Representative DEM column of `adj[k]` (parallel to `adj`).
    adj_col: Vec<u32>,
    /// Representative DEM column of each node's boundary edge (unused when absent).
    boundary_col: Vec<u32>,
```

Change the tuple types: edges become `(u32, u32, i64, u64, u32)` (`a, b, w, obs, col`), boundary `(u32, i64, u64, u32)`. In `from_matching_graph` append `e.column` to each tuple. In `from_int_edges` (test ctor) number columns as `edge index` for edges and `edges.len() + boundary index` for boundary entries:

```rust
        let e: Vec<_> = edges
            .iter()
            .enumerate()
            .map(|(k, &(a, b, w, o))| (a, b, 2 * w, o, k as u32))
            .collect();
        let b: Vec<_> = boundary
            .iter()
            .enumerate()
            .map(|(k, &(a, w, o))| (a, 2 * w, o, (edges.len() + k) as u32))
            .collect();
```

In `build`: `per` entries become `(u32, i64, usize, u64, u32)`; sort key unchanged `(v, w, k)`; when extending `adj` also push `col` into `adj_col`; boundary loop records `boundary_col[a] = c` whenever it records `boundary_w`. Accessors:

```rust
    /// Representative columns of `edges(u)`, parallel to it.
    #[inline]
    pub(crate) fn edge_cols(&self, u: u32) -> &[u32] {
        let u = u as usize;
        &self.adj_col[self.offsets[u] as usize..self.offsets[u + 1] as usize]
    }

    /// Representative column of `u`'s boundary edge (meaningful only when `boundary(u)` is `Some`).
    #[inline]
    pub(crate) fn boundary_col(&self, u: u32) -> u32 {
        self.boundary_col[u as usize]
    }
```

- [ ] **Step 4: `resolve_with` in `matcher.rs`** — make the three resolution functions generic over a sink and re-express `resolve` through it, so there is one traversal:

```rust
    /// Turn top-level matches into `(obs, doubled weight)`; blossoms are resolved recursively.
    pub(crate) fn resolve(&self) -> (u64, i64) {
        let (mut obs, mut w) = (0u64, 0i64);
        self.resolve_with(&mut |e: &CEdge| {
            obs ^= e.obs;
            w += e.weight;
        });
        (obs, w)
    }

    /// Visit every matched defect-to-defect edge of the final matching (`e.to == BOUNDARY` for a
    /// boundary match), top-level matches first, blossoms resolved recursively.
    pub(crate) fn resolve_with(&self, f: &mut impl FnMut(&CEdge)) {
        for r in 0..self.regions.len() as RegionId {
            let reg = &self.regions[r as usize];
            if reg.blossom_parent != NONE {
                continue;
            }
            if reg.matched_to == NONE {
                self.resolve_exposed(r, f);
                continue;
            }
            let e = reg.match_edge;
            if reg.matched_to == BOUNDARY {
                f(&e);
                self.descend(r, e.from, f);
            } else if r < reg.matched_to {
                f(&e);
                self.descend(r, e.from, f);
                self.descend(reg.matched_to, e.to, f);
            }
        }
    }
```

`resolve_exposed(&self, r, f: &mut impl FnMut(&CEdge))` and `descend(&self, r, defect, f: &mut impl FnMut(&CEdge))` keep their loops verbatim, replacing each `*obs ^= e.obs; *w += e.weight;` pair with `f(&e);` and passing `f` down. Run `cargo test -p aleph-qec sparse_blossom` — all existing tests must still pass before continuing.

- [ ] **Step 5: Retrace scratch + algorithm** — `state.rs`: add to `State`

```rust
    /// Path-retrace scratch (`retrace.rs`): tentative distance per node, `(predecessor node,
    /// column)` per node, and the nodes touched since the last reset.
    pub rt_dist: Vec<i64>,
    pub rt_pred: Vec<(NodeId, u32)>,
    pub rt_touched: Vec<NodeId>,
    pub rt_heap: std::collections::BinaryHeap<std::cmp::Reverse<(i64, NodeId)>>,
```

initialised in `State::new(n)` as `vec![i64::MAX; n]`, `vec![(NONE, 0); n]`, `Vec::new()`, `BinaryHeap::new()`.

Create `crates/aleph-qec/src/sparse_blossom/retrace.rs`:

```rust
//! Shortest-path retrace of a matched pair, for the per-column error output. The matcher only
//! remembers *which* defects are paired and the tight path length; like PyMatching's
//! `decode_to_edges`, the actual edges are recovered afterwards by a Dijkstra from one defect
//! bounded at that length. Ties are broken by `(distance, node index)` and first-relaxation, so
//! the result is deterministic; on a genuine tie the retraced path may differ from the path the
//! region growth took, which is why `decode_errors`' observable parity can differ from
//! `decode`'s — never its weight, never `H ê = s`.

use std::cmp::Reverse;

use super::graph::CompiledGraph;
use super::state::{CEdge, NodeId, State, BOUNDARY, NONE};

impl State {
    /// XOR the edges of a shortest path realising `e` (`e.from` → `e.to`, or `e.from` → any
    /// boundary edge when `e.to == BOUNDARY`) into `ehat` by representative column.
    pub(crate) fn retrace(&mut self, g: &CompiledGraph, e: &CEdge, ehat: &mut [u8]) {
        for &u in &self.rt_touched {
            self.rt_dist[u as usize] = i64::MAX;
            self.rt_pred[u as usize] = (NONE, 0);
        }
        self.rt_touched.clear();
        self.rt_heap.clear();

        let src = e.from;
        self.rt_dist[src as usize] = 0;
        self.rt_touched.push(src);
        self.rt_heap.push(Reverse((0, src)));
        let mut end: Option<(NodeId, Option<u32>)> = None; // (last node, boundary column)

        while let Some(Reverse((d, u))) = self.rt_heap.pop() {
            if d > self.rt_dist[u as usize] {
                continue;
            }
            if e.to == BOUNDARY {
                if let Some((wb, _)) = g.boundary(u) {
                    if d + wb == e.weight {
                        end = Some((u, Some(g.boundary_col(u))));
                        break;
                    }
                }
            } else if u == e.to {
                debug_assert_eq!(d, e.weight, "retrace reached the mate at the wrong distance");
                end = Some((u, None));
                break;
            }
            for (k, ne) in g.edges(u).iter().enumerate() {
                let nd = d + ne.w;
                if nd > e.weight {
                    continue; // bounded: nothing past the tight length can be on the path
                }
                let v = ne.v as usize;
                if nd < self.rt_dist[v] {
                    if self.rt_dist[v] == i64::MAX {
                        self.rt_touched.push(ne.v);
                    }
                    self.rt_dist[v] = nd;
                    self.rt_pred[v] = (u, g.edge_cols(u)[k]);
                    self.rt_heap.push(Reverse((nd, ne.v)));
                }
            }
        }

        // Tightness guarantees `end`; the fallback keeps library code panic-free if it ever
        // isn't (a bug upstream, caught by the debug assertion in tests).
        debug_assert!(end.is_some(), "retrace did not reach the mate within the tight weight");
        let Some((mut v, bcol)) = end else { return };
        if let Some(c) = bcol {
            ehat[c as usize] ^= 1;
        }
        while v != src {
            let (p, col) = self.rt_pred[v as usize];
            ehat[col as usize] ^= 1;
            v = p;
        }
    }
}
```

Add `pub(crate) mod retrace;` to `mod.rs`'s module list. (`NodeId`, `BOUNDARY`, `NONE`, `CEdge` are already `pub(crate)` in `state.rs` — confirm with `grep -n "pub(crate) const BOUNDARY\|pub(crate) type NodeId" crates/aleph-qec/src/sparse_blossom/state.rs`; make them `pub(crate)` if they are private.)

- [ ] **Step 6: `with_state` refactor + `run_edges` + `decode_errors`** in `mod.rs`

Split `State::run` so the event loop is reusable:

```rust
impl State {
    /// Decode one syndrome: `(observable mask, total weight in undoubled units)`.
    pub(crate) fn run(&mut self, g: &CompiledGraph, defects: &[u32]) -> (u64, i64) {
        self.run_to_matching(g, defects);
        let (obs, w) = self.resolve();
        (obs, w / 2)
    }

    /// Like `run`, but also XORs every matched path's edges into `ehat` by column.
    pub(crate) fn run_edges(&mut self, g: &CompiledGraph, defects: &[u32], ehat: &mut [u8]) -> (u64, i64) {
        self.run_to_matching(g, defects);
        let mut matched: Vec<CEdge> = Vec::new();
        self.resolve_with(&mut |e: &CEdge| matched.push(*e));
        let (mut obs, mut w) = (0u64, 0i64);
        for e in &matched {
            obs ^= e.obs;
            w += e.weight;
            self.retrace(g, e, ehat);
        }
        (obs, w / 2)
    }

    /// Everything `run` does up to (not including) `resolve`.
    fn run_to_matching(&mut self, g: &CompiledGraph, defects: &[u32]) {
        // body of the old `run` from `debug_assert!(defects.windows(2)...)` through
        // `if self.active_trees > 0 { self.dissolve_leftover_trees(g); }`
    }
}
```

(`matched` is collected first because `retrace` needs `&mut self` while `resolve_with` borrows `&self`; `CEdge` must derive `Copy` — check `state.rs`, add `#[derive(Clone, Copy, ...)]` if missing.)

Replace the body of `decode_with_stats` with a shared cache accessor and add `decode_errors`:

```rust
    /// Run `f` on this thread's cached `State` for this decoder (allocating and caching one on
    /// first use; evicting FIFO past `CACHE_CAP`). Falls back to an uncached `State` if the
    /// thread-local is unavailable (TLS teardown) — never panics.
    fn with_state<R>(&self, f: impl FnOnce(&mut State) -> R) -> R {
        let n = self.graph.num_nodes();
        CACHE
            .try_with(|cache| {
                if let Ok(mut cache) = cache.try_borrow_mut() {
                    if let Some((_, st)) = cache.iter_mut().find(|(id, _)| *id == self.id) {
                        return f(st);
                    }
                    if cache.len() >= CACHE_CAP {
                        cache.remove(0);
                    }
                    let mut st = State::new(n);
                    let out = f(&mut st);
                    cache.push((self.id, st));
                    out
                } else {
                    f(&mut State::new(n))
                }
            })
            .unwrap_or_else(|_| f(&mut State::new(n)))
    }

    pub(crate) fn decode_with_stats(&self, defects: &[u32]) -> ((u64, i64), Stats) {
        self.with_state(|st| {
            let out = st.run(&self.graph, defects);
            (out, st.stats)
        })
    }

    /// `decode`, plus the matched paths XORed into `ehat` by representative DEM column.
    pub(crate) fn decode_errors(&self, defects: &[u32], ehat: &mut [u8]) -> (u64, i64) {
        self.with_state(|st| st.run_edges(&self.graph, defects, ehat))
    }
```

The `FnOnce` closure in `with_state` is called in exactly one branch per path, so the compiler accepts it; if it complains about a possibly-moved closure, restructure as `let mut f = Some(f);` and `f.take()` in each branch.

- [ ] **Step 7: `MwpmDecoder::decode_errors`** in `mwpm.rs`, after `decode_sparse`:

```rust
    /// Decode `syndrome` and return the error estimate over DEM columns (`ehat[j] == 1` ⇔ the
    /// mechanism `j` is in the correction), the output a `cudaq-qec` decoder returns. The matched
    /// pairs are the same as [`decode`](Decoder::decode)'s; each pair's path is retraced by a
    /// bounded shortest-path search, so `H ê = s` and the total weight always agree with
    /// `decode`, while the observable parity may differ on a genuine tie (equal-weight paths).
    pub fn decode_errors(&self, syndrome: &Syndrome) -> Vec<u8> {
        let mut ehat = vec![0u8; self.graph.num_columns()];
        let defects = self.defects_of(syndrome);
        if !defects.is_empty() {
            let d32: Vec<u32> = defects.iter().map(|&d| d as u32).collect();
            self.sparse.decode_errors(&d32, &mut ehat);
        }
        ehat
    }

    /// Test/bench helper: the scaled weight of the edges an `ehat` marks (undoubled units, so it
    /// compares with `decode_sparse`'s weight).
    #[doc(hidden)]
    pub fn ehat_weight(&self, ehat: &[u8]) -> i64 {
        self.graph
            .edges()
            .iter()
            .filter(|e| ehat[e.column as usize] == 1)
            .map(|e| (e.weight * WEIGHT_SCALE).round() as i64)
            .sum()
    }
```

- [ ] **Step 8: Run the unit test**

Run: `cargo test -p aleph-qec decode_errors_marks 2>&1 | tail -3`
Expected: PASS.

- [ ] **Step 9: Property + differential tests** — in `sparse_blossom/mod.rs` `mod tests`, extend the `proptest!` block:

```rust
        #[test]
        fn decode_errors_is_weight_and_syndrome_exact((g, defects) in graph_strategy()) {
            let has_boundary = (0..g.num_nodes() as u32).any(|u| g.boundary(u).is_some());
            prop_assume!(defects.len() % 2 == 0 || has_boundary);
            let m = SparseMatcher::new(g.clone());
            let (_, w) = m.decode(&defects);
            // Column numbering from `from_int_edges`: edges 0..E, then boundary edges.
            let ncols = (0..g.num_nodes() as u32).map(|u| g.edge_cols(u).iter().copied().max().map_or(0, |c| c + 1)).max().unwrap_or(0)
                .max((0..g.num_nodes() as u32).filter(|&u| g.boundary(u).is_some()).map(|u| g.boundary_col(u) + 1).max().unwrap_or(0)) as usize;
            let mut ehat = vec![0u8; ncols];
            let (_, we) = m.decode_errors(&defects, &mut ehat);
            prop_assert_eq!(we, w);
            // Σ weight of marked columns == matching weight, and parity at every node == defect.
            let mut sum = 0i64;
            let mut parity = vec![0u8; g.num_nodes()];
            for u in 0..g.num_nodes() as u32 {
                for (e, &c) in g.edges(u).iter().zip(g.edge_cols(u)) {
                    if u < e.v && ehat[c as usize] == 1 {
                        sum += e.w;
                        parity[u as usize] ^= 1;
                        parity[e.v as usize] ^= 1;
                    }
                }
                if let Some((wb, _)) = g.boundary(u) {
                    if ehat[g.boundary_col(u) as usize] == 1 {
                        sum += wb;
                        parity[u as usize] ^= 1;
                    }
                }
            }
            prop_assert_eq!(sum / 2, w, "marked-edge weight != matching weight");
            for u in 0..g.num_nodes() as u32 {
                prop_assert_eq!(parity[u as usize] == 1, defects.contains(&u), "node {} parity", u);
            }
        }
```

and in `mwpm.rs` `mod tests`, after `sparse_matches_dense_on_circuit_level_dems`:

```rust
    /// `decode_errors` invariants on the same phenomenological shot set as the #331 differential:
    /// H ê = s and Σw(ê) = matching weight on every shot; O ê disagrees with `decode` only at a
    /// genuine-tie rate bounded by the `decode_local` sentinel.
    #[test]
    fn decode_errors_invariants_on_phenomenological_shots() {
        use crate::{build_dem, SurfaceCode};
        for d in [3usize, 5, 7, 9, 11] {
            for p in [0.01, 0.03, 0.06] {
                let exp = SurfaceCode::new(d).memory_z_experiment(d);
                let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
                let dec = MwpmDecoder::new(&dem).unwrap();
                let shots = if cfg!(debug_assertions) { 40 } else if d >= 11 { 1500 } else { 8000 };
                let (mut ties, mut local_ties) = (0usize, 0usize);
                for fired in sample_defects(&dem, shots, 0xE44 ^ (d as u64) << 8 ^ (p * 1000.0) as u64) {
                    let s = Syndrome::new(dem.detectors, fired);
                    let (cs, ws) = dec.decode_sparse(&s);
                    let ehat = dec.decode_errors(&s);
                    assert_eq!(dec.ehat_weight(&ehat), ws, "d={d} p={p}: Σw(ê) != weight on {:?}", s.fired);
                    let mut hs = vec![false; dem.detectors];
                    let mut os = vec![false; dem.observables];
                    for (j, &b) in ehat.iter().enumerate() {
                        if b == 1 {
                            for &x in &dem.errors[j].dets { hs[x as usize] ^= true; }
                            for &o in &dem.errors[j].obs { os[o as usize] ^= true; }
                        }
                    }
                    let want: Vec<bool> = (0..dem.detectors as u32).map(|x| s.is_fired(x)).collect();
                    assert_eq!(hs, want, "d={d} p={p}: H ê != s on {:?}", s.fired);
                    if os != cs.observable_flips { ties += 1; }
                    if dec.decode_local(&s).0 != dec.decode_dense_weighted(&s).0 { local_ties += 1; }
                }
                assert!(ties <= local_ties * 2 + 20, "d={d} p={p}: {ties} O·ê disagreements vs local-oracle baseline {local_ties}");
            }
        }
    }
```

- [ ] **Step 10: Run them (release for the differential — it is the same size as the #331 one)**

Run: `cargo test -p aleph-qec decode_errors 2>&1 | tail -5 && cargo test --release -p aleph-qec decode_errors_invariants 2>&1 | tail -3`
Expected: PASS (debug proptest 3000 cases; release differential ~1–2 min).

- [ ] **Step 11: Existing sparse tests + whole crate, lint, commit**

Run: `cargo test -p aleph-qec 2>&1 | tail -3 && cargo clippy -p aleph-qec --all-targets -- -D warnings && cargo fmt`

```bash
git add crates/aleph-qec/src/sparse_blossom crates/aleph-qec/src/mwpm.rs
git commit -m "[F6] sparse_blossom: matched-pair retrace → MwpmDecoder::decode_errors

resolve_with visits the final matching's defect-to-defect edges; each is
retraced by a Dijkstra bounded at its tight length (as PyMatching's
decode_to_edges does) and XORed into a per-column error estimate. Weight
and H·ê=s are exact; observable parity may differ from decode() on ties."
```

---

### Task 4: `decode_errors` for the BP family

**Files:**
- Modify: `crates/aleph-qec/src/bp.rs`, `osd.rs`, `relay_bp.rs` (one method each + tests)

**Interfaces:**
- Produces: `BpDecoder::decode_errors(&self, &Syndrome) -> (Vec<u8>, bool)`, `OsdDecoder::decode_errors -> (Vec<u8>, bool)`, `RelayBpDecoder::decode_errors -> (Vec<u8>, bool)`, `RelayBpOsdDecoder::decode_errors -> (Vec<u8>, bool)`. Second element = `converged` per the spec's §4.2 table: BP — BP converged; OSD — `true` when OSD ran or BP converged (always `true`); relay — a valid solution was found; relay+OSD — `true`.

- [ ] **Step 1: Failing tests** — in `relay_bp.rs` `mod tests` (the file already has gross-code fixtures; reuse `BBCode::gross().code_capacity_dem(0.01)` or whatever DEM the existing tests construct — check `grep -n "code_capacity_dem\|fn dem" crates/aleph-qec/src/relay_bp.rs`):

```rust
    #[test]
    fn decode_errors_agrees_with_decode_and_satisfies_syndrome_when_converged() {
        use crate::{BBCode, BpDecoder, OsdDecoder};
        let dem = BBCode::gross().code_capacity_dem(0.01);
        let (relay, osd, bp, relay_osd) = (
            RelayBpDecoder::new(&dem),
            OsdDecoder::new(&dem),
            BpDecoder::new(&dem),
            RelayBpOsdDecoder::new(&dem, 0),
        );
        let (syns, _) = crate::sample_shots(&dem, 200, 7);
        let check = |name: &str, s: &Syndrome, ehat: &[u8], conv: bool, flips: &[bool]| {
            assert_eq!(ehat.len(), dem.errors.len(), "{name}");
            let mut hs = vec![false; dem.detectors];
            let mut os = vec![false; dem.observables];
            for (j, &b) in ehat.iter().enumerate() {
                if b == 1 {
                    for &d in &dem.errors[j].dets { hs[d as usize] ^= true; }
                    for &o in &dem.errors[j].obs { os[o as usize] ^= true; }
                }
            }
            if conv {
                let want: Vec<bool> = (0..dem.detectors as u32).map(|d| s.is_fired(d)).collect();
                assert_eq!(hs, want, "{name}: converged but H ê != s");
            }
            assert_eq!(os, flips, "{name}: O ê != decode()");
        };
        for s in &syns {
            let (e, c) = bp.decode_errors(s);
            check("bp", s, &e, c, &bp.decode(s).observable_flips);
            let (e, c) = osd.decode_errors(s);
            assert!(c);
            check("osd", s, &e, c, &osd.decode(s).observable_flips);
            let (e, c) = relay.decode_errors(s);
            check("relay", s, &e, c, &relay.decode(s).observable_flips);
            let (e, c) = relay_osd.decode_errors(s);
            assert!(c);
            check("relay-osd", s, &e, c, &relay_osd.decode(s).observable_flips);
        }
    }
```

(`Decoder` trait must be in scope: `use crate::Decoder;` at the top of the test module if not already.)

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p aleph-qec decode_errors_agrees 2>&1 | tail -3` → compile error.

- [ ] **Step 3: Implement** — `bp.rs`, after `decode_bp_soft`:

```rust
    /// Per-column error estimate `ê` and whether BP converged (`H ê = s` held within the
    /// iteration cap). The output a `cudaq-qec` decoder returns.
    pub fn decode_errors(&self, syndrome: &Syndrome) -> (Vec<u8>, bool) {
        let soft = self.decode_bp_soft(syndrome);
        (soft.ehat, soft.converged)
    }
```

`osd.rs`, after `decode_osd_ehat`:

```rust
    /// Per-column error estimate `ê` and a convergence flag. OSD always returns a solution of
    /// `H ê = s` (BP's if it converged, otherwise the OSD solve), so the flag is always `true`;
    /// it exists so every decoder has the same `(ê, converged)` shape.
    pub fn decode_errors(&self, syndrome: &Syndrome) -> (Vec<u8>, bool) {
        let (ehat, _osd_ran) = self.decode_osd_ehat(syndrome);
        (ehat, true)
    }
```

`relay_bp.rs`, in `impl RelayBpDecoder` after `decode_soft`:

```rust
    /// Per-column error estimate `ê` and whether any relay leg found a valid solution.
    pub fn decode_errors(&self, syndrome: &Syndrome) -> (Vec<u8>, bool) {
        let soft = self.decode_soft(syndrome);
        (soft.ehat, soft.converged)
    }
```

and in `impl RelayBpOsdDecoder` after `decode_relay_osd`:

```rust
    /// Per-column error estimate `ê`: relay-BP's if it converged, else the OSD refinement of its
    /// soft output. Always satisfies `H ê = s`, so the flag is always `true`.
    pub fn decode_errors(&self, syndrome: &Syndrome) -> (Vec<u8>, bool) {
        let soft = self.relay.decode_soft(syndrome);
        if soft.converged {
            return (soft.ehat, true);
        }
        (self.osd.ehat_from_soft(syndrome, &soft), true)
    }
```

`OsdDecoder` needs a public `ehat_from_soft` that mirrors `correction_from_soft` without the projection — add next to it in `osd.rs`:

```rust
    /// [`correction_from_soft`](Self::correction_from_soft) without the observable projection:
    /// the per-column decision itself.
    pub fn ehat_from_soft(&self, syndrome: &Syndrome, soft: &crate::BpSoft) -> Vec<u8> {
        if soft.converged {
            return soft.ehat.clone();
        }
        self.osd_solve(syndrome, &soft.ehat, &soft.llr)
    }
```

and make `correction_from_soft` call it (`self.bp.correction_of(&self.ehat_from_soft(syndrome, soft))`) so there is one path.

- [ ] **Step 4: Run tests, lint, commit**

Run: `cargo test -p aleph-qec decode_errors_agrees 2>&1 | tail -3 && cargo test -p aleph-qec 2>&1 | tail -3 && cargo clippy -p aleph-qec --all-targets -- -D warnings && cargo fmt`

```bash
git add crates/aleph-qec/src/bp.rs crates/aleph-qec/src/osd.rs crates/aleph-qec/src/relay_bp.rs
git commit -m "[F6] aleph-qec: decode_errors for BP, BP+OSD, relay-BP, relay-BP+OSD

Uniform (ehat, converged) shape over the existing soft/ehat paths."
```

---

### Task 5: pyo3 surface — `dem_from_matrices`, `decode_batch_errors`, `num_errors`, `gross_code_dem`

**Files:**
- Modify: `crates/aleph-py/src/qec_core.rs` (`AnyDecoder::decode_errors`, `decode_errors_rows`, tests)
- Modify: `crates/aleph-py/src/qec.rs` (`_dem_from_csc`, `decode_batch_errors`, `num_errors`, `gross_code_dem`, registration)
- Modify: `crates/aleph-py/python/aleph/qec.py` (`dem_from_matrices`)
- Modify: `scripts/python/test_qec.py` (tests)

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: Python `aleph.qec.dem_from_matrices(H, O=None, error_rate_vec=None) -> DetectorErrorModel`; `aleph.qec.Decoder.decode_batch_errors(dets) -> (np.ndarray[uint8, (shots, E)], np.ndarray[bool, (shots,)])`; `aleph.qec.Decoder.num_errors: int`; `aleph.qec.gross_code_dem(rounds: int, p: float) -> DetectorErrorModel`. Native: `aleph._native.qec._dem_from_csc(detectors, observables, h_indptr, h_indices, o_indptr, o_indices, probs)`.

- [ ] **Step 1: Rust-side batch driver + test** — `qec_core.rs`: add to `impl AnyDecoder`

```rust
    /// Per-column error estimate and convergence flag (see the crate docs of each decoder).
    pub fn decode_errors(&self, s: &Syndrome) -> (Vec<u8>, bool) {
        match self {
            AnyDecoder::Mwpm(d) => (d.decode_errors(s), true),
            AnyDecoder::UnionFind(d) => (d.decode_errors(s), true),
            AnyDecoder::Bp(d) => d.decode_errors(s),
            AnyDecoder::BpOsd(d) => d.decode_errors(s),
            AnyDecoder::Relay(d) => d.decode_errors(s),
            AnyDecoder::RelayOsd(d) => d.decode_errors(s),
        }
    }

```

(The column count lives on `PyDecoder` in `qec.rs` — set from `dem.inner.errors.len()` — and is
passed into the driver; `AnyDecoder` stays a plain enum.)

Free function after `decode_packed`:

```rust
/// Decode dense 0/1 rows `[shots × detectors]` to per-column error rows `[shots × errors]` plus a
/// converged flag per shot.
pub fn decode_errors_rows(
    dec: &AnyDecoder,
    bits: &[u8],
    shots: usize,
    detectors: usize,
    errors: usize,
) -> (Vec<u8>, Vec<bool>) {
    let mut out = vec![0u8; shots * errors];
    let mut conv = vec![false; shots];
    if shots == 0 || errors == 0 {
        // `par_chunks_mut(0)` panics; an empty row also carries nothing to write.
        if errors == 0 {
            conv.iter_mut().for_each(|c| *c = true);
        }
        return (out, conv);
    }
    let rows: Vec<(&mut [u8], &mut bool)> = out.chunks_mut(errors).zip(conv.iter_mut()).collect();
    rows.into_par_iter().enumerate().for_each(|(i, (o, c))| {
        let fired = if detectors == 0 {
            Vec::new()
        } else {
            let row = &bits[i * detectors..(i + 1) * detectors];
            (0..detectors as u32).filter(|&d| row[d as usize] != 0).collect()
        };
        let (ehat, ok) = dec.decode_errors(&Syndrome { detectors, fired });
        o.copy_from_slice(&ehat);
        *c = ok;
    });
    (out, conv)
}
```

`AnyDecoder` must be `Sync` for `into_par_iter` — every variant already is (the batch drivers rely on it); if the compiler disagrees, `unsafe impl Sync` is **not** the answer — find which variant isn't and fix it.

Test in `qec_core::tests` (the module has a 3-detector `DEM` const):

```rust
    #[test]
    fn decode_errors_rows_matches_decode_rows_through_o() {
        let dem = DetectorErrorModel::parse(DEM).unwrap();
        for name in ["mwpm", "union-find", "bp", "bp-osd", "relay-bp", "relay-bp-osd"] {
            let dec = AnyDecoder::build(&dem, name, &DecoderParams::default()).unwrap();
            let bits = [1u8, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1]; // 4 shots × 3 detectors
            let flips = decode_rows(dec.get(), &bits, 4, 3, 1);
            let (ehat, conv) = decode_errors_rows(&dec, &bits, 4, 3, dem.errors.len());
            assert_eq!(conv, vec![true; 4], "{name}");
            for shot in 0..4 {
                let row = &ehat[shot * 4..(shot + 1) * 4];
                let o: u8 = row.iter().zip(&dem.errors).map(|(&b, e)| b & (e.obs.contains(&0) as u8)).fold(0, |a, b| a ^ b);
                assert_eq!(o, flips[shot], "{name} shot {shot}");
            }
        }
        let (e, c) = decode_errors_rows(&AnyDecoder::build(&dem, "mwpm", &DecoderParams::default()).unwrap(), &[], 0, 3, 4);
        assert!(e.is_empty() && c.is_empty());
    }
```

Run: `cargo test -p aleph-py decode_errors_rows 2>&1 | tail -3` → PASS (after implementation).

- [ ] **Step 2: pyo3 methods** — `qec.rs`:

Add `errors: usize` to `PyDecoder` (set from `dem.inner.errors.len()` in `new`), a getter, and the batch method:

```rust
    /// Number of error mechanisms (DEM columns) in a `decode_batch_errors` row.
    #[getter]
    fn num_errors(&self) -> usize {
        self.errors
    }

    /// Decode `[shots, num_detectors]` bool/uint8 -> `(errors uint8 [shots, num_errors],
    /// converged bool [shots])`: the per-column error estimate every decoder can produce (what a
    /// cudaq-qec decoder returns). GIL released; shots decode in parallel.
    fn decode_batch_errors<'py>(
        &self,
        py: Python<'py>,
        dets: &Bound<'py, PyAny>,
    ) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray1<bool>>)> {
        let (shots, bits) = to_bytes(dets, 2, self.detectors, "decode_batch_errors")?;
        let (d, e) = (self.detectors, self.errors);
        let dec = &self.dec;
        let (out, conv) = py.allow_threads(|| qec_core::decode_errors_rows(dec, &bits, shots, d, e));
        let errors = PyArray1::from_vec_bound(py, out).reshape([shots, e])?;
        Ok((errors, PyArray1::from_vec_bound(py, conv)))
    }
```

`to_bytes` with `ndim == 2` and zero rows: `shape[0] == 0` is fine (rows = 0, data empty) — verify by the empty-batch Python test below.

Native DEM-from-CSC function (module-level `#[pyfunction]`):

```rust
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
                .map(|&i| u32::try_from(i).map_err(|_| value_err(format!("{what}: negative index {i}"))))
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
        return Err(value_err(format!("p must be a finite number in [0, 1), got {p}")));
    }
    aleph_qec::BBCode::gross()
        .circuit_level_dem(rounds, aleph_qec::CircuitNoise::uniform(p))
        .map(|inner| PyDem { inner })
        .map_err(|e| value_err(e.to_string()))
}
```

Register both in `register()`: `m.add_function(wrap_pyfunction!(_dem_from_csc, &m)?)?; m.add_function(wrap_pyfunction!(gross_code_dem, &m)?)?;`.

- [ ] **Step 3: Python wrapper** — `crates/aleph-py/python/aleph/qec.py`:

```python
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
```

Keep the module docstring at the top as it is.

- [ ] **Step 4: Python tests** — append to `scripts/python/test_qec.py` inside `TestApi` (needs only aleph + numpy):

```python
    def test_dem_from_matrices_accepts_dense_int64_and_sparse(self):
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int64)
        O = np.array([[1, 0, 0]], dtype=np.int64)
        dem = qec.dem_from_matrices(H, O, [0.1, 0.2, 0.3])
        self.assertEqual((dem.num_detectors, dem.num_observables, dem.num_errors), (2, 1, 3))
        self.assertEqual(dem.to_dem_string(), qec.DetectorErrorModel("error(0.1) D0 L0\nerror(0.2) D0 D1\nerror(0.3) D1\n").to_dem_string())
        try:
            import scipy.sparse as sp
        except ImportError:
            return
        coo = sp.coo_matrix(([1, 1, 1, 0, 1], ([0, 0, 1, 1, 1], [0, 1, 1, 2, 2])), shape=(2, 3))  # explicit zero + duplicate at (1,2)
        dem2 = qec.dem_from_matrices(coo, sp.csr_matrix(O), np.array([0.1, 0.2, 0.3]))
        self.assertEqual(dem2.to_dem_string(), dem.to_dem_string())
        self.assertEqual(qec.dem_from_matrices(H, None, [0.1, 0.2, 0.3]).num_observables, 0)

    def test_dem_from_matrices_rejects_bad_input(self):
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "error_rate_vec"):
            qec.dem_from_matrices(H)
        with self.assertRaisesRegex(ValueError, "3 columns"):
            qec.dem_from_matrices(H, None, [0.1, 0.2])
        with self.assertRaisesRegex(ValueError, "column 1"):
            qec.dem_from_matrices(H, None, [0.1, float("nan"), 0.3])
        with self.assertRaisesRegex(ValueError, "at most 64"):
            qec.dem_from_matrices(H, np.ones((65, 3), dtype=np.uint8), [0.1, 0.2, 0.3])

    def test_decode_batch_errors(self):
        dem = qec.DetectorErrorModel(self.DEM)  # D0 L0 | D0 D1 | D1
        for name in ALL:
            dec = qec.Decoder(dem, name)
            self.assertEqual(dec.num_errors, 3)
            dets = np.array([[1, 0], [0, 0], [1, 1], [0, 1]], dtype=bool)
            ehat, conv = dec.decode_batch_errors(dets)
            self.assertEqual((ehat.dtype, ehat.shape, conv.dtype, conv.shape), (np.uint8, (4, 3), np.bool_, (4,)))
            # H ê = s (all converge on this tiny model) and O ê = decode_batch.
            H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
            self.assertTrue(conv.all(), name)
            np.testing.assert_array_equal((ehat @ H.T) % 2, dets.astype(np.uint8), name)
            O = np.array([[1, 0, 0]], dtype=np.uint8)
            np.testing.assert_array_equal(((ehat @ O.T) % 2).astype(bool), dec.decode_batch(dets), name)

    def test_decode_batch_errors_empty_batch(self):
        dec = qec.Decoder(qec.DetectorErrorModel(self.DEM), "mwpm")
        ehat, conv = dec.decode_batch_errors(np.zeros((0, 2), dtype=bool))
        self.assertEqual((ehat.shape, conv.shape), ((0, 3), (0,)))

    def test_gross_code_dem(self):
        dem = qec.gross_code_dem(2, 0.003)
        self.assertEqual(dem.num_observables, 12)
        self.assertGreater(dem.num_detectors, 100)
        with self.assertRaises(ValueError):
            qec.gross_code_dem(0, 0.003)
        with self.assertRaises(ValueError):
            qec.gross_code_dem(2, float("nan"))
```

- [ ] **Step 5: Build the wheel and run**

```bash
cd crates/aleph-py && maturin develop --release && cd ../..
python -m unittest scripts.python.test_qec -v 2>&1 | tail -15
```

(If `maturin develop` needs a venv: `uv venv -p 3.12 /tmp/f6venv && VIRTUAL_ENV=/tmp/f6venv /tmp/f6venv/bin/maturin develop --release` after `uv pip install -p /tmp/f6venv/bin/python maturin numpy scipy stim pymatching sinter`; then run the tests with `/tmp/f6venv/bin/python`.)

Expected: all `test_qec` tests pass, including the five new ones.

- [ ] **Step 6: Lint (with the `python` feature, since that is what CI builds) and commit**

```bash
cargo clippy -p aleph-py --features python --all-targets -- -D warnings && cargo fmt
git add crates/aleph-py/src/qec_core.rs crates/aleph-py/src/qec.rs crates/aleph-py/python/aleph/qec.py scripts/python/test_qec.py
git commit -m "[F6] aleph-py: dem_from_matrices, decode_batch_errors, gross_code_dem

The three pieces of Python surface the cudaq plugin needs: a DEM from
(H, O, error_rate_vec) without densifying sparse input, a GIL-free batch
returning the per-column error estimate + converged flag, and the gross
code's circuit-level DEM (cudaq has no bivariate-bicycle code)."
```

---

### Task 6: `aleph.cudaq` plugin module + extra + `test_cudaq.py`

**Files:**
- Create: `crates/aleph-py/python/aleph/cudaq.py`
- Modify: `crates/aleph-py/pyproject.toml` (extra)
- Create: `scripts/python/test_cudaq.py`

**Interfaces:**
- Consumes: Task 5's Python API.
- Produces: `aleph.cudaq.NAMES` (tuple of the seven bare names), `aleph.cudaq.DECODERS` (`{bare name: class}`), `aleph.cudaq.dem_to_matrices(dem) -> (H: np.uint8 [D×E], O: np.uint8 [num_obs×E], rates: np.float64 [E])` (splits `^` parts into separate columns), decoders registered as `aleph-<name>` in `cudaq_qec`.

- [ ] **Step 0: Probe the `BatchDecoderResult` constructor signature on the box** (it is documented by prose only):

```bash
ssh root@openwebgui.splynx.com '/root/cqvenv/bin/python -W ignore -c "
import cudaq_qec as qec, numpy as np
print(qec.BatchDecoderResult.__init__.__doc__)
r = qec.BatchDecoderResult(result=np.zeros((2,3)), converged=np.ones(2,dtype=bool), opt_results=None, batch_opt_results=None)
print(r.result.shape, r.converged)
r0 = qec.BatchDecoderResult(result=np.zeros((0,3)), converged=np.zeros(0,dtype=bool), opt_results=None, batch_opt_results=None)
print(r0.result.shape)
"'
```

If keyword construction fails, use the positional order the docstring prints and note it in a comment in `cudaq.py`. If the empty batch is rejected, `decode_batch` must return `BatchDecoderResult(result=np.zeros((0, 0)), ...)` — the docstring says an empty batch yields `(0, 0)`.

- [ ] **Step 1: Write the failing tests** — `scripts/python/test_cudaq.py`:

```python
"""aleph.cudaq: aleph decoders registered as cudaq-qec decoders.

Skips unless cudaq_qec is importable (it is not in CI; run on the GPU box:
  /root/cqvenv/bin/python -m unittest scripts.python.test_cudaq -v).
"""
import unittest

import numpy as np

try:
    import cudaq_qec as cq
    import aleph.qec as aq
    import aleph.cudaq as ac
    HAVE = True
except ImportError:
    HAVE = False

try:
    import stim
    HAVE_STIM = True
except ImportError:
    HAVE_STIM = False

DEM = "error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1\n"
H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
O = np.array([[1, 0, 0]], dtype=np.uint8)
RATES = [0.1, 0.1, 0.1]


@unittest.skipUnless(HAVE, "needs cudaq_qec + aleph")
class TestRegistration(unittest.TestCase):
    def test_all_seven_resolve(self):
        self.assertEqual(sorted(ac.NAMES), sorted(aq.decoder_names()))
        for name in ac.NAMES:
            d = cq.get_decoder(f"aleph-{name}", H, O=O, error_rate_vec=RATES)
            self.assertEqual((d.get_block_size(), d.get_syndrome_size()), (3, 2), name)

    def test_decode_contract(self):
        for name in ac.NAMES:
            d = cq.get_decoder(f"aleph-{name}", H, O=O, error_rate_vec=RATES)
            r = d.decode([1.0, 0.0])
            self.assertIsInstance(r, cq.DecoderResult)
            self.assertEqual(len(r.result), 3, name)
            self.assertTrue(r.converged, name)
            ehat = (np.asarray(r.result) > 0.5).astype(np.uint8)
            np.testing.assert_array_equal((H @ ehat) % 2, [1, 0], name)
            self.assertEqual(((O @ ehat) % 2).tolist(), [1], name)  # D0 alone -> boundary edge with L0

    def test_decode_batch_contract(self):
        dets = np.array([[1, 0], [0, 0], [1, 1], [0, 1]], dtype=np.uint8)
        ref = aq.Decoder(aq.dem_from_matrices(H, O, RATES), "mwpm").decode_batch(dets.astype(bool))
        d = cq.get_decoder("aleph-mwpm", H, O=O, error_rate_vec=RATES)
        br = d.decode_batch(dets.astype(np.float64).tolist())
        self.assertIsInstance(br, cq.BatchDecoderResult)
        self.assertEqual(br.result.shape, (4, 3))
        self.assertEqual(br.converged.tolist(), [True] * 4)
        ehat = (br.result > 0.5).astype(np.uint8)
        np.testing.assert_array_equal((ehat @ O.T % 2).astype(bool), ref)

    def test_decode_batch_empty(self):
        d = cq.get_decoder("aleph-bp", H, O=O, error_rate_vec=RATES)
        br = d.decode_batch([])
        self.assertEqual(br.result.shape[0], 0)
        self.assertEqual(br.converged.shape, (0,))

    def test_soft_syndrome_thresholds(self):
        d = cq.get_decoder("aleph-mwpm", H, O=O, error_rate_vec=RATES)
        hard = np.asarray(d.decode([1.0, 0.0]).result)
        soft = np.asarray(d.decode([0.7, 0.3]).result)
        np.testing.assert_array_equal(hard, soft)

    def test_scalar_error_rate_and_missing_rates(self):
        d = cq.get_decoder("aleph-relay-bp", H, O=O, error_rate=0.1)
        self.assertEqual(len(d.decode([0.0, 1.0]).result), 3)
        with self.assertRaisesRegex(ValueError, "error_rate"):
            cq.get_decoder("aleph-relay-bp", H, O=O)

    def test_unknown_param_and_wrong_width(self):
        with self.assertRaisesRegex(ValueError, "osd_order"):
            cq.get_decoder("aleph-mwpm", H, O=O, error_rate_vec=RATES, osd_order=2)
        d = cq.get_decoder("aleph-mwpm", H, O=O, error_rate_vec=RATES)
        with self.assertRaisesRegex(ValueError, "2"):
            d.decode([1.0, 0.0, 0.0])

    def test_matching_decoder_rejects_hyperedge_column(self):
        H3 = np.array([[1], [1], [1]], dtype=np.uint8)
        for name in ("mwpm", "union-find", "union-find-weighted"):
            with self.assertRaisesRegex(ValueError, "graph-?like"):
                cq.get_decoder(f"aleph-{name}", H3, error_rate_vec=[0.1])
        cq.get_decoder("aleph-bp", H3, error_rate_vec=[0.1])  # BP is fine with hyperedges

    def test_scipy_sparse_h(self):
        import scipy.sparse as sp
        d = cq.get_decoder("aleph-union-find", sp.csr_matrix(H), O=sp.csr_matrix(O), error_rate_vec=RATES)
        self.assertEqual(d.get_block_size(), 3)


@unittest.skipUnless(HAVE and HAVE_STIM, "needs cudaq_qec + aleph + stim")
class TestStimDem(unittest.TestCase):
    def circuit(self):
        return stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3,
                                      after_clifford_depolarization=0.003, before_round_data_depolarization=0.003,
                                      before_measure_flip_probability=0.003, after_reset_flip_probability=0.003)

    def test_dem_string_path_for_bp_family(self):
        dem = self.circuit().detector_error_model(decompose_errors=True)
        d = cq.get_decoder("aleph-relay-bp-osd", str(dem))  # cudaq injects O and error_rate_vec
        self.assertEqual(d.get_syndrome_size(), dem.num_detectors)
        with self.assertRaisesRegex(ValueError, "graph-?like"):
            cq.get_decoder("aleph-mwpm", str(dem))  # cudaq keeps `^` parts as one column

    def test_dem_to_matrices_makes_mwpm_agree_with_aleph_qec(self):
        circ = self.circuit()
        dem = circ.detector_error_model(decompose_errors=True)
        Hm, Om, rates = ac.dem_to_matrices(dem)
        self.assertEqual(Hm.shape[0], dem.num_detectors)
        self.assertEqual(Om.shape[0], dem.num_observables)
        self.assertTrue((Hm.sum(axis=0) <= 2).all(), "columns must be graphlike after splitting ^")
        dets, obs = circ.compile_detector_sampler(seed=3).sample(500, separate_observables=True)
        d = cq.get_decoder("aleph-mwpm", Hm, O=Om, error_rate_vec=rates)
        br = d.decode_batch(dets.astype(np.float64).tolist())
        pred = ((br.result > 0.5).astype(np.uint8) @ Om.T % 2).astype(bool)
        ref = aq.Decoder(aq.DetectorErrorModel(dem), "mwpm").decode_batch(dets)
        # Same model, same matcher; equal up to genuine ties (a few % at most at p=0.003, d=3).
        self.assertGreater((pred == ref).all(axis=1).mean(), 0.97)
        np.testing.assert_array_equal(((br.result > 0.5).astype(np.uint8) @ Hm.T % 2).astype(bool), dets)
```

- [ ] **Step 2: Run to verify they fail (locally they skip; on the box they fail on import of `aleph.cudaq`)**

Local: `python -m unittest scripts.python.test_cudaq -v 2>&1 | tail -3` → `skipped`.

- [ ] **Step 3: Write the plugin** — `crates/aleph-py/python/aleph/cudaq.py`:

```python
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
"""
import cudaq_qec as _qec  # the only cudaq import in aleph; ImportError means "install the extra"
import numpy as np

import aleph.qec as _aq

NAMES = tuple(_aq.decoder_names())

__all__ = ["NAMES", "DECODERS", "dem_to_matrices"]


def _rates(n, error_rate_vec, error_rate):
    if error_rate_vec is not None:
        r = np.asarray(error_rate_vec, dtype=np.float64).ravel()
        if r.shape[0] != n:
            raise ValueError(f"error_rate_vec has {r.shape[0]} entries but H has {n} columns")
        return r
    if error_rate is not None:
        return np.full(n, float(error_rate))
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
            ehat, conv = self._inner.decode_batch_errors(_to_bits(syndrome, self._width))
            r = _qec.DecoderResult()
            r.converged = bool(conv[0])
            r.result = ehat[0].astype(np.float64).tolist()
            r.opt_results = None
            return r

        def decode_batch(self, syndromes):
            ehat, conv = self._inner.decode_batch_errors(_to_bits(syndromes, self._width))
            if ehat.shape[0] == 0:
                ehat = np.zeros((0, 0), dtype=np.float64)   # cudaq's documented empty-batch shape
            return _qec.BatchDecoderResult(result=ehat.astype(np.float64), converged=conv,
                                           opt_results=None, batch_opt_results=None)

    AlephDecoder.__name__ = AlephDecoder.__qualname__ = f"Aleph_{name.replace('-', '_')}"
    return AlephDecoder


DECODERS = {name: _make(name) for name in NAMES}


def dem_to_matrices(dem):
    """Stim DEM (``stim.DetectorErrorModel``, DEM text, or ``aleph.qec.DetectorErrorModel``) ->
    ``(H, O, error_rate_vec)`` with every ``^``-separated part of a mechanism as its own column,
    so matching decoders can take the result as ``H``. ``H`` is ``uint8 (detectors, E)``, ``O``
    is ``uint8 (observables, E)``, ``error_rate_vec`` is ``float64 (E,)``.
    """
    adem = dem if isinstance(dem, _aq.DetectorErrorModel) else _aq.DetectorErrorModel(dem)
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
        H[dets, j] ^= 1
        O[obs, j] ^= 1
        rates[j] = p
    return H, O, rates
```

Notes for the implementer: cudaq's `get_decoder` passes `H` through unchanged, so `H.shape` works for both dense and scipy inputs. `H[dets, j] ^= 1` with a repeated detector in one part cancels (Stim parity); `np.ix_` is not needed for a 1-D row index list with a scalar column.

- [ ] **Step 4: Extra in `pyproject.toml`** — under `[project.optional-dependencies]`:

```toml
# `aleph.cudaq`: aleph decoders registered inside NVIDIA CUDA-Q QEC. The meta-package picks
# the cu12/cu13 wheel; scipy is what cudaq hands us as a sparse H.
cudaq = ["cudaq-qec>=0.8,<0.9", "scipy>=1.10"]
```

- [ ] **Step 5: Build the wheel, ship it to the box, run the tests there**

```bash
maturin build --release -m crates/aleph-py/Cargo.toml --out dist 2>&1 | tail -2   # on the Mac this is an arm64 wheel — NOT installable on the box
```

The box is x86_64 Linux, so build **on the box** instead: the repo must be there. Use rsync of the working tree (excluding `target/`):

```bash
rsync -az --delete --exclude target --exclude .git --exclude dist /Users/ex/GitHub/aleph/ root@openwebgui.splynx.com:/root/aleph-f6/
ssh root@openwebgui.splynx.com 'export PATH=/root/.cargo/bin:$PATH; cd /root/aleph-f6/crates/aleph-py && /root/cqvenv/bin/pip install -q maturin scipy && VIRTUAL_ENV=/root/cqvenv /root/cqvenv/bin/maturin develop --release 2>&1 | tail -2 && cd /root/aleph-f6 && /root/cqvenv/bin/python -W ignore -m unittest scripts.python.test_cudaq -v 2>&1 | tail -25'
```

Expected: `OK` with all `TestRegistration` and `TestStimDem` tests passing. Fix and re-run until green; paste the final output into the PR body later.

- [ ] **Step 6: Local sanity (`import aleph` must not need cudaq) and commit**

```bash
python -c "import aleph, aleph.qec; print('ok')"
python -m unittest scripts.python.test_cudaq 2>&1 | tail -2   # skipped locally
git add crates/aleph-py/python/aleph/cudaq.py crates/aleph-py/pyproject.toml scripts/python/test_cudaq.py
git commit -m "[F6] aleph.cudaq: register aleph decoders in cudaq-qec

Seven decoders (aleph-mwpm ... aleph-relay-bp-osd) via @cudaq_qec.decoder,
taking H/O/error_rate_vec and returning per-column error estimates through
Decoder.decode_batch_errors. dem_to_matrices splits ^ parts so matching
decoders get a graph-like H. Extra: aleph-sim[cudaq]."
```

---

### Task 7: A/B harness + perf record

**Files:**
- Create: `scripts/python/ab_cudaq.py`
- Create: `docs/perf/f6-cudaq-plugin.md`

**Interfaces:**
- Consumes: `aleph.cudaq`, `aleph.qec.gross_code_dem`, `aleph.cudaq.dem_to_matrices`, cudaq decoders `nv-qldpc-decoder`, `pymatching`, `nv-fusion-decoder`.

- [ ] **Step 1: Write the harness** — `scripts/python/ab_cudaq.py`:

```python
"""A/B of aleph decoders vs NVIDIA's inside cudaq-qec: logical error rate + shots/s.

Run on the GPU box only:
  /root/cqvenv/bin/python scripts/python/ab_cudaq.py [--quick]
Prints Markdown tables; paste into docs/perf/f6-cudaq-plugin.md with the header block.

Workload G: gross [[144,12,12]] circuit-level DEM (aleph.qec.gross_code_dem), rounds=12,
  p in {0.001, 0.002, 0.003}; aleph-relay-bp(-osd) vs nv-qldpc-decoder in relay mode (+/-OSD).
Workload S: stim surface_code:rotated_memory_x d in {5, 9}, rounds=d, p=0.003, decomposed;
  aleph-mwpm / aleph-union-find-weighted vs cudaq pymatching vs nv-fusion-decoder.
Every decoder in a workload sees the identical (H, O, error_rate_vec) and the identical
syndrome/observable arrays. LER = shots with any wrong observable / shots, 95% Wilson CI.
"""
import argparse
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np
import stim

import cudaq_qec as cq
import aleph
import aleph.qec as aq
import aleph.cudaq as ac

QUICK = "--quick" in sys.argv
SEED = 20260924

# NVIDIA relay-BP mode per the 0.8.0 docs: min-sum with dynamic memory (bp_method=3), sequential
# relay composition (composition=1), sparse kernels; everything else at their defaults.
NV_RELAY = dict(bp_method=3, composition=1, use_sparsity=True, max_iterations=100,
                srelay_config=dict(pre_iter=60, num_sets=4, stopping_criterion="All"))
ALEPH_OSD = dict(osd_order=12)   # docs/perf/qec-q5-circuit-dem.md settings; relay defaults otherwise


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def sample(dem_text, shots, seed):
    s = stim.DetectorErrorModel(dem_text).compile_sampler(seed=seed)
    dets, obs, _ = s.sample(shots)
    return dets.astype(np.uint8), obs.astype(np.uint8)


def run_decoder(name, H, O, rates, dets, obs, params, threads=None):
    """(LER, lo, hi, shots/s, nonconverged) for cudaq decoder `name` on this batch."""
    env = None
    if threads is not None:
        return _run_in_child(name, H, O, rates, dets, obs, params, threads)
    d = cq.get_decoder(name, H, O=O, error_rate_vec=rates.tolist(), **params)
    rows = dets.astype(np.float64).tolist()
    d.decode_batch(rows[:200])                       # warm-up
    best = 0.0
    for _ in range(3):
        t = time.perf_counter()
        br = d.decode_batch(rows)
        best = max(best, len(rows) / (time.perf_counter() - t))
    ehat = (np.asarray(br.result) > 0.5).astype(np.uint8)
    pred = (ehat @ O.T) % 2
    wrong = int((pred != obs).any(axis=1).sum())
    nconv = int((~np.asarray(br.converged, dtype=bool)).sum())
    return (*wilson(wrong, len(rows)), best, nconv)


def _run_in_child(name, H, O, rates, dets, obs, params, threads):
    """aleph decoders size rayon's pool once per process: 1-thread numbers come from a child."""
    import json, tempfile
    with tempfile.TemporaryDirectory() as td:
        np.savez(f"{td}/in.npz", H=H, O=O, rates=rates, dets=dets, obs=obs)
        code = (
            "import json,sys,numpy as np,warnings; warnings.simplefilter('ignore');"
            "sys.path.insert(0, %r); import ab_cudaq as m; z=np.load(%r);"
            "print(json.dumps(m.run_decoder(%r, z['H'], z['O'], z['rates'], z['dets'], z['obs'], json.loads(%r))))"
            % (os.path.dirname(os.path.abspath(__file__)), f"{td}/in.npz", name, json.dumps(params))
        )
        out = subprocess.run([sys.executable, "-c", code], env={**os.environ, "RAYON_NUM_THREADS": str(threads)},
                             capture_output=True, text=True, check=True).stdout
        return tuple(json.loads(out.strip().splitlines()[-1]))


def row(workload, decoder, cfg, threads, shots, r):
    ler, lo, hi, rate, nconv = r
    return f"| {workload} | {decoder} | {cfg} | {threads} | {shots:,} | {ler:.2e} [{lo:.1e}, {hi:.1e}] | {nconv} | {rate:,.0f} |"


HEADER = "| workload | decoder | config | threads | shots | LER [95% CI] | non-conv | shots/s |\n|---|---|---|---:|---:|---|---:|---:|"


def workload_g():
    print("\n### Workload G — gross [[144,12,12]], rounds=12, circuit-level uniform p\n")
    print(HEADER)
    for p in ([0.003] if QUICK else [0.001, 0.002, 0.003]):
        dem = aq.gross_code_dem(12, p)
        H, O, rates = ac.dem_to_matrices(dem)
        shots = 2000 if QUICK else (100_000 if p <= 0.001 else 20_000)
        dets, obs = sample(dem.to_dem_string(), shots, SEED)
        w = f"gross p={p}"
        for cfg_name, name, params in [
            ("relay+OSD-12", "aleph-relay-bp-osd", ALEPH_OSD),
            ("relay, no OSD", "aleph-relay-bp", {}),
        ]:
            print(row(w, name, cfg_name, os.cpu_count(), shots, run_decoder(name, H, O, rates, dets, obs, params)))
            print(row(w, name, cfg_name, 1, shots, run_decoder(name, H, O, rates, dets, obs, params, threads=1)))
        for cfg_name, params in [
            ("relay+OSD-12", {**NV_RELAY, "use_osd": True, "osd_order": 12, "osd_method": 1}),
            ("relay, no OSD", {**NV_RELAY, "use_osd": False}),
        ]:
            try:
                print(row(w, "nv-qldpc-decoder", cfg_name, "GPU", shots, run_decoder("nv-qldpc-decoder", H, O, rates, dets, obs, params)))
            except Exception as e:  # report, never drop
                print(f"| {w} | nv-qldpc-decoder | {cfg_name} | GPU | {shots:,} | ERROR: {str(e)[:120]} | | |")


def surface(d, p=0.003):
    return stim.Circuit.generated("surface_code:rotated_memory_x", distance=d, rounds=d,
                                  after_clifford_depolarization=p, before_round_data_depolarization=p,
                                  before_measure_flip_probability=p, after_reset_flip_probability=p)


def workload_s():
    print("\n### Workload S — surface_code:rotated_memory_x, rounds=d, p=0.003, decomposed\n")
    print(HEADER)
    for d, shots in ([(5, 2000)] if QUICK else [(5, 20_000), (9, 5_000)]):
        circ = surface(d)
        dem = circ.detector_error_model(decompose_errors=True)
        H, O, rates = ac.dem_to_matrices(dem)
        dets, obs = circ.compile_detector_sampler(seed=SEED).sample(shots, separate_observables=True)
        dets, obs = dets.astype(np.uint8), obs.astype(np.uint8)
        w = f"surface d={d}"
        for name in ["aleph-mwpm", "aleph-union-find-weighted"]:
            print(row(w, name, "-", os.cpu_count(), shots, run_decoder(name, H, O, rates, dets, obs, {})))
            print(row(w, name, "-", 1, shots, run_decoder(name, H, O, rates, dets, obs, {}, threads=1)))
        for name, params in [("pymatching", {}), ("nv-fusion-decoder", {"num_threads": os.cpu_count()})]:
            try:
                print(row(w, name, "defaults", "1" if name == "pymatching" else os.cpu_count(), shots,
                          run_decoder(name, H, O, rates, dets, obs, params)))
            except Exception as e:
                print(f"| {w} | {name} | defaults | | {shots:,} | ERROR: {str(e)[:120]} | | |")


if __name__ == "__main__":
    import cudaq
    print(f"aleph {aleph.version()}, cudaq-qec {cq.__version__}, cudaq {cudaq.__version__}, stim {stim.__version__}, "
          f"numpy {np.__version__}, {platform.platform()}, {os.cpu_count()} CPUs, python {platform.python_version()}")
    print(subprocess.run(["bash", "-c", "uptime; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || cat /proc/driver/nvidia/version | head -1"],
                         capture_output=True, text=True).stdout.strip())
    workload_g()
    workload_s()
```

Before relying on `NV_RELAY`, confirm each key exists in `cq.decoder_param_schema("nv-qldpc-decoder")` / `("srelay_bp")` (Task-0 output of the spec lists them: `bp_method`, `composition`, `use_sparsity`, `max_iterations`, `use_osd`, `osd_method`, `osd_order`, `srelay_config{pre_iter,num_sets,stopping_criterion,stop_nconv}`). If NVIDIA's documented relay defaults differ from `pre_iter=60, num_sets=4`, use the documented ones and say so in the record.

- [ ] **Step 2: Quick run on an idle box**

```bash
rsync -az --exclude target --exclude .git /Users/ex/GitHub/aleph/ root@openwebgui.splynx.com:/root/aleph-f6/
ssh root@openwebgui.splynx.com 'uptime; pgrep -af "cargo|python|Runner" | grep -v pgrep; cd /root/aleph-f6 && /root/cqvenv/bin/python -W ignore scripts/python/ab_cudaq.py --quick 2>&1 | grep -v Warning'
```

Expected: both tables print, no `ERROR:` rows (if `nv-qldpc-decoder` rejects a key, fix the config and re-run). Sanity: aleph and NVIDIA LERs on the same batch should be the same order of magnitude; pymatching and aleph-mwpm LER should be nearly identical.

- [ ] **Step 3: Full run (box idle; ~30–60 min) and capture**

```bash
ssh root@openwebgui.splynx.com 'cd /root/aleph-f6 && nohup /root/cqvenv/bin/python -W ignore scripts/python/ab_cudaq.py > /root/aleph-f6/ab_cudaq.out 2>&1 &'
# later:
ssh root@openwebgui.splynx.com 'tail -50 /root/aleph-f6/ab_cudaq.out'
```

- [ ] **Step 4: Write `docs/perf/f6-cudaq-plugin.md`** with: purpose (one paragraph, from the spec §1), box + package versions (the harness header line), the exact command, the two tables verbatim, both parameter sets verbatim (`ALEPH_OSD`, relay defaults from `aleph.qec`, `NV_RELAY` + OSD keys), and a "Reading the numbers" section following the spec §8 honesty rules: relay parameterisations are not equivalent; CPU (20 cores / 1 thread) vs GPU throughput is stated, not ranked; any `ERROR`/non-converged cells discussed; MWPM vs pymatching should agree in LER within CI (if they don't, that is a bug — stop and investigate before publishing). Link the two follow-up issues (Task 8).

- [ ] **Step 5: Commit**

```bash
git add scripts/python/ab_cudaq.py docs/perf/f6-cudaq-plugin.md
git commit -m "[F6] bench: cudaq A/B harness + perf record

Gross qLDPC (aleph relay-BP(+OSD) vs nv-qldpc-decoder relay) and surface
(aleph-mwpm/UF vs pymatching vs nv-fusion-decoder) on identical inputs:
LER with Wilson CI and shots/s. Numbers and caveats in docs/perf/f6-cudaq-plugin.md."
```

---

### Task 8: Docs, changelog, follow-up issues, PR

**Files:**
- Modify: `crates/aleph-py/README.md` (new section before `## Links`)
- Modify: `CHANGELOG.md` (Unreleased → Added)
- Modify: `docs/qec/open-silicon-program.md` (tick F6)

- [ ] **Step 1: README section** — insert before `## Links`:

````markdown
## Using aleph decoders from CUDA-Q QEC

`aleph.cudaq` registers every aleph decoder inside NVIDIA's `cudaq_qec`, so a
CUDA-Q / NVQLink workflow can A/B them against `nv-qldpc-decoder`,
`nv-fusion-decoder` or `pymatching` without leaving its harness:

```bash
pip install "aleph-sim[cudaq]"      # pulls cudaq-qec (cu12/cu13 auto-selected) and scipy
```

```python
import numpy as np, stim, cudaq_qec as qec
import aleph.qec, aleph.cudaq                       # the import registers aleph-* decoders

circ = stim.Circuit.generated("surface_code:rotated_memory_x", distance=5, rounds=5,
                              after_clifford_depolarization=0.003)
H, O, rates = aleph.cudaq.dem_to_matrices(circ.detector_error_model(decompose_errors=True))
dets, obs = circ.compile_detector_sampler().sample(10_000, separate_observables=True)

dec = qec.get_decoder("aleph-mwpm", H, O=O, error_rate_vec=rates)    # or "aleph-relay-bp-osd", ...
res = dec.decode_batch(dets.astype(float).tolist())
pred = ((res.result > 0.5).astype(np.uint8) @ O.T) % 2
print("logical error rate:", (pred != obs).any(axis=1).mean())
```

Names: `aleph-mwpm`, `aleph-union-find`, `aleph-union-find-weighted`, `aleph-bp`,
`aleph-bp-osd`, `aleph-relay-bp`, `aleph-relay-bp-osd`. Keyword parameters are
those of `aleph.qec.Decoder` (`legs`, `alpha`, `gamma_min`, `gamma_max`, `seed`,
`osd_order`, `max_iter`); priors come from `error_rate_vec` (cudaq fills it in
when you pass a DEM string) or a scalar `error_rate`. Results are per-column
error estimates like cudaq's own decoders, so `O @ ê` gives the observable flips.

Matching decoders need a graph-like `H` (≤ 2 ones per column). cudaq's DEM
parser keeps a `^`-decomposed hyperedge as one column, so for `aleph-mwpm` /
`aleph-union-find*` pass `dem_to_matrices(dem)` rather than the DEM string
(the BP family takes either). For the gross [[144,12,12]] code, which cudaq has
no built-in for, `aleph.qec.gross_code_dem(rounds, p)` gives the circuit-level
model. A/B numbers against NVIDIA's decoders: `docs/perf/f6-cudaq-plugin.md`.
````

- [ ] **Step 2: CHANGELOG** under `## [Unreleased]` add `### Added` (before the existing `### Changed`):

```markdown
### Added

- **`aleph.cudaq` — aleph decoders inside CUDA-Q QEC.** `pip install
  "aleph-sim[cudaq]"` + `import aleph.cudaq` registers all seven decoders as
  `cudaq_qec` decoders (`aleph-mwpm` … `aleph-relay-bp-osd`) with cudaq's
  `H`/`O`/`error_rate_vec` contract and per-column error output; `dem_to_matrices`
  splits `^` parts for the matching decoders. Supporting API:
  `aleph.qec.dem_from_matrices`, `Decoder.decode_batch_errors`,
  `Decoder.num_errors`, `aleph.qec.gross_code_dem`; in Rust,
  `DetectorErrorModel::from_check_matrices` and `decode_errors` on every
  decoder (Sparse Blossom retraces matched pairs to edges). A/B vs
  `nv-qldpc-decoder` / `pymatching` / `nv-fusion-decoder`:
  `docs/perf/f6-cudaq-plugin.md`. Rust-side guard for > 64 observables
  (closes #512).
```

- [ ] **Step 3: Tick F6 in `docs/qec/open-silicon-program.md`** — change `- [ ] **Task F6: ...` to `- [x] **Task F6 (done 2026-09-24, PR #<n>): ...` keeping the text, and append one sentence: "Shipped as the Python plugin `aleph.cudaq` (`docs/perf/f6-cudaq-plugin.md`); the native `.so` plugin for the realtime path is issue #<follow-up>."

- [ ] **Step 4: File the two follow-up issues** (fill the PR/issue numbers back into Step 3 and the perf record):

```bash
gh issue create --title "[F6b] Native C++ cudaq-qec decoder plugin (realtime/NVQLink path)" --label "area:qec,area:decoder,type:feature,priority:medium" --body "$(cat <<'EOF'
`aleph.cudaq` (F6) registers aleph's decoders in cudaq-qec as **Python** decoders, which cudaq only uses offline. The realtime / NVQLink path loads native plugins: a `.so` in `<dir of libcudaq-qec-decoders.so>/decoder-plugins/` implementing `cudaq::qec::decoder` (`CUDAQ_EXT_PT_REGISTER_TYPE`).

Blocked on cudaqx's plugin ABI settling: the 0.8.0 wheel exports one `extension_point<decoder, ...>::get_registry()` signature and `main` already uses a different `decoder_init` + `std::optional<decode_result_type>` creator; the wheel ships no headers, so a plugin needs a source checkout at the exact tag (`cmake -S libs/qec -DCUDAQ_QEC_DECODERS_ONLY=ON`).

Plan when unblocked: Rust C ABI over `DetectorErrorModel::from_check_matrices` + `AnyDecoder::decode_errors` (both landed in F6), thin C++ shim class, build against the tagged headers, install into the wheel's `decoder-plugins/`, verify `get_decoder("aleph-relay-bp", H)` from C++ and the realtime config path.
EOF
)"
gh issue create --title "[Q5] f64 RelayBpDecoder: explicit iters_per_leg / early-exit knob" --label "area:qec,area:decoder,type:feature,priority:low" --body "$(cat <<'EOF'
`RelayBpDecoder::with_params` fixes iterations per leg at `max(100/legs, 8)`; only `FixedRelayBp::with_budget` takes an explicit budget and early-exit flag. For A/B against other relay-BP implementations (cudaq's `nv-qldpc-decoder` `max_iterations`/`srelay_config`, the ASIC's 6×10 schedule) the f64 decoder should expose `iters_per_leg` and `early_exit` too, threaded through `aleph.qec.Decoder("relay-bp", ...)` and `aleph.cudaq`. Surfaced by `docs/perf/f6-cudaq-plugin.md`.
EOF
)"
```

- [ ] **Step 5: Full local verification, push, PR**

```bash
cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace 2>&1 | tail -5
cargo clippy -p aleph-py --features python --all-targets -- -D warnings
git add crates/aleph-py/README.md CHANGELOG.md docs/qec/open-silicon-program.md
git commit -m "[F6] docs: aleph.cudaq README section, changelog, program tick"
git push -u origin f6-cudaq-plugin
gh pr create --title "[F6] aleph.cudaq: aleph decoders inside CUDA-Q QEC" --body "$(cat <<'EOF'
Closes #512

## Summary
<spec §1 in three sentences; list the seven names; link spec + plan + perf record>

## What's new
- Rust: `DetectorErrorModel::from_check_matrices`, `decode_errors` on MWPM (Sparse Blossom matched-pair retrace), UF, BP, BP+OSD, relay-BP, relay-BP+OSD; `MatchingEdge::column`.
- Python: `aleph.qec.dem_from_matrices`, `Decoder.decode_batch_errors`, `Decoder.num_errors`, `aleph.qec.gross_code_dem`, new module `aleph.cudaq`, extra `aleph-sim[cudaq]`.

## Tests
- `cargo test --workspace` (paste tail), incl. `decode_errors_invariants_on_phenomenological_shots` (release) and the 3000-case proptest.
- `scripts/python/test_qec.py` locally; `scripts/python/test_cudaq.py` on the GPU box (paste output — it skips in CI).

## A/B (docs/perf/f6-cudaq-plugin.md)
<paste the two tables + the one-paragraph reading>

## Follow-ups
- #<F6b> native C++ plugin; #<iters> relay iteration knob.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01MiLwGfRbAMYb6vJXJDq98t
EOF
)"
```

Then update `docs/qec/open-silicon-program.md` and the perf record with the real PR/issue numbers in one more small commit.

---

## Self-review notes (done while writing)

- **Spec coverage:** §4.1 → T1; §4.2 table → T2 (UF), T3 (MWPM), T4 (BP family), T5 (`AnyDecoder::decode_errors`); §4.3 → T3; §4.4 + §5.3 → T5; §5.1/5.2 → T5; §6 → T6; §7 → T1–T6 tests; §8 → T7; §9 → T8; §10 → T8 issues; #512 closed by T1 + T8 PR body.
- **Type consistency:** `decode_errors` returns `Vec<u8>` for MWPM/UF and `(Vec<u8>, bool)` for the BP family; `AnyDecoder::decode_errors` normalises to `(Vec<u8>, bool)` (T5). `MatchingEdge::column: u32`, `MatchingGraph::num_columns() -> usize`, `CompiledGraph::edge_cols(u) -> &[u32]`, `boundary_col(u) -> u32` used consistently in T3.
- **Known judgement calls for the implementer:** `with_state` closure ownership (T3 step 6 has the fallback); `BatchDecoderResult` constructor form (T6 step 0 probes it); NVIDIA relay defaults (T7 step 1 says to prefer the documented ones).
