# QEC decoders in Python + v0.3.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make aleph's DEM decoders (MWPM, Union-Find, BP, BP-OSD, relay-BP, relay-BP-OSD) usable from Python and sinter on real stim DEMs, and ship it as v0.3.0.

**Architecture:** Complete the Rust DEM parser (`repeat`, `shift_detectors`, preserved `^` decomposition) so stim's DEMs load unmodified. Add a GIL-free batch-decode core in `aleph-py` (plain Rust, unit-tested without Python), a thin pyo3 layer (`aleph._native.qec`), and a pure-Python package (`aleph/__init__.py`, `aleph/qec.py`, `aleph/sinter.py`) built by maturin's mixed layout.

**Tech Stack:** Rust 1.89, pyo3 0.22 (abi3-py310), rust-numpy 0.22, rayon, maturin ≥1.5; Python: numpy, stim, sinter, pymatching (tests), unittest.

**Spec:** `docs/superpowers/specs/2026-09-22-qec-python-v0.3-design.md`

## Global Constraints

- Branch `qec-python-v0.3` in the main checkout `/Users/ex/GitHub/aleph`. **No git worktrees.**
- Library code: no `unwrap()`/`expect()`/`panic!` on input; errors via `aleph_qec::Error` (thiserror) or `PyValueError`.
- Any float gate on user input (`alpha`, `gamma_min`, `gamma_max`) must reject non-finite values explicitly before comparisons (CLAUDE.md NaN rule / ADR 0006).
- Python floor: **3.10** (`abi3-py310`). `import aleph` and `import aleph.qec` must never import stim or sinter.
- Existing `scripts/python/test_aleph.py` must pass unchanged.
- Gates before every commit: `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo +nightly clippy --workspace --all-targets -- -D warnings` (CI runs beta lints).
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01DGzx9Sw2xqbdXTVFRh9shN
  ```
- **Never** push a `v*` tag or publish to PyPI without explicit user confirmation.

**Python environment for Tasks 3–7** (local system Python is 3.9, too old). Created once in Task 3:

```bash
VENV=/private/tmp/claude-501/-Users-ex-GitHub-aleph/1325061f-e0ea-48c8-9cee-b160dd345f2f/scratchpad/venv310
```
Build + install the extension into it: `source $VENV/bin/activate && maturin develop --release -m crates/aleph-py/Cargo.toml`.

---

## File Structure

| File | Responsibility |
|---|---|
| `crates/aleph-qec/src/dem.rs` (modify) | Parser: block structure, `repeat`, `shift_detectors`, `^` components; emitter round-trip |
| `crates/aleph-qec/src/error.rs` (modify) | Drop `UnsupportedDem` |
| `crates/aleph-qec/src/matching.rs` (modify) | One matching edge per `^` component |
| `crates/aleph-qec/src/builder.rs` (modify) | Struct literal gets `components` |
| `crates/aleph-py/src/qec_core.rs` (create) | No-pyo3 core: `DecoderParams`, `AnyDecoder`, `build_decoder`, `decode_rows`, `decode_packed` |
| `crates/aleph-py/src/qec.rs` (create) | pyo3 classes `DetectorErrorModel`, `Decoder`, fn `decoder_names` |
| `crates/aleph-py/src/lib.rs` (modify) | Module rename to `_native`, register `qec` submodule |
| `crates/aleph-py/python/aleph/{__init__,qec,sinter}.py` (create) | Python package surface |
| `crates/aleph-py/{Cargo.toml,pyproject.toml}` (modify) | deps, abi3-py310, mixed layout, extras |
| `scripts/python/test_qec.py` (create) | Python differential, oracle, sinter, pickle tests |
| `scripts/python/bench_qec.py` (create) | Throughput table for README |
| `.github/workflows/{ci,release}.yml` (modify) | Python 3.10+3.12 matrix, stim deps, QEC smoke |
| `Cargo.toml`, `CHANGELOG.md`, `README.md`, `crates/aleph-py/README.md` | v0.3.0 |

---

### Task 1: DEM parser — `repeat` blocks and `shift_detectors`

**Files:**
- Modify: `crates/aleph-qec/src/dem.rs` (whole `parse`, module docs, tests)
- Modify: `crates/aleph-qec/src/error.rs` (remove `UnsupportedDem`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `DetectorErrorModel::parse(&str) -> Result<DetectorErrorModel>` accepting nested `repeat N { … }` and `shift_detectors[(coords)] k`. Signature unchanged.

- [ ] **Step 1: Write the failing tests** — in `dem.rs` `mod tests`, replace `rejects_unsupported_repeat` with:

```rust
    #[test]
    fn repeat_unrolls_with_shift() {
        let text = "\
error(0.1) D0
repeat 3 {
    error(0.2) D0 D1
    shift_detectors 1
}
error(0.3) D0 L0
";
        let m = DetectorErrorModel::parse(text).unwrap();
        let got: Vec<(f64, Vec<u32>, Vec<u32>)> =
            m.errors.iter().map(|e| (e.prob, e.dets.clone(), e.obs.clone())).collect();
        assert_eq!(
            got,
            vec![
                (0.1, vec![0], vec![]),
                (0.2, vec![0, 1], vec![]),
                (0.2, vec![1, 2], vec![]),
                (0.2, vec![2, 3], vec![]),
                (0.3, vec![3], vec![0]), // offset persists after the block
            ]
        );
        assert_eq!(m.detectors, 4);
        assert_eq!(m.observables, 1);
    }

    #[test]
    fn nested_repeat_and_zero_count() {
        let text = "\
repeat 2 {
    repeat 2 {
        error(0.1) D0
        shift_detectors(0, 0, 1) 1
    }
}
repeat 0 {
    error(0.5) D99
}
";
        let m = DetectorErrorModel::parse(text).unwrap();
        let dets: Vec<Vec<u32>> = m.errors.iter().map(|e| e.dets.clone()).collect();
        assert_eq!(dets, vec![vec![0], vec![1], vec![2], vec![3]]);
        assert_eq!(m.detectors, 4); // D99 never executed
    }

    #[test]
    fn detector_declaration_is_shifted() {
        let m = DetectorErrorModel::parse("shift_detectors 5\ndetector(1, 2) D2\n").unwrap();
        assert_eq!(m.detectors, 8);
    }

    #[test]
    fn repeat_errors_are_reported() {
        for bad in [
            "repeat 2 {\nerror(0.1) D0\n",   // unclosed
            "}\n",                           // unmatched close
            "repeat x {\n}\n",               // bad count
            "repeat 2\n}\n",                 // missing brace
            "shift_detectors q\n",           // bad shift
            "detector_separator 1\n",        // unsupported instruction
        ] {
            assert!(
                matches!(DetectorErrorModel::parse(bad), Err(Error::DemParse { .. })),
                "should reject: {bad:?}"
            );
        }
    }

    #[test]
    fn shifted_index_overflow_is_an_error() {
        let text = format!("shift_detectors {}\nerror(0.1) D1\n", u32::MAX);
        assert!(matches!(
            DetectorErrorModel::parse(&text),
            Err(Error::DemParse { .. })
        ));
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p aleph-qec --lib dem::tests`
Expected: the five new tests FAIL (repeat currently returns `UnsupportedDem`).

- [ ] **Step 3: Implement.** Rewrite `parse` as a two-phase parse → execute. Replace the body of `DetectorErrorModel::parse` and add private types/functions below `split_paren_arg`:

```rust
    pub fn parse(text: &str) -> Result<Self> {
        let mut lines = text
            .lines()
            .enumerate()
            .map(|(i, raw)| (i + 1, strip_comment(raw).trim()))
            .filter(|(_, l)| !l.is_empty());
        let program = parse_block(&mut lines, None)?;
        let mut st = ExecState::default();
        exec_block(&program, &mut st)?;
        Ok(DetectorErrorModel {
            detectors: (st.max_det + 1) as usize,
            observables: (st.max_obs + 1) as usize,
            errors: st.errors,
        })
    }
```

```rust
/// One parsed DEM instruction; `repeat` bodies are kept as a tree and unrolled by [`exec_block`].
enum Instr {
    /// `error(p) …` — targets relative to the current detector offset; one entry per `^` part.
    Error { line: usize, prob: f64, parts: Vec<(Vec<u32>, Vec<u32>)> },
    /// `detector(…) D…` — declares detectors (coords ignored).
    Detector { line: usize, dets: Vec<u32> },
    /// `logical_observable L…`.
    Observable(Vec<u32>),
    /// `shift_detectors(…) k`.
    Shift(u64),
    /// `repeat n { … }`.
    Repeat(u64, Vec<Instr>),
}

#[derive(Default)]
struct ExecState {
    offset: u64,
    max_det: i64,
    max_obs: i64,
    errors: Vec<DemError>,
}

/// Parse instructions until the matching `}` (when `open` is `Some(line of the repeat)`) or EOF.
fn parse_block<'a>(
    lines: &mut impl Iterator<Item = (usize, &'a str)>,
    open: Option<usize>,
) -> Result<Vec<Instr>> {
    let mut out = Vec::new();
    while let Some((line_no, line)) = lines.next() {
        if line == "}" {
            return match open {
                Some(_) => Ok(out),
                None => Err(Error::DemParse { line: line_no, msg: "unmatched `}`".into() }),
            };
        }
        out.push(parse_instr(line_no, line, lines)?);
    }
    match open {
        Some(l) => Err(Error::DemParse { line: l, msg: "unclosed `repeat` block".into() }),
        None => Ok(out),
    }
}

fn parse_instr<'a>(
    line_no: usize,
    line: &str,
    lines: &mut impl Iterator<Item = (usize, &'a str)>,
) -> Result<Instr> {
    let bad = |msg: String| Error::DemParse { line: line_no, msg };
    // `detector_separator` must be checked before the `detector` prefix match.
    if line.starts_with("detector_separator") {
        return Err(bad("`detector_separator` is not supported".into()));
    }
    if let Some(rest) = line.strip_prefix("repeat") {
        let rest = rest.trim();
        let body = rest
            .strip_suffix('{')
            .ok_or_else(|| bad("`repeat` must end with `{`".into()))?;
        let n = body
            .trim()
            .parse::<u64>()
            .map_err(|e| bad(format!("invalid repeat count `{}`: {e}", body.trim())))?;
        return Ok(Instr::Repeat(n, parse_block(lines, Some(line_no))?));
    }
    if let Some(rest) = line.strip_prefix("shift_detectors") {
        let (_coords, k) = split_paren_arg(rest);
        let k = k
            .trim()
            .parse::<u64>()
            .map_err(|e| bad(format!("invalid shift `{}`: {e}", k.trim())))?;
        return Ok(Instr::Shift(k));
    }
    if let Some(rest) = line.strip_prefix("error") {
        let (arg, targets) = split_paren_arg(rest);
        let prob_str = arg.ok_or_else(|| bad("`error` requires a probability in parentheses".into()))?;
        let prob = prob_str
            .trim()
            .parse::<f64>()
            .map_err(|e| bad(format!("invalid probability `{prob_str}`: {e}")))?;
        let mut parts = vec![(Vec::new(), Vec::new())];
        for tok in targets.split_whitespace() {
            if tok == "^" {
                parts.push((Vec::new(), Vec::new()));
                continue;
            }
            // `parts` is never empty: it starts with one entry and only grows.
            let last = parts.len() - 1;
            match parse_target(tok, line_no)? {
                Target::Det(d) => parts[last].0.push(d),
                Target::Obs(o) => parts[last].1.push(o),
            }
        }
        return Ok(Instr::Error { line: line_no, prob, parts });
    }
    if let Some(rest) = line.strip_prefix("detector") {
        let (_coords, targets) = split_paren_arg(rest);
        let mut dets = Vec::new();
        for tok in targets.split_whitespace() {
            if let Target::Det(d) = parse_target(tok, line_no)? {
                dets.push(d);
            }
        }
        return Ok(Instr::Detector { line: line_no, dets });
    }
    if let Some(rest) = line.strip_prefix("logical_observable") {
        let mut obs = Vec::new();
        for tok in rest.split_whitespace() {
            if let Target::Obs(o) = parse_target(tok, line_no)? {
                obs.push(o);
            }
        }
        return Ok(Instr::Observable(obs));
    }
    Err(bad(format!("unknown instruction: `{line}`")))
}

/// Apply the running detector offset, rejecting indices that leave `u32`.
fn shifted(d: u32, offset: u64, line: usize) -> Result<u32> {
    u32::try_from(d as u64 + offset).map_err(|_| Error::DemParse {
        line,
        msg: format!("detector D{d} shifted by {offset} exceeds u32"),
    })
}

fn exec_block(block: &[Instr], st: &mut ExecState) -> Result<()> {
    for ins in block {
        match ins {
            Instr::Error { line, prob, parts } => {
                let mut abs = Vec::with_capacity(parts.len());
                for (ds, os) in parts {
                    let mut dets = Vec::with_capacity(ds.len());
                    for &d in ds {
                        let d = shifted(d, st.offset, *line)?;
                        st.max_det = st.max_det.max(d as i64);
                        dets.push(d);
                    }
                    for &o in os {
                        st.max_obs = st.max_obs.max(o as i64);
                    }
                    abs.push((dets, os.clone()));
                }
                st.errors.push(DemError::from_parts(*prob, abs));
            }
            Instr::Detector { line, dets } => {
                for &d in dets {
                    let d = shifted(d, st.offset, *line)?;
                    st.max_det = st.max_det.max(d as i64);
                }
            }
            Instr::Observable(obs) => {
                for &o in obs {
                    st.max_obs = st.max_obs.max(o as i64);
                }
            }
            Instr::Shift(k) => st.offset += k,
            Instr::Repeat(n, body) => {
                for _ in 0..*n {
                    exec_block(body, st)?;
                }
            }
        }
    }
    Ok(())
}
```

`ExecState::default()` must start `max_det`/`max_obs` at `-1`, so implement `Default` by hand instead of deriving:

```rust
impl Default for ExecState {
    fn default() -> Self {
        ExecState { offset: 0, max_det: -1, max_obs: -1, errors: Vec::new() }
    }
}
```

For this task only, add a temporary merging constructor in `impl DemError` (Task 2 replaces its body to keep the parts):

```rust
    /// Build from `^`-separated parts (Task 1: parts are merged).
    fn from_parts(prob: f64, parts: Vec<(Vec<u32>, Vec<u32>)>) -> Self {
        let (mut dets, mut obs) = (Vec::new(), Vec::new());
        for (d, o) in parts {
            dets.extend(d);
            obs.extend(o);
        }
        DemError::new(prob, dets, obs)
    }
```

Delete the `UnsupportedDem` variant from `error.rs`. Update the module doc in `dem.rs`: replace the "Q0-01 subset" paragraph with: "Supported instructions: `error` (with `^` separable components), `detector`, `logical_observable`, `shift_detectors`, and (nested) `repeat` blocks, which are unrolled into a flat model. `detector_separator` is rejected." and update the `# Errors` doc on `parse` to name only `Error::DemParse`.

- [ ] **Step 4: Run tests**

Run: `cargo test -p aleph-qec --lib dem::tests && cargo test -p aleph-qec`
Expected: all PASS (existing `accepts_separable_component_marker` still passes because parts are merged).

- [ ] **Step 5: Lint + commit**

```bash
cargo fmt && cargo clippy --workspace --all-targets -- -D warnings && cargo +nightly clippy --workspace --all-targets -- -D warnings
git add crates/aleph-qec/src/dem.rs crates/aleph-qec/src/error.rs
git commit -m "[qec-py] DEM parser: unroll repeat blocks and shift_detectors

stim emits repeat/shift_detectors for any multi-round circuit; rejecting
them made every real stim DEM unusable. Two-phase parse (tree, then
execute with a running detector offset) keeps nesting and error lines.

<attribution lines>"
```

---

### Task 2: Preserve `^` decomposition; matching uses components

**Files:**
- Modify: `crates/aleph-qec/src/dem.rs` (`DemError`, `from_parts`, `to_dem_string`, tests, proptest)
- Modify: `crates/aleph-qec/src/matching.rs:89-150` (`from_dem`) + its test literals (lines ~333–372)
- Modify: `crates/aleph-qec/src/builder.rs:179`

**Interfaces:**
- Consumes: `DemError::from_parts` from Task 1.
- Produces: `DemError { prob, dets, obs, components: Vec<(Vec<u32>, Vec<u32>)> }`; `DemError::with_components(prob: f64, parts: Vec<(Vec<u32>, Vec<u32>)>) -> DemError` (pub); `MatchingGraph::from_dem` accepts decomposed hyperedges.

- [ ] **Step 1: Failing tests.** In `dem.rs` tests, replace `accepts_separable_component_marker` with:

```rust
    #[test]
    fn separable_components_are_preserved() {
        let m = DetectorErrorModel::parse("error(0.1) D2 D0 ^ D1 L0\n").expect("parse");
        let e = &m.errors[0];
        assert_eq!(e.dets, vec![0, 1, 2]);
        assert_eq!(e.obs, vec![0]);
        assert_eq!(e.components, vec![(vec![0, 2], vec![]), (vec![1], vec![0])]);
        assert_eq!(
            *e,
            DemError::with_components(0.1, vec![(vec![2, 0], vec![]), (vec![1], vec![0])])
        );
        let text = m.to_dem_string();
        assert!(text.contains(" ^ "), "{text}");
        assert_eq!(DetectorErrorModel::parse(&text).unwrap(), m);
    }

    #[test]
    fn single_part_has_no_components() {
        let e = DemError::with_components(0.1, vec![(vec![1, 0], vec![])]);
        assert_eq!(e, DemError::new(0.1, vec![0, 1], vec![]));
        assert!(e.components.is_empty());
    }
```

Change `arb_error` so the proptest covers components:

```rust
    fn arb_error(detectors: usize, observables: usize) -> impl Strategy<Value = DemError> {
        let part = move || {
            let dets = if detectors == 0 {
                Just(Vec::new()).boxed()
            } else {
                prop::collection::vec(0u32..detectors as u32, 0..3).boxed()
            };
            let obs = if observables == 0 {
                Just(Vec::new()).boxed()
            } else {
                prop::collection::vec(0u32..observables as u32, 0..2).boxed()
            };
            (dets, obs)
        };
        (0.0001f64..0.5, prop::collection::vec(part(), 1..4))
            .prop_map(|(p, parts)| DemError::with_components(p, parts))
    }
```

In `matching.rs` tests add:

```rust
    #[test]
    fn decomposed_hyperedge_becomes_component_edges() {
        let dem = DetectorErrorModel::parse("error(0.1) D0 D1 ^ D2 D3 L0\n").unwrap();
        let g = MatchingGraph::from_dem(&dem).unwrap();
        let mut ends: Vec<(NodeId, NodeId, Vec<u32>)> =
            g.edges().iter().map(|e| (e.a, e.b, e.observables.clone())).collect();
        ends.sort();
        assert_eq!(ends, vec![(0, 1, vec![]), (2, 3, vec![0])]);
        assert!(g.edges().iter().all(|e| (e.prob - 0.1).abs() < 1e-12));
    }

    #[test]
    fn component_with_three_detectors_is_still_rejected() {
        let dem = DetectorErrorModel::parse("error(0.1) D0 D1 D2 ^ D3\n").unwrap();
        assert!(matches!(
            MatchingGraph::from_dem(&dem),
            Err(Error::NonGraphlike { dets: 3 })
        ));
    }
```
(add `use crate::Error;` to that test module if absent).

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p aleph-qec --lib`
Expected: compile error (`components` / `with_components` missing).

- [ ] **Step 3: Implement.** In `dem.rs`:

```rust
pub struct DemError {
    /// Probability of this mechanism firing, in `[0, 1]`.
    pub prob: f64,
    /// Detector indices flipped by this mechanism (sorted ascending; all `^` parts merged).
    pub dets: Vec<u32>,
    /// Logical observable indices flipped by this mechanism (sorted ascending; parts merged).
    pub obs: Vec<u32>,
    /// The `^`-separated `(dets, obs)` parts as written, each sorted; **empty** when the
    /// mechanism had a single part. Matching decoders build one edge per part (stim's
    /// `decompose_errors` hint); BP-family decoders use the merged `dets`/`obs`.
    pub components: Vec<(Vec<u32>, Vec<u32>)>,
}

impl DemError {
    /// Build a single-part mechanism, normalising target order so equality is order-independent.
    pub fn new(prob: f64, mut dets: Vec<u32>, mut obs: Vec<u32>) -> Self {
        dets.sort_unstable();
        obs.sort_unstable();
        DemError { prob, dets, obs, components: Vec::new() }
    }

    /// Build a mechanism from `^`-separated parts. One part is identical to [`DemError::new`].
    pub fn with_components(prob: f64, parts: Vec<(Vec<u32>, Vec<u32>)>) -> Self {
        if parts.len() <= 1 {
            let (d, o) = parts.into_iter().next().unwrap_or_default();
            return DemError::new(prob, d, o);
        }
        let mut merged = DemError::new(prob, Vec::new(), Vec::new());
        let mut components = Vec::with_capacity(parts.len());
        for (mut d, mut o) in parts {
            d.sort_unstable();
            o.sort_unstable();
            merged.dets.extend_from_slice(&d);
            merged.obs.extend_from_slice(&o);
            components.push((d, o));
        }
        merged.dets.sort_unstable();
        merged.obs.sort_unstable();
        merged.components = components;
        merged
    }

    fn from_parts(prob: f64, parts: Vec<(Vec<u32>, Vec<u32>)>) -> Self {
        Self::with_components(prob, parts)
    }
}
```

(Or delete `from_parts` and call `with_components` directly from `exec_block`. Prefer deleting.)

In `to_dem_string`, replace the per-error target loops with:

```rust
            let single = [(e.dets.clone(), e.obs.clone())];
            let parts: &[(Vec<u32>, Vec<u32>)] =
                if e.components.is_empty() { &single } else { &e.components };
            for (k, (ds, os)) in parts.iter().enumerate() {
                if k > 0 {
                    out.push_str(" ^");
                }
                for &d in ds {
                    out.push_str(" D");
                    out.push_str(&d.to_string());
                    used_det = used_det.max(d as usize + 1);
                }
                for &o in os {
                    out.push_str(" L");
                    out.push_str(&o.to_string());
                    used_obs = used_obs.max(o as usize + 1);
                }
            }
```

In `matching.rs` `from_dem`, replace the body of the `for e in &dem.errors` loop after the probability check with a loop over parts:

```rust
            let single = [(e.dets.clone(), e.obs.clone())];
            let parts: &[(Vec<u32>, Vec<u32>)] =
                if e.components.is_empty() { &single } else { &e.components };
            // A decomposed (`^`) mechanism contributes one edge per part, each at the
            // mechanism's probability — the decomposition stim emits for matching decoders.
            for (pd, po) in parts {
                let dets = odd_parity(pd);
                let obs = odd_parity(po);
                let (a, b) = match dets.len() {
                    0 => continue,
                    1 => (dets[0] as NodeId, boundary),
                    2 => (dets[0] as NodeId, dets[1] as NodeId),
                    n => return Err(Error::NonGraphlike { dets: n }),
                };
                debug_assert!(a < b, "endpoints must be distinct and ordered");
                let key = (a, b, obs);
                if let Some(p) = merged.get_mut(&key) {
                    *p = xor_combine(*p, e.prob);
                } else {
                    merged.insert(key.clone(), e.prob);
                    order.push(key);
                }
            }
```
(Keep the existing comments about dropping 0-detector parts; the `continue` now skips only that part.) Update the `from_dem` doc: "Mechanisms with `^` components contribute one edge per component."

Add `components: Vec::new()` to the struct literals at `builder.rs:179` and the five in `matching.rs` tests. Then `grep -rn "DemError {" crates` must show only the struct definition and constructors.

- [ ] **Step 4: Run tests**

Run: `cargo test -p aleph-qec && cargo test --workspace`
Expected: PASS. (Workspace test catches other crates constructing `DemError` by literal.)

- [ ] **Step 5: Lint + commit**

```bash
cargo fmt && cargo clippy --workspace --all-targets -- -D warnings && cargo +nightly clippy --workspace --all-targets -- -D warnings
git add -u crates/aleph-qec
git commit -m "[qec-py] DEM: keep ^ decomposition; matching builds one edge per component

stim's decompose_errors=True DEMs contain hyperedges split with ^.
Merging the parts made MWPM and Union-Find reject every circuit-level
DEM. Parts are kept alongside the merged view (BP decoders unchanged).

<attribution lines>"
```

---

### Task 3: Python package restructure (mixed layout, `_native`, abi3-py310)

**Files:**
- Modify: `crates/aleph-py/Cargo.toml`, `crates/aleph-py/pyproject.toml`, `crates/aleph-py/src/lib.rs`
- Create: `crates/aleph-py/python/aleph/__init__.py`

**Interfaces:**
- Produces: extension module `aleph._native`; `aleph` re-exports every existing symbol. Cargo deps `aleph-qec` and `rayon` available to `aleph-py` (non-optional).

- [ ] **Step 1: Environment**

```bash
brew list uv >/dev/null 2>&1 || brew install uv
VENV=/private/tmp/claude-501/-Users-ex-GitHub-aleph/1325061f-e0ea-48c8-9cee-b160dd345f2f/scratchpad/venv310
uv venv "$VENV" -p 3.10
uv pip install -p "$VENV" "maturin>=1.5,<2.0" numpy stim sinter pymatching
```

- [ ] **Step 2: Baseline** — confirm the existing suite fails on 3.10 today (the current wheel requires 3.12):

```bash
source "$VENV/bin/activate" && maturin develop --release -m crates/aleph-py/Cargo.toml; python -c "import aleph"
```
Expected: build succeeds but the abi3-py312 wheel is rejected for 3.10, or the import fails. Record what happens.

- [ ] **Step 3: Implement.**

`Cargo.toml` (aleph-py): pyo3 features `["extension-module", "abi3-py310"]`; add to `[dependencies]`:
```toml
aleph-qec     = { path = "../aleph-qec" }
# Parallel batch decode in `qec_core` (decoders are `Sync`).
rayon         = { workspace = true }
```

`pyproject.toml`:
```toml
requires-python = ">=3.10"
description = "High-performance quantum circuit simulator in Rust: state-vector, MPS, and stabilizer backends, plus QEC decoders for stim/sinter"
classifiers = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Rust",
    "Topic :: Scientific/Engineering :: Physics",
]
dependencies = ["numpy>=1.23"]

[project.optional-dependencies]
# `aleph.sinter` adapters; stim is what produces the DEMs sinter hands us.
sinter = ["stim>=1.13", "sinter>=1.13"]

[tool.maturin]
features = ["python"]
module-name = "aleph._native"
python-source = "python"
```

`lib.rs`: rename the module function and set its Python name:
```rust
    #[pymodule]
    #[pyo3(name = "_native")]
    fn native(m: &Bound<'_, PyModule>) -> PyResult<()> {
```
(body unchanged for now).

`python/aleph/__init__.py`:
```python
"""aleph: a high-performance quantum circuit simulator written in Rust.

The compiled extension is ``aleph._native``; this package re-exports it so
``import aleph`` exposes the same names as before v0.3, and adds pure-Python
submodules (``aleph.qec``, ``aleph.sinter``).
"""
from ._native import *  # noqa: F401,F403
from ._native import __version__  # noqa: F401  (dunders are not covered by *)
```

- [ ] **Step 4: Verify** existing tests pass on 3.10:

```bash
source "$VENV/bin/activate" && maturin develop --release -m crates/aleph-py/Cargo.toml && python -m unittest discover -s scripts/python -p "test_aleph.py" -v
```
Expected: all PASS (same count as on main). If any test asserts on `aleph.__file__` or module name, fix only the test's expectation of the module path, not behaviour. If pyo3 rejects an API as unavailable under abi3-py310, replace it with the portable equivalent and note it in the commit body.

- [ ] **Step 5: Lint + commit**

```bash
cargo fmt && cargo clippy --workspace --all-targets -- -D warnings && cargo +nightly clippy --workspace --all-targets -- -D warnings
git add crates/aleph-py Cargo.lock
git commit -m "[qec-py] aleph-py: mixed maturin layout (aleph._native), Python floor 3.10

Pure-Python submodules (qec, sinter adapter) need a package around the
extension. abi3-py310 widens reach to the stim/sinter user base.
import aleph is unchanged.

<attribution lines>"
```

---

### Task 4: Batch-decode core (no pyo3)

**Files:**
- Create: `crates/aleph-py/src/qec_core.rs`
- Modify: `crates/aleph-py/src/lib.rs` (add `mod qec_core;` **without** the `python` cfg, so `cargo test -p aleph-py` exercises it)

**Interfaces:**
- Consumes: `aleph_qec::{Decoder, DetectorErrorModel, Syndrome, MwpmDecoder, UnionFindDecoder, BpDecoder, OsdDecoder, RelayBpDecoder, RelayBpOsdDecoder, DEFAULT_LEGS, DEFAULT_MAX_ITER}`.
- Produces (used by Task 5):
  - `pub const DECODER_NAMES: &[(&str, &[&str])]`: name → allowed parameter names.
  - `pub struct DecoderParams { pub max_iter: Option<u32>, pub alpha: Option<f64>, pub osd_order: Option<usize>, pub legs: Option<usize>, pub gamma_min: Option<f64>, pub gamma_max: Option<f64>, pub seed: Option<u64> }` (derive `Default, Clone, Debug`).
  - `pub enum AnyDecoder` with `pub fn build(dem: &DetectorErrorModel, name: &str, p: &DecoderParams) -> Result<AnyDecoder, String>` and `pub fn get(&self) -> &(dyn Decoder + Sync)`.
  - `pub fn decode_rows(dec: &(dyn Decoder + Sync), bits: &[u8], shots: usize, detectors: usize, observables: usize) -> Vec<u8>` — row-major 0/1 bytes in, 0/1 bytes `[shots × observables]` out.
  - `pub fn decode_packed(dec: &(dyn Decoder + Sync), packed: &[u8], shots: usize, detectors: usize, observables: usize) -> Vec<u8>` — stim b8 rows in (`ceil(detectors/8)` bytes each), packed rows out (`ceil(observables/8)` bytes each).
  - `pub fn packed_len(bits: usize) -> usize` (= `bits.div_ceil(8)`).

- [ ] **Step 1: Failing tests** (at the bottom of `qec_core.rs`):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Repetition code d=3 over 2 rounds would be overkill; a 3-detector chain with a
    // logical on the boundary edge is enough to get non-trivial corrections.
    const DEM: &str = "\
error(0.1) D0 L0
error(0.1) D0 D1
error(0.1) D1 D2
error(0.1) D2
";

    fn dem() -> DetectorErrorModel {
        DetectorErrorModel::parse(DEM).unwrap()
    }

    #[test]
    fn every_name_builds() {
        for (name, _) in DECODER_NAMES {
            assert!(AnyDecoder::build(&dem(), name, &DecoderParams::default()).is_ok(), "{name}");
        }
    }

    #[test]
    fn unknown_name_lists_valid_ones() {
        let err = AnyDecoder::build(&dem(), "nope", &DecoderParams::default()).unwrap_err();
        assert!(err.contains("relay-bp") && err.contains("mwpm"), "{err}");
    }

    #[test]
    fn non_finite_float_param_is_rejected() {
        let p = DecoderParams { alpha: Some(f64::NAN), ..Default::default() };
        assert!(AnyDecoder::build(&dem(), "bp", &p).is_err());
        let p = DecoderParams { gamma_max: Some(f64::INFINITY), ..Default::default() };
        assert!(AnyDecoder::build(&dem(), "relay-bp", &p).is_err());
    }

    #[test]
    fn rows_match_single_decode() {
        let d = AnyDecoder::build(&dem(), "mwpm", &DecoderParams::default()).unwrap();
        // all 8 syndromes over 3 detectors
        let bits: Vec<u8> = (0..8u8).flat_map(|s| (0..3).map(move |i| (s >> i) & 1)).collect();
        let out = decode_rows(d.get(), &bits, 8, 3, 1);
        for s in 0..8usize {
            let fired: Vec<u32> = (0..3).filter(|i| bits[s * 3 + *i as usize] == 1).collect();
            let want = d.get().decode(&Syndrome::new(3, fired));
            assert_eq!(out[s] == 1, want.observable_flips[0], "syndrome {s}");
        }
    }

    #[test]
    fn packed_matches_rows() {
        // 11 detectors -> 2 bytes per row; exercises the partial trailing byte.
        let text = (0..11).map(|i| format!("error(0.05) D{i} D{}\n", (i + 1) % 11)).collect::<String>()
            + "error(0.05) D0 L0\nerror(0.05) D5 L1\n";
        let dem = DetectorErrorModel::parse(&text).unwrap();
        let d = AnyDecoder::build(&dem, "bp-osd", &DecoderParams::default()).unwrap();
        let shots = 64;
        let mut bits = vec![0u8; shots * 11];
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        for b in bits.iter_mut() {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            *b = (x % 5 == 0) as u8;
        }
        let mut packed = vec![0u8; shots * packed_len(11)];
        for s in 0..shots {
            for i in 0..11 {
                packed[s * 2 + i / 8] |= bits[s * 11 + i] << (i % 8);
            }
        }
        let rows = decode_rows(d.get(), &bits, shots, 11, 2);
        let pk = decode_packed(d.get(), &packed, shots, 11, 2);
        assert_eq!(pk.len(), shots * packed_len(2));
        for s in 0..shots {
            for o in 0..2 {
                assert_eq!((pk[s] >> o) & 1, rows[s * 2 + o], "shot {s} obs {o}");
            }
        }
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p aleph-py --lib qec_core`
Expected: compile errors (items not defined).

- [ ] **Step 3: Implement** `qec_core.rs` above the tests:

```rust
//! Batch decoding over the `aleph-qec` decoders, free of pyo3 so it is unit-tested by plain
//! `cargo test` and callable with the GIL released. Row layouts follow stim: dense rows of
//! 0/1 bytes, or bit-packed rows with detector `d` at bit `d % 8` of byte `d / 8`.

use aleph_qec::{
    BpDecoder, Decoder, DetectorErrorModel, MwpmDecoder, OsdDecoder, RelayBpDecoder,
    RelayBpOsdDecoder, Syndrome, UnionFindDecoder, DEFAULT_LEGS, DEFAULT_MAX_ITER,
};
use rayon::prelude::*;

/// Decoder names accepted by [`AnyDecoder::build`] and the parameters each one takes.
pub const DECODER_NAMES: &[(&str, &[&str])] = &[
    ("mwpm", &[]),
    ("union-find", &[]),
    ("union-find-weighted", &[]),
    ("bp", &["max_iter", "alpha"]),
    ("bp-osd", &["max_iter", "alpha", "osd_order"]),
    ("relay-bp", &["legs", "alpha", "gamma_min", "gamma_max", "seed"]),
    ("relay-bp-osd", &["legs", "alpha", "gamma_min", "gamma_max", "seed", "osd_order"]),
];

// Defaults mirror the Rust constructors (`RelayBpDecoder::new`, `OsdDecoder::new`, `BpDecoder::new`).
const RELAY_ALPHA: f64 = 0.875;
const RELAY_GAMMA: (f64, f64) = (-0.3, 0.9);
const RELAY_SEED: u64 = 0x5E1A_4B9C;
const OSD_ALPHA: f64 = 0.875;
const BP_ALPHA: f64 = 1.0;

/// Optional overrides; `None` means the decoder's default.
#[derive(Clone, Debug, Default)]
pub struct DecoderParams {
    pub max_iter: Option<u32>,
    pub alpha: Option<f64>,
    pub osd_order: Option<usize>,
    pub legs: Option<usize>,
    pub gamma_min: Option<f64>,
    pub gamma_max: Option<f64>,
    pub seed: Option<u64>,
}

/// One of the DEM-constructed decoders, behind a single type for the bindings.
pub enum AnyDecoder {
    Mwpm(MwpmDecoder),
    UnionFind(UnionFindDecoder),
    Bp(BpDecoder),
    BpOsd(OsdDecoder),
    Relay(RelayBpDecoder),
    RelayOsd(RelayBpOsdDecoder),
}

fn finite(name: &str, v: Option<f64>, default: f64) -> Result<f64, String> {
    let v = v.unwrap_or(default);
    // Explicit is_finite: NaN would slip through every later comparison (ADR 0006).
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!("parameter `{name}` must be finite, got {v}"))
    }
}

impl AnyDecoder {
    /// Build decoder `name` for `dem`. Errors are human-readable (surfaced as `ValueError`).
    pub fn build(dem: &DetectorErrorModel, name: &str, p: &DecoderParams) -> Result<Self, String> {
        let max_iter = p.max_iter.unwrap_or(DEFAULT_MAX_ITER);
        let relay = || -> Result<RelayBpDecoder, String> {
            let alpha = finite("alpha", p.alpha, RELAY_ALPHA)?;
            let gmin = finite("gamma_min", p.gamma_min, RELAY_GAMMA.0)?;
            let gmax = finite("gamma_max", p.gamma_max, RELAY_GAMMA.1)?;
            if gmin > gmax {
                return Err(format!("gamma_min ({gmin}) must be <= gamma_max ({gmax})"));
            }
            Ok(RelayBpDecoder::with_params(
                dem,
                p.legs.unwrap_or(DEFAULT_LEGS),
                alpha,
                (gmin, gmax),
                p.seed.unwrap_or(RELAY_SEED),
            ))
        };
        Ok(match name {
            "mwpm" => AnyDecoder::Mwpm(MwpmDecoder::new(dem).map_err(|e| e.to_string())?),
            "union-find" => AnyDecoder::UnionFind(UnionFindDecoder::new(dem).map_err(|e| e.to_string())?),
            "union-find-weighted" => {
                AnyDecoder::UnionFind(UnionFindDecoder::new_weighted(dem).map_err(|e| e.to_string())?)
            }
            "bp" => AnyDecoder::Bp(BpDecoder::with_params(dem, max_iter, finite("alpha", p.alpha, BP_ALPHA)?)),
            "bp-osd" => AnyDecoder::BpOsd(OsdDecoder::with_params(
                dem,
                max_iter,
                finite("alpha", p.alpha, OSD_ALPHA)?,
                p.osd_order.unwrap_or(0),
            )),
            "relay-bp" => AnyDecoder::Relay(relay()?),
            "relay-bp-osd" => {
                let alpha = finite("alpha", p.alpha, RELAY_ALPHA)?;
                AnyDecoder::RelayOsd(RelayBpOsdDecoder::with_parts(
                    relay()?,
                    OsdDecoder::with_params(dem, DEFAULT_MAX_ITER, alpha, p.osd_order.unwrap_or(0)),
                ))
            }
            other => {
                let names: Vec<&str> = DECODER_NAMES.iter().map(|(n, _)| *n).collect();
                return Err(format!("unknown decoder `{other}`; valid: {}", names.join(", ")));
            }
        })
    }

    /// The decoder as a trait object (all variants are `Sync`, so batches decode in parallel).
    pub fn get(&self) -> &(dyn Decoder + Sync) {
        match self {
            AnyDecoder::Mwpm(d) => d,
            AnyDecoder::UnionFind(d) => d,
            AnyDecoder::Bp(d) => d,
            AnyDecoder::BpOsd(d) => d,
            AnyDecoder::Relay(d) => d,
            AnyDecoder::RelayOsd(d) => d,
        }
    }
}

/// Bytes needed for `bits` bit-packed values.
pub fn packed_len(bits: usize) -> usize {
    bits.div_ceil(8)
}

/// Decode dense 0/1 rows `[shots × detectors]`; returns 0/1 rows `[shots × observables]`.
pub fn decode_rows(
    dec: &(dyn Decoder + Sync),
    bits: &[u8],
    shots: usize,
    detectors: usize,
    observables: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; shots * observables];
    if observables == 0 {
        return out;
    }
    out.par_chunks_mut(observables)
        .zip(bits.par_chunks(detectors.max(1)))
        .for_each(|(o, row)| {
            let fired = (0..detectors as u32).filter(|&d| row[d as usize] != 0).collect();
            let c = dec.decode(&Syndrome { detectors, fired });
            for (dst, &f) in o.iter_mut().zip(&c.observable_flips) {
                *dst = f as u8;
            }
        });
    out
}

/// Decode stim b8 rows (`packed_len(detectors)` bytes each); returns packed observable rows.
pub fn decode_packed(
    dec: &(dyn Decoder + Sync),
    packed: &[u8],
    shots: usize,
    detectors: usize,
    observables: usize,
) -> Vec<u8> {
    let (ib, ob) = (packed_len(detectors), packed_len(observables));
    let mut out = vec![0u8; shots * ob];
    if ob == 0 {
        return out;
    }
    out.par_chunks_mut(ob)
        .zip(packed.par_chunks(ib.max(1)))
        .for_each(|(o, row)| {
            let mut fired = Vec::new();
            for (byte_i, &byte) in row.iter().enumerate() {
                let mut b = byte;
                while b != 0 {
                    let d = byte_i * 8 + b.trailing_zeros() as usize;
                    if d < detectors {
                        fired.push(d as u32);
                    }
                    b &= b - 1;
                }
            }
            let c = dec.decode(&Syndrome { detectors, fired });
            for (i, &f) in c.observable_flips.iter().enumerate() {
                o[i / 8] |= (f as u8) << (i % 8);
            }
        });
    out
}
```

Note: when `detectors == 0` the input chunking uses `max(1)` and the zip length is governed by `out`; callers (Task 5) validate `bits.len() == shots * detectors` first, and for `detectors == 0` pass a zero-filled `shots`-length buffer. If rayon's zip truncates in that case, special-case `detectors == 0` by decoding an empty syndrome per shot. Cover it with a test `zero_detector_model` if you add the special case.

If `MwpmDecoder`/`UnionFindDecoder` etc. are not `Sync`, the compiler says so in `get()`; stop and report rather than wrapping in a `Mutex` (the spec relies on parallel decode).

- [ ] **Step 4: Run tests**

Run: `cargo test -p aleph-py --lib qec_core`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
cargo fmt && cargo clippy --workspace --all-targets -- -D warnings && cargo +nightly clippy --workspace --all-targets -- -D warnings
git add crates/aleph-py/src/qec_core.rs crates/aleph-py/src/lib.rs
git commit -m "[qec-py] Batch-decode core: named decoders, dense and stim-b8 rows, rayon

<attribution lines>"
```

---

### Task 5: pyo3 bindings `aleph.qec`

**Files:**
- Create: `crates/aleph-py/src/qec.rs`, `crates/aleph-py/python/aleph/qec.py`, `scripts/python/test_qec.py`
- Modify: `crates/aleph-py/src/lib.rs`, `crates/aleph-py/python/aleph/__init__.py`

**Interfaces:**
- Consumes: Task 4 items (`qec_core::{AnyDecoder, DecoderParams, DECODER_NAMES, decode_rows, decode_packed, packed_len}`).
- Produces (Python, used by Task 6): `aleph.qec.DetectorErrorModel(text_or_stim_obj)`, `.from_file(path)`, `.num_detectors`, `.num_observables`, `.num_errors`, `.to_dem_string()`; `aleph.qec.Decoder(dem, name, **params)` with `.name`, `.decode(a)`, `.decode_batch(a)`, `.decode_batch_bit_packed(a)`; `aleph.qec.decoder_names() -> list[str]`.

- [ ] **Step 1: Failing Python tests** — `scripts/python/test_qec.py`:

```python
"""Tests for aleph.qec (DEM parsing + decoders) against stim-generated models.

Run after installing the wheel:  python -m unittest discover -s scripts/python -v
Skips cleanly when stim / pymatching are absent.
"""
import unittest

import numpy as np

try:
    import aleph.qec as qec
    HAVE_ALEPH = True
except ImportError:
    HAVE_ALEPH = False

try:
    import stim
    HAVE_STIM = True
except ImportError:
    HAVE_STIM = False

try:
    import pymatching
    HAVE_PYMATCHING = True
except ImportError:
    HAVE_PYMATCHING = False

ALL = ["mwpm", "union-find", "union-find-weighted", "bp", "bp-osd", "relay-bp", "relay-bp-osd"]
MATCHING = {"mwpm", "union-find", "union-find-weighted"}


def surface(d, p, rounds=None):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=d,
        rounds=rounds or d,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def color(d, p):
    return stim.Circuit.generated(
        "color_code:memory_xyz", distance=d, rounds=d,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
    )


@unittest.skipUnless(HAVE_ALEPH, "aleph not installed")
class TestApi(unittest.TestCase):
    DEM = "error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1\n"

    def test_names(self):
        self.assertEqual(sorted(qec.decoder_names()), sorted(ALL))

    def test_dem_properties(self):
        dem = qec.DetectorErrorModel(self.DEM)
        self.assertEqual((dem.num_detectors, dem.num_observables, dem.num_errors), (2, 1, 3))
        self.assertEqual(qec.DetectorErrorModel(dem.to_dem_string()).to_dem_string(), dem.to_dem_string())

    def test_bad_inputs_raise_value_error(self):
        dem = qec.DetectorErrorModel(self.DEM)
        with self.assertRaises(ValueError):
            qec.DetectorErrorModel("frobnicate D0\n")
        with self.assertRaisesRegex(ValueError, "relay-bp"):
            qec.Decoder(dem, "nope")
        with self.assertRaisesRegex(ValueError, "osd_order"):
            qec.Decoder(dem, "mwpm", osd_order=2)
        with self.assertRaises(ValueError):
            qec.Decoder(dem, "bp", alpha=float("nan"))
        dec = qec.Decoder(dem, "mwpm")
        with self.assertRaises(ValueError):
            dec.decode_batch(np.zeros((4, 3), dtype=bool))  # wrong detector count
        with self.assertRaises(ValueError):
            dec.decode_batch_bit_packed(np.zeros((4, 2), dtype=np.uint8))  # needs 1 byte/row

    def test_decode_shapes_and_dtypes(self):
        dec = qec.Decoder(qec.DetectorErrorModel(self.DEM), "mwpm")
        self.assertEqual(dec.name, "mwpm")
        one = dec.decode(np.array([1, 0], dtype=np.uint8))
        self.assertEqual((one.dtype, one.shape), (np.bool_, (1,)))
        self.assertTrue(one[0])  # D0 alone -> boundary edge carrying L0
        batch = dec.decode_batch(np.array([[1, 0], [0, 0], [1, 1]], dtype=bool))
        self.assertEqual(batch.shape, (3, 1))
        self.assertEqual(batch[:, 0].tolist(), [True, False, False])
        packed = dec.decode_batch_bit_packed(np.array([[1], [0], [3]], dtype=np.uint8))
        self.assertEqual((packed.dtype, packed.shape), (np.uint8, (3, 1)))
        self.assertEqual(packed[:, 0].tolist(), [1, 0, 0])


@unittest.skipUnless(HAVE_ALEPH and HAVE_STIM, "needs aleph + stim")
class TestStimDems(unittest.TestCase):
    def test_compressed_equals_flattened(self):
        for circ, decompose in ((surface(3, 0.003), True), (color(3, 0.003), False)):
            dem = circ.detector_error_model(decompose_errors=decompose)
            a = qec.DetectorErrorModel(dem)
            b = qec.DetectorErrorModel(dem.flattened())
            self.assertEqual(a.to_dem_string(), b.to_dem_string())
            self.assertEqual(a.num_detectors, dem.num_detectors)
            self.assertEqual(a.num_observables, dem.num_observables)

    def test_every_decoder_on_decomposed_surface_code(self):
        circ = surface(3, 0.003)
        dem = qec.DetectorErrorModel(circ.detector_error_model(decompose_errors=True))
        dets, _ = circ.compile_detector_sampler(seed=7).sample(2000, separate_observables=True)
        packed = np.packbits(dets, axis=1, bitorder="little")
        for name in ALL:
            with self.subTest(name=name):
                dec = qec.Decoder(dem, name)
                batch = dec.decode_batch(dets)
                again = dec.decode_batch(dets)
                np.testing.assert_array_equal(batch, again)  # deterministic
                single = np.array([dec.decode(row) for row in dets[:200]])
                np.testing.assert_array_equal(batch[:200], single)
                pk = dec.decode_batch_bit_packed(packed)
                np.testing.assert_array_equal(
                    np.unpackbits(pk, axis=1, bitorder="little", count=batch.shape[1]).astype(bool),
                    batch,
                )

    def test_hypergraph_dem(self):
        circ = color(3, 0.003)
        dem = qec.DetectorErrorModel(circ.detector_error_model())
        for name in MATCHING:
            with self.assertRaisesRegex(ValueError, "graph"):
                qec.Decoder(dem, name)
        dets, obs = circ.compile_detector_sampler(seed=3).sample(5000, separate_observables=True)
        raw = int(np.any(obs, axis=1).sum())
        for name in ("bp-osd", "relay-bp-osd"):
            pred = qec.Decoder(dem, name).decode_batch(dets)
            errs = int(np.any(pred != obs, axis=1).sum())
            self.assertLess(errs, raw, f"{name}: {errs} >= undecoded {raw}")


@unittest.skipUnless(HAVE_ALEPH and HAVE_STIM and HAVE_PYMATCHING, "needs pymatching")
class TestOracle(unittest.TestCase):
    def test_mwpm_matches_pymatching(self):
        circ = surface(5, 0.005)
        sdem = circ.detector_error_model(decompose_errors=True)
        dets, obs = circ.compile_detector_sampler(seed=11).sample(20000, separate_observables=True)
        ours = qec.Decoder(qec.DetectorErrorModel(sdem), "mwpm").decode_batch(dets)
        theirs = pymatching.Matching.from_detector_error_model(sdem).decode_batch(dets).astype(bool)
        a = int(np.any(ours != obs, axis=1).sum())
        p = int(np.any(theirs != obs, axis=1).sum())
        self.assertLessEqual(abs(a - p), max(5, 0.1 * p), f"aleph {a} vs pymatching {p}")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `source $VENV/bin/activate && maturin develop --release -m crates/aleph-py/Cargo.toml && python -m unittest scripts/python/test_qec.py -v`
Expected: ImportError / AttributeError for `aleph.qec`.

- [ ] **Step 3: Implement** `crates/aleph-py/src/qec.rs`:

```rust
//! `aleph.qec` bindings: [`DetectorErrorModel`] parsing and the DEM decoders of `aleph-qec`,
//! with numpy batch decoding that releases the GIL (see `qec_core`).
// pyo3 0.22 proc-macro expansion emits trivial PyErr→PyErr .into() calls.
#![allow(clippy::useless_conversion)]

use crate::qec_core::{self, AnyDecoder, DecoderParams, DECODER_NAMES};
use aleph_qec::DetectorErrorModel;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
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
            self.inner.detectors, self.inner.observables, self.inner.errors.len()
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
            _ => unreachable!("every allowed parameter is matched above"),
        }
    }
    Ok(p)
}

/// Flatten a bool or uint8 numpy array to 0/1 bytes, checking the trailing dimension.
fn to_bytes(a: &Bound<'_, PyAny>, ndim: usize, cols: usize, what: &str) -> PyResult<(usize, Vec<u8>)> {
    let (shape, data): (Vec<usize>, Vec<u8>) = if let Ok(arr) = a.extract::<PyReadonlyArrayDyn<'_, bool>>() {
        (arr.shape().to_vec(), arr.as_array().iter().map(|&b| b as u8).collect())
    } else if let Ok(arr) = a.extract::<PyReadonlyArrayDyn<'_, u8>>() {
        (arr.shape().to_vec(), arr.as_array().iter().copied().collect())
    } else {
        return Err(value_err(format!("{what}: expected a numpy bool or uint8 array")));
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
}

#[pymethods]
impl PyDecoder {
    #[new]
    #[pyo3(signature = (dem, name, **params))]
    fn new(dem: PyRef<'_, PyDem>, name: &str, params: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let p = parse_params(name, params)?;
        let dec = AnyDecoder::build(&dem.inner, name, &p).map_err(value_err)?;
        Ok(PyDecoder {
            dec,
            name: name.to_string(),
            detectors: dem.inner.detectors,
            observables: dem.inner.observables,
        })
    }

    /// Decode one shot: `[num_detectors]` bool/uint8 → `[num_observables]` bool.
    fn decode<'py>(&self, py: Python<'py>, dets: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let (_, bits) = to_bytes(dets, 1, self.detectors, "decode")?;
        let out = qec_core::decode_rows(self.dec.get(), &bits, 1, self.detectors, self.observables);
        Ok(PyArray1::from_vec_bound(py, out.into_iter().map(|b| b != 0).collect()))
    }

    /// Decode `[shots, num_detectors]` bool/uint8 → `[shots, num_observables]` bool (GIL released).
    fn decode_batch<'py>(&self, py: Python<'py>, dets: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArray2<bool>>> {
        let (shots, bits) = to_bytes(dets, 2, self.detectors, "decode_batch")?;
        let (d, o) = (self.detectors, self.observables);
        let dec = self.dec.get();
        let out = py.allow_threads(|| qec_core::decode_rows(dec, &bits, shots, d, o));
        let flat = PyArray1::from_vec_bound(py, out.into_iter().map(|b| b != 0).collect());
        flat.reshape([shots, o])
    }

    /// stim b8 rows `[shots, ceil(D/8)]` uint8 → packed predictions `[shots, ceil(O/8)]` uint8.
    fn decode_batch_bit_packed<'py>(&self, py: Python<'py>, packed: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArray2<u8>>> {
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

    fn __repr__(&self) -> String {
        format!("Decoder({:?}, detectors={}, observables={})", self.name, self.detectors, self.observables)
    }
}

/// Register the `qec` submodule on the native module.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "qec")?;
    m.add_class::<PyDem>()?;
    m.add_class::<PyDecoder>()?;
    m.add_function(wrap_pyfunction!(decoder_names, &m)?)?;
    parent.add_submodule(&m)
}
```

Notes for the implementer: `unreachable!` above is guarded by the `allowed` check — if clippy/CLAUDE.md review objects, replace with an explicit `Err(value_err(...))`. Exact numpy 0.22 method names (`from_vec_bound`, `reshape`, `as_any`) may differ slightly; follow the compiler, keep behaviour. `allow_threads` needs the closure to be `Send`: `&(dyn Decoder + Sync)` is `Send`.

`lib.rs`: add `#[cfg(feature = "python")] mod qec;` and at the end of the `_native` module body `crate::qec::register(m)?;`.

`python/aleph/qec.py`:
```python
"""QEC decoding: stim-format Detector Error Models and aleph's decoders.

    import aleph.qec as qec
    dem = qec.DetectorErrorModel(stim_circuit.detector_error_model(decompose_errors=True))
    predictions = qec.Decoder(dem, "relay-bp-osd").decode_batch(detection_events)

``decoder_names()`` lists the available decoders. stim is not required.
"""
from . import _native

DetectorErrorModel = _native.qec.DetectorErrorModel
Decoder = _native.qec.Decoder
decoder_names = _native.qec.decoder_names

__all__ = ["DetectorErrorModel", "Decoder", "decoder_names"]
```

Append to `python/aleph/__init__.py`:
```python
from . import qec  # noqa: E402,F401  (replaces the native `qec` attribute with the Python module)
```

- [ ] **Step 4: Run tests**

```bash
source $VENV/bin/activate && maturin develop --release -m crates/aleph-py/Cargo.toml && python -m unittest discover -s scripts/python -v
```
Expected: all PASS including `TestOracle`. If `test_mwpm_matches_pymatching` fails, **do not loosen the tolerance** — stop and report the two counts (a real gap between two exact MWPMs is a bug in edge weights or in component handling).

- [ ] **Step 5: Lint + commit**

```bash
cargo fmt && cargo clippy --workspace --all-targets -- -D warnings && cargo +nightly clippy --workspace --all-targets -- -D warnings
cargo clippy -p aleph-py --features python --all-targets -- -D warnings
git add crates/aleph-py scripts/python/test_qec.py
git commit -m "[qec-py] aleph.qec: DetectorErrorModel + Decoder bindings with GIL-free batch decode

Differential vs single-shot and bit-packed paths on stim DEMs; MWPM
matches pymatching on d=5 (same shots).

<attribution lines>"
```

---

### Task 6: sinter adapter

**Files:**
- Create: `crates/aleph-py/python/aleph/sinter.py`, `scripts/python/test_sinter.py`

**Interfaces:**
- Consumes: `aleph.qec.{DetectorErrorModel, Decoder, decoder_names}` (Task 5).
- Produces: `aleph.sinter.decoders(params: dict | None = None) -> dict[str, AlephSinterDecoder]`, keys `"aleph-<name>"`; classes `AlephSinterDecoder`, `CompiledAlephDecoder`.

- [ ] **Step 1: Failing tests** — `scripts/python/test_sinter.py`:

```python
"""aleph.sinter adapters: picklable, and usable by sinter.collect with worker processes."""
import pickle
import unittest

import numpy as np

try:
    import sinter
    import stim
    import aleph.sinter as asinter
    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "needs aleph + stim + sinter")
class TestSinter(unittest.TestCase):
    def test_all_names_present_and_picklable(self):
        import aleph.qec as qec
        decs = asinter.decoders(params={"relay-bp": {"legs": 2}})
        self.assertEqual(sorted(decs), sorted(f"aleph-{n}" for n in qec.decoder_names()))
        for key, d in decs.items():
            back = pickle.loads(pickle.dumps(d))
            self.assertEqual((back.name, back.params), (d.name, d.params), key)
        self.assertEqual(decs["aleph-relay-bp"].params, {"legs": 2})

    def test_compiled_decoder_bit_packed(self):
        circ = stim.Circuit.generated("repetition_code:memory", distance=5, rounds=5,
                                      after_clifford_depolarization=0.01)
        dem = circ.detector_error_model(decompose_errors=True)
        dets, obs = circ.compile_detector_sampler(seed=5).sample(
            1000, separate_observables=True, bit_packed=True)
        comp = asinter.decoders()["aleph-mwpm"].compile_decoder_for_dem(dem=dem)
        pred = comp.decode_shots_bit_packed(bit_packed_detection_event_data=dets)
        self.assertEqual((pred.dtype, pred.shape), (np.uint8, obs.shape))

    def test_collect_with_workers(self):
        circ = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3,
                                      after_clifford_depolarization=0.005)
        tasks = [sinter.Task(circuit=circ, json_metadata={"d": 3})]
        stats = sinter.collect(num_workers=2, tasks=tasks,
                               decoders=["aleph-mwpm", "aleph-relay-bp-osd"],
                               custom_decoders=asinter.decoders(),
                               max_shots=2000, max_errors=50)
        self.assertEqual(sorted(s.decoder for s in stats), ["aleph-mwpm", "aleph-relay-bp-osd"])
        for s in stats:
            self.assertGreater(s.shots, 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m unittest scripts/python/test_sinter.py -v` → skipped/ImportError for `aleph.sinter` (the `HAVE` flag is False because the import fails; confirm with `python -c "import aleph.sinter"` → ModuleNotFoundError).

- [ ] **Step 3: Implement** `python/aleph/sinter.py`:

```python
"""sinter adapters for aleph's decoders.

    import sinter, aleph.sinter
    sinter.collect(tasks=..., decoders=["aleph-relay-bp-osd"],
                   custom_decoders=aleph.sinter.decoders(), num_workers=8)

Adapters hold only a decoder name and parameters, so they pickle cleanly into
sinter's worker processes; the Rust decoder is built per DEM in each worker.
"""
from __future__ import annotations

try:
    import sinter as _sinter
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "aleph.sinter requires sinter; install with: pip install 'aleph-sim[sinter]'"
    ) from e

from . import qec as _qec

__all__ = ["AlephSinterDecoder", "CompiledAlephDecoder", "decoders"]


class CompiledAlephDecoder(_sinter.CompiledDecoder):
    """An aleph decoder built for one DEM."""

    def __init__(self, decoder: _qec.Decoder) -> None:
        self._decoder = decoder

    def decode_shots_bit_packed(self, *, bit_packed_detection_event_data):
        return self._decoder.decode_batch_bit_packed(bit_packed_detection_event_data)


class AlephSinterDecoder(_sinter.Decoder):
    """sinter decoder for aleph decoder ``name`` with keyword ``params``."""

    def __init__(self, name: str, **params) -> None:
        self.name = name
        self.params = dict(params)

    def compile_decoder_for_dem(self, *, dem) -> CompiledAlephDecoder:
        model = _qec.DetectorErrorModel(dem)
        return CompiledAlephDecoder(_qec.Decoder(model, self.name, **self.params))

    def decode_via_files(self, *, num_shots, num_dets, num_obs, dem_path,
                         dets_b8_in_path, obs_predictions_b8_out_path, tmp_dir) -> None:
        import stim

        dem = stim.DetectorErrorModel.from_file(dem_path)
        dets = stim.read_shot_data_file(path=dets_b8_in_path, format="b8",
                                        num_detectors=num_dets, bit_packed=True)
        pred = self.compile_decoder_for_dem(dem=dem).decode_shots_bit_packed(
            bit_packed_detection_event_data=dets)
        stim.write_shot_data_file(data=pred, path=obs_predictions_b8_out_path, format="b8",
                                  num_observables=num_obs)

    def __repr__(self) -> str:
        return f"AlephSinterDecoder({self.name!r}, **{self.params!r})"


def decoders(params: dict | None = None) -> dict[str, AlephSinterDecoder]:
    """``{"aleph-<name>": AlephSinterDecoder}`` for every aleph decoder.

    ``params`` maps a decoder name (e.g. ``"relay-bp"``) to its keyword parameters.
    """
    params = params or {}
    return {f"aleph-{n}": AlephSinterDecoder(n, **params.get(n, {})) for n in _qec.decoder_names()}
```

(`from __future__ import annotations` keeps the `dict | None` hint valid on 3.10's runtime even though 3.10 supports it; keep it for consistency.) Do **not** import `aleph.sinter` from `aleph/__init__.py`.

- [ ] **Step 4: Run tests**

```bash
source $VENV/bin/activate && maturin develop --release -m crates/aleph-py/Cargo.toml && python -m unittest discover -s scripts/python -v
python -c "import sys, aleph, aleph.qec; assert 'sinter' not in sys.modules and 'stim' not in sys.modules; print('clean import')"
```
Expected: all PASS; `clean import`.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-py/python/aleph/sinter.py scripts/python/test_sinter.py
git commit -m "[qec-py] aleph.sinter: picklable sinter adapters for every aleph decoder

<attribution lines>"
```

---

### Task 7: CI and release workflow

**Files:**
- Modify: `.github/workflows/ci.yml` (`test-python` job), `.github/workflows/release.yml` (smoke test)

- [ ] **Step 1: `ci.yml` `test-python`** — add a matrix and QEC deps:

```yaml
  test-python:
    name: test python ${{ matrix.python }}
    ...
    strategy:
      fail-fast: false
      matrix:
        python: ["3.10", "3.12"]
    steps:
      ...
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - run: pip install maturin
      ...
      - name: Install the wheel
        run: pip install dist/*.whl
      # stim/sinter/pymatching drive the aleph.qec differential + oracle tests
      # (scripts/python/test_qec.py, test_sinter.py); without them those skip.
      - name: Install QEC test deps
        run: pip install stim sinter pymatching
```
Keep the existing comments; `timeout-minutes` 15 → 20 (the oracle test samples 20k d=5 shots).

- [ ] **Step 2: `release.yml` smoke test** — use Python `"3.10"` in the smoke-test `setup-python` (proves the abi3 floor on the artifact that ships) and extend the smoke command:

```bash
python -c "import aleph; c = aleph.Circuit(2); c.h(0); c.cx(0, 1); r = aleph.run(c, shots=100, seed=0); assert set(r.counts()) == {'00', '11'}, r.counts(); print('wheel ok:', r.counts())"
python -c "import numpy as np, aleph.qec as q; d = q.Decoder(q.DetectorErrorModel('error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1\n'), 'mwpm'); assert d.decode(np.array([1, 0], dtype=bool)).tolist() == [True]; print('qec ok')"
```

- [ ] **Step 3: Validate YAML locally**

Run: `python3 -c "import yaml,sys; [yaml.safe_load(open(f)) for f in sys.argv[1:]]; print('ok')" .github/workflows/ci.yml .github/workflows/release.yml` (install `pyyaml` into `$VENV` if missing and use its python).
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml .github/workflows/release.yml
git commit -m "[qec-py] CI: Python 3.10 + 3.12 matrix with stim/sinter/pymatching; QEC wheel smoke

<attribution lines>"
```

---

### Task 8: v0.3.0 — version, CHANGELOG, README, throughput table

**Files:**
- Modify: `Cargo.toml` (`[workspace.package] version`), `Cargo.lock`, `README.md`, `crates/aleph-py/README.md`
- Create: `CHANGELOG.md`, `scripts/python/bench_qec.py`

- [ ] **Step 1: Version** — `version = "0.3.0"` in root `Cargo.toml`; `cargo build --workspace` to refresh `Cargo.lock`. `grep -rn '0\.2\.0' README.md crates/aleph-py/README.md docs/*.md` and update install snippets that pin the version (leave historical perf reports alone).

- [ ] **Step 2: Bench script** — `scripts/python/bench_qec.py`:

```python
"""Decode throughput (shots/s) of aleph decoders vs pymatching / ldpc on stim DEMs.

Usage: python scripts/python/bench_qec.py  (prints a Markdown table)
"""
import platform
import time

import numpy as np
import stim

import aleph.qec as qec


def surface(d, p=0.003):
    return stim.Circuit.generated("surface_code:rotated_memory_x", distance=d, rounds=d,
                                  after_clifford_depolarization=p,
                                  before_round_data_depolarization=p,
                                  before_measure_flip_probability=p,
                                  after_reset_flip_probability=p)


def color(d, p=0.003):
    return stim.Circuit.generated("color_code:memory_xyz", distance=d, rounds=d,
                                  after_clifford_depolarization=p,
                                  before_measure_flip_probability=p)


def rate(fn, packed, shots):
    fn(packed[: min(200, shots)])  # warm-up
    t = time.perf_counter()
    fn(packed)
    return shots / (time.perf_counter() - t)


def main():
    rows = []
    cases = [("surface d=5", surface(5), True, ["mwpm", "union-find-weighted", "bp-osd", "relay-bp-osd"]),
             ("surface d=9", surface(9), True, ["mwpm", "union-find-weighted", "relay-bp-osd"]),
             ("color d=5", color(5), False, ["bp-osd", "relay-bp", "relay-bp-osd"])]
    for label, circ, decompose, names in cases:
        sdem = circ.detector_error_model(decompose_errors=decompose)
        shots = 20000
        packed, _ = circ.compile_detector_sampler(seed=1).sample(shots, separate_observables=True, bit_packed=True)
        dem = qec.DetectorErrorModel(sdem)
        for n in names:
            dec = qec.Decoder(dem, n)
            rows.append((label, f"aleph {n}", rate(dec.decode_batch_bit_packed, packed, shots)))
        if decompose:
            try:
                import pymatching
                m = pymatching.Matching.from_detector_error_model(sdem)
                dense = np.unpackbits(packed, axis=1, bitorder="little", count=sdem.num_detectors)
                rows.append((label, "pymatching", rate(m.decode_batch, dense, shots)))
            except ImportError:
                pass
        else:
            try:
                from ldpc.sinter_decoders import SinterBpOsdDecoder
                c = SinterBpOsdDecoder().compile_decoder_for_dem(dem=sdem)
                rows.append((label, "ldpc bp-osd",
                             rate(lambda x: c.decode_shots_bit_packed(bit_packed_detection_event_data=x), packed, shots)))
            except ImportError:
                pass
    print(f"Machine: {platform.platform()} / {platform.processor() or platform.machine()}\n")
    print("| DEM | decoder | shots/s |\n|---|---|---:|")
    for label, name, r in rows:
        print(f"| {label} | {name} | {r:,.0f} |")


if __name__ == "__main__":
    main()
```

Run: `source $VENV/bin/activate && uv pip install -p $VENV ldpc; python scripts/python/bench_qec.py`. Before running, check the machine is idle (`uptime`, no cargo/bench processes). Save the output; it goes verbatim into the README.

- [ ] **Step 3: CHANGELOG.md** — new file, Keep-a-Changelog style. `## [0.3.0] — <release date>` with sections *Added* (aleph.qec decoders + sinter adapters; DEM `repeat`/`shift_detectors`/`^`; CUDA SV fusion / Aer-GPU parity (P5.9, `docs/perf/p5.9-gpu-fusion.md`); tiled fused-block kernel, host paging, FP32 CUDA backend (P5.10); each P5.11 item with its `docs/perf/p5.11-0*.md` link — read each report's first paragraph for the one-line summary; noise models v1 (P4.6)), *Changed* (Python ≥3.10; extension module is now `aleph._native`, `import aleph` unchanged), and `## [0.2.0] — 2026-06-12` (one line: "CPU parity release; see `docs/perf/parity.md`"). Every claim must link to the doc it comes from; no numbers that are not in those docs.

- [ ] **Step 4: README** — in root `README.md` and `crates/aleph-py/README.md` (the PyPI page), add a `## QEC decoding` section:

````markdown
## QEC decoding

aleph ships MWPM, Union-Find, BP, BP-OSD and relay-BP decoders for stim
Detector Error Models, usable directly or through sinter:

```python
pip install "aleph-sim[sinter]"

import sinter, stim, aleph.sinter
circ = stim.Circuit.generated("surface_code:rotated_memory_x", distance=5, rounds=5,
                              after_clifford_depolarization=0.003)
stats = sinter.collect(tasks=[sinter.Task(circuit=circ)], num_workers=4,
                       decoders=["aleph-mwpm", "aleph-relay-bp-osd", "pymatching"],
                       custom_decoders=aleph.sinter.decoders(), max_shots=100_000)
```

Throughput (`scripts/python/bench_qec.py`, <machine line from the script>):

<table from Step 2, verbatim>

pymatching (Sparse Blossom) is faster than aleph's MWPM; aleph's matching decoder is an
exact dense-blossom implementation (Sparse Blossom is tracked in #331).
````
Adjust the last sentence to what the table actually shows — state plainly where aleph is slower and where it is faster.

- [ ] **Step 5: Full verification**

```bash
cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo +nightly clippy --workspace --all-targets -- -D warnings && cargo test --workspace
source $VENV/bin/activate && maturin develop --release -m crates/aleph-py/Cargo.toml && python -m unittest discover -s scripts/python -v
```
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock CHANGELOG.md README.md crates/aleph-py/README.md scripts/python/bench_qec.py
git commit -m "[qec-py] v0.3.0: version bump, CHANGELOG, README QEC section with measured throughput

<attribution lines>"
```

---

### Task 9: PR (controller, not a subagent)

- [ ] Create the GitHub issue `[QEC-PY] QEC decoders in Python + v0.3.0` whose body links the spec; open PR `[QEC-PY] aleph.qec + sinter adapters, v0.3.0` from `qec-python-v0.3` with `Closes #<issue-number>`, summary, test results (Rust + Python on 3.10/3.12, oracle counts), the throughput table, and follow-ups (windowed decoders, Sparse Blossom #331).
- [ ] Wait for CI green; self-review the diff.
- [ ] **Ask the user** before: merging, pushing tag `v0.3.0`, and the TestPyPI → PyPI publish (`release.yml`).
