//! Detector Error Model (DEM) — the graph-like description of independent error mechanisms
//! that a decoder consumes.
//!
//! Each error mechanism flips a set of *detectors* (parity checks across space-time) and a
//! set of *logical observables*, with an independent probability. This mirrors Stim's
//! `.dem` format so we can cross-check against Stim-generated models (Q0-03) and decode the
//! same DEMs as PyMatching (Q1).
//!
//! Format reference: <https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem.md>
//!
//! Supported instructions: `error` (with `^` separable components), `detector`,
//! `logical_observable`, `shift_detectors`, and (nested) `repeat` blocks, which are unrolled
//! into a flat model. `detector_separator` is rejected.

use crate::error::{Error, Result};

/// One independent error mechanism: with probability [`prob`](DemError::prob) it flips the
/// listed detectors and logical observables.
///
/// Detector and observable indices are kept sorted ascending (see [`DemError::new`]) so two
/// mechanisms compare equal regardless of the order targets were written in.
#[derive(Clone, Debug, PartialEq)]
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
        DemError {
            prob,
            dets,
            obs,
            components: Vec::new(),
        }
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
}

/// A Detector Error Model: a count of detectors and observables plus the list of error
/// mechanisms over them.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct DetectorErrorModel {
    /// Number of detectors (their valid indices are `0..detectors`).
    pub detectors: usize,
    /// Number of logical observables (their valid indices are `0..observables`).
    pub observables: usize,
    /// The independent error mechanisms, in declaration order.
    pub errors: Vec<DemError>,
}

impl DetectorErrorModel {
    /// Parse a Stim-style `.dem` text into a [`DetectorErrorModel`].
    ///
    /// `detectors`/`observables` are taken as one past the largest index that appears in any
    /// `error`, `detector`, or `logical_observable` instruction.
    ///
    /// # Errors
    /// Returns [`Error::DemParse`] on malformed input.
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

    /// Emit the model as Stim-style `.dem` text.
    ///
    /// One `error(p) D.. L..` line per mechanism, with targets in ascending order. If the
    /// declared [`detectors`](Self::detectors)/[`observables`](Self::observables) counts
    /// exceed the largest index actually used by a mechanism, a single trailing `detector` /
    /// `logical_observable` declaration pins the count so [`parse`](Self::parse) recovers it
    /// exactly. The result therefore round-trips: `parse(emit(m)) == m`.
    pub fn to_dem_string(&self) -> String {
        let mut out = String::new();
        let (mut used_det, mut used_obs) = (0usize, 0usize); // one-past-max used index
        for e in &self.errors {
            out.push_str("error(");
            out.push_str(&e.prob.to_string());
            out.push(')');
            let single = [(e.dets.clone(), e.obs.clone())];
            let parts: &[(Vec<u32>, Vec<u32>)] = if e.components.is_empty() {
                &single
            } else {
                &e.components
            };
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
            out.push('\n');
        }
        if self.detectors > used_det {
            out.push_str(&format!("detector D{}\n", self.detectors - 1));
        }
        if self.observables > used_obs {
            out.push_str(&format!("logical_observable L{}\n", self.observables - 1));
        }
        out
    }
}

/// A single parsed `D<n>` / `L<n>` target.
enum Target {
    Det(u32),
    Obs(u32),
}

fn parse_target(tok: &str, line: usize) -> Result<Target> {
    let parse_idx = |s: &str, kind: &str| {
        s.parse::<u32>().map_err(|e| Error::DemParse {
            line,
            msg: format!("invalid {kind} index in `{tok}`: {e}"),
        })
    };
    if let Some(n) = tok.strip_prefix('D') {
        Ok(Target::Det(parse_idx(n, "detector")?))
    } else if let Some(n) = tok.strip_prefix('L') {
        Ok(Target::Obs(parse_idx(n, "observable")?))
    } else {
        Err(Error::DemParse {
            line,
            msg: format!("unexpected target `{tok}` (expected D<n> or L<n>)"),
        })
    }
}

/// Strip a trailing `# ...` comment from a DEM line.
fn strip_comment(s: &str) -> &str {
    match s.find('#') {
        Some(i) => &s[..i],
        None => s,
    }
}

/// Split an optional leading `(...)` argument from the rest of an instruction tail.
///
/// `"(0.1) D0 D1"` -> `(Some("0.1"), "D0 D1")`; `" D0"` -> `(None, "D0")`.
fn split_paren_arg(s: &str) -> (Option<&str>, &str) {
    let s = s.trim_start();
    if let Some(rest) = s.strip_prefix('(') {
        if let Some(idx) = rest.find(')') {
            return (Some(&rest[..idx]), rest[idx + 1..].trim_start());
        }
    }
    (None, s)
}

/// One parsed DEM instruction; `repeat` bodies are kept as a tree and unrolled by [`exec_block`].
#[derive(Debug)]
enum Instr {
    /// `error(p) …` — targets relative to the current detector offset; one entry per `^` part.
    Error {
        line: usize,
        prob: f64,
        parts: Vec<(Vec<u32>, Vec<u32>)>,
    },
    /// `detector(…) D…` — declares detectors (coords ignored).
    Detector { line: usize, dets: Vec<u32> },
    /// `logical_observable L…`.
    Observable(Vec<u32>),
    /// `shift_detectors(…) k`.
    Shift(u64),
    /// `repeat n { … }`.
    Repeat(u64, Vec<Instr>),
}

#[derive(Debug)]
struct ExecState {
    offset: u64,
    max_det: i64,
    max_obs: i64,
    errors: Vec<DemError>,
}

impl Default for ExecState {
    fn default() -> Self {
        ExecState {
            offset: 0,
            max_det: -1,
            max_obs: -1,
            errors: Vec::new(),
        }
    }
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
                None => Err(Error::DemParse {
                    line: line_no,
                    msg: "unmatched `}`".into(),
                }),
            };
        }
        out.push(parse_instr(line_no, line, lines)?);
    }
    match open {
        Some(l) => Err(Error::DemParse {
            line: l,
            msg: "unclosed `repeat` block".into(),
        }),
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
        let prob_str =
            arg.ok_or_else(|| bad("`error` requires a probability in parentheses".into()))?;
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
        return Ok(Instr::Error {
            line: line_no,
            prob,
            parts,
        });
    }
    if let Some(rest) = line.strip_prefix("detector") {
        let (_coords, targets) = split_paren_arg(rest);
        let mut dets = Vec::new();
        for tok in targets.split_whitespace() {
            if let Target::Det(d) = parse_target(tok, line_no)? {
                dets.push(d);
            }
        }
        return Ok(Instr::Detector {
            line: line_no,
            dets,
        });
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
                st.errors.push(DemError::with_components(*prob, abs));
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

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn parses_known_stim_snippet() {
        // A flat DEM with coordinates, a multi-detector edge, an observable, and a comment.
        let dem = "\
# a small repetition-code-like DEM
error(0.125) D0
error(0.125) D0 D1
error(0.125) D1 L0
detector(1, 0, 0) D0
detector(3, 0, 0) D1
";
        let m = DetectorErrorModel::parse(dem).expect("parse");
        assert_eq!(m.detectors, 2);
        assert_eq!(m.observables, 1);
        assert_eq!(m.errors.len(), 3);
        assert_eq!(m.errors[0], DemError::new(0.125, vec![0], vec![]));
        assert_eq!(m.errors[1], DemError::new(0.125, vec![0, 1], vec![]));
        assert_eq!(m.errors[2], DemError::new(0.125, vec![1], vec![0]));
    }

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

    #[test]
    fn target_order_is_normalised() {
        let a = DetectorErrorModel::parse("error(0.2) D5 D1 L1 L0\n").unwrap();
        let b = DetectorErrorModel::parse("error(0.2) D1 D5 L0 L1\n").unwrap();
        assert_eq!(a, b);
    }

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
        let got: Vec<(f64, Vec<u32>, Vec<u32>)> = m
            .errors
            .iter()
            .map(|e| (e.prob, e.dets.clone(), e.obs.clone()))
            .collect();
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
            "repeat 2 {\nerror(0.1) D0\n", // unclosed
            "}\n",                         // unmatched close
            "repeat x {\n}\n",             // bad count
            "repeat 2\n}\n",               // missing brace
            "shift_detectors q\n",         // bad shift
            "detector_separator 1\n",      // unsupported instruction
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

    #[test]
    fn rejects_garbage() {
        assert!(matches!(
            DetectorErrorModel::parse("frobnicate D0\n").unwrap_err(),
            Error::DemParse { .. }
        ));
        assert!(matches!(
            DetectorErrorModel::parse("error D0\n").unwrap_err(),
            Error::DemParse { .. } // missing (prob)
        ));
    }

    #[test]
    fn boundary_count_round_trips_via_trailing_declaration() {
        // detectors=4 but errors only reach D1 -> emit must pin the count.
        let m = DetectorErrorModel {
            detectors: 4,
            observables: 2,
            errors: vec![DemError::new(0.1, vec![0, 1], vec![0])],
        };
        let text = m.to_dem_string();
        assert!(text.contains("detector D3"));
        assert!(text.contains("logical_observable L1"));
        assert_eq!(DetectorErrorModel::parse(&text).unwrap(), m);
    }

    // Generate a canonical DEM: counts are exactly one past the largest index any mechanism
    // uses (so the model is self-consistent), and targets are within those bounds.
    prop_compose! {
        fn arb_dem()(detectors in 0usize..6, observables in 0usize..3)
                    (errors in prop::collection::vec(arb_error(detectors, observables), 0..8),
                     detectors in Just(detectors), observables in Just(observables))
                    -> DetectorErrorModel {
            // Recompute counts from what is actually used so emit/parse agree without needing
            // trailing boundary declarations for *every* case (that path is covered above).
            let used_det = errors.iter().flat_map(|e| e.dets.iter()).copied().max().map_or(0, |m| m as usize + 1);
            let used_obs = errors.iter().flat_map(|e| e.obs.iter()).copied().max().map_or(0, |m| m as usize + 1);
            DetectorErrorModel {
                detectors: detectors.max(used_det),
                observables: observables.max(used_obs),
                errors,
            }
        }
    }

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

    proptest! {
        // Rust's f64 Display is the shortest round-trippable representation, so probabilities
        // survive emit->parse exactly; integer targets are exact. Hence full identity.
        #[test]
        fn emit_parse_round_trip(m in arb_dem()) {
            let text = m.to_dem_string();
            let reparsed = DetectorErrorModel::parse(&text).expect("reparse");
            prop_assert_eq!(reparsed, m);
        }
    }
}
