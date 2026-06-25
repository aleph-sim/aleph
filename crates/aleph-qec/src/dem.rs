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
//! **Q0-01 subset.** We parse/emit *flat* DEMs: `error`, `detector`, and
//! `logical_observable` instructions. `repeat` blocks and `shift_detectors` are rejected
//! with [`Error::UnsupportedDem`]; they are handled when we start consuming Stim output
//! directly (Q0-03). The `^` separable-component marker inside an `error` is accepted and
//! its components are merged (a decoder's decomposition hint is not needed at this layer).

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
    /// Detector indices flipped by this mechanism (sorted ascending).
    pub dets: Vec<u32>,
    /// Logical observable indices flipped by this mechanism (sorted ascending).
    pub obs: Vec<u32>,
}

impl DemError {
    /// Build a mechanism, normalising target order so equality is order-independent.
    pub fn new(prob: f64, mut dets: Vec<u32>, mut obs: Vec<u32>) -> Self {
        dets.sort_unstable();
        obs.sort_unstable();
        DemError { prob, dets, obs }
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
    /// Returns [`Error::DemParse`] on malformed input and [`Error::UnsupportedDem`] on
    /// `repeat`/`shift_detectors` (see the module docs).
    pub fn parse(text: &str) -> Result<Self> {
        let mut errors = Vec::new();
        // -1 sentinel so "no index seen" maps to a count of 0.
        let mut max_det: i64 = -1;
        let mut max_obs: i64 = -1;

        for (i, raw) in text.lines().enumerate() {
            let line_no = i + 1;
            let line = strip_comment(raw).trim();
            if line.is_empty() {
                continue;
            }

            if let Some(rest) = line.strip_prefix("error") {
                let (arg, targets) = split_paren_arg(rest);
                let prob_str = arg.ok_or_else(|| Error::DemParse {
                    line: line_no,
                    msg: "`error` requires a probability in parentheses".into(),
                })?;
                let prob = prob_str
                    .trim()
                    .parse::<f64>()
                    .map_err(|e| Error::DemParse {
                        line: line_no,
                        msg: format!("invalid probability `{prob_str}`: {e}"),
                    })?;
                let (mut dets, mut obs) = (Vec::new(), Vec::new());
                for tok in targets.split_whitespace() {
                    if tok == "^" {
                        continue; // separable-component marker; merge components
                    }
                    match parse_target(tok, line_no)? {
                        Target::Det(d) => {
                            max_det = max_det.max(d as i64);
                            dets.push(d);
                        }
                        Target::Obs(o) => {
                            max_obs = max_obs.max(o as i64);
                            obs.push(o);
                        }
                    }
                }
                errors.push(DemError::new(prob, dets, obs));
            } else if let Some(rest) = line.strip_prefix("detector") {
                // Optional `(coords)` then a single `D<n>` target; coords are ignored here.
                let (_coords, targets) = split_paren_arg(rest);
                for tok in targets.split_whitespace() {
                    if let Target::Det(d) = parse_target(tok, line_no)? {
                        max_det = max_det.max(d as i64);
                    }
                }
            } else if let Some(rest) = line.strip_prefix("logical_observable") {
                for tok in rest.split_whitespace() {
                    if let Target::Obs(o) = parse_target(tok, line_no)? {
                        max_obs = max_obs.max(o as i64);
                    }
                }
            } else if line.starts_with("repeat") || line.starts_with("shift_detectors") {
                let what = line.split_whitespace().next().unwrap_or(line).to_string();
                return Err(Error::UnsupportedDem {
                    line: line_no,
                    what,
                });
            } else {
                return Err(Error::DemParse {
                    line: line_no,
                    msg: format!("unknown instruction: `{line}`"),
                });
            }
        }

        Ok(DetectorErrorModel {
            detectors: (max_det + 1) as usize,
            observables: (max_obs + 1) as usize,
            errors,
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
            for &d in &e.dets {
                out.push_str(" D");
                out.push_str(&d.to_string());
                used_det = used_det.max(d as usize + 1);
            }
            for &o in &e.obs {
                out.push_str(" L");
                out.push_str(&o.to_string());
                used_obs = used_obs.max(o as usize + 1);
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
    fn accepts_separable_component_marker() {
        let m = DetectorErrorModel::parse("error(0.1) D0 ^ D1 L0\n").expect("parse");
        assert_eq!(m.errors[0], DemError::new(0.1, vec![0, 1], vec![0]));
    }

    #[test]
    fn target_order_is_normalised() {
        let a = DetectorErrorModel::parse("error(0.2) D5 D1 L1 L0\n").unwrap();
        let b = DetectorErrorModel::parse("error(0.2) D1 D5 L0 L1\n").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn rejects_unsupported_repeat() {
        let err = DetectorErrorModel::parse("repeat 5 {\n").unwrap_err();
        assert!(matches!(err, Error::UnsupportedDem { what, .. } if what == "repeat"));
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
        let dets = if detectors == 0 {
            Just(Vec::new()).boxed()
        } else {
            prop::collection::vec(0u32..detectors as u32, 0..4).boxed()
        };
        let obs = if observables == 0 {
            Just(Vec::new()).boxed()
        } else {
            prop::collection::vec(0u32..observables as u32, 0..2).boxed()
        };
        (0.0001f64..0.5, dets, obs).prop_map(|(p, d, o)| DemError::new(p, d, o))
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
