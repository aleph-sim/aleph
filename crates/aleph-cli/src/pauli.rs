//! `--expectation` argument parser.  See spec §4.4.
//!
//! Grammar:
//! ```text
//! pauli_arg     = [ coeff "*" ] pauli_string
//! coeff         = signed-f64 (parsed via str::parse::<f64>)
//! pauli_string  = { 'I' | 'X' | 'Y' | 'Z' }+   (q0 = leftmost char)
//! ```

use aleph_core::{Pauli, PauliString};

/// Parse a `--expectation` argument like `"ZZ"`, `"1.5*ZZ"`, or
/// `"-0.5*X"` into a [`PauliString`].
///
/// Does not bounds-check qubit indices against any circuit — the
/// caller knows the circuit's `num_qubits` and is responsible for
/// rejecting out-of-range Pauli terms.
pub fn parse_pauli_arg(raw: &str) -> Result<PauliString, PauliArgError> {
    // Split optional "coeff*" prefix from the Pauli body. We split on
    // the FIRST '*' so a malformed input like "ZZ*1.5" surfaces as an
    // InvalidChar('*') in the body half rather than a confusing
    // coefficient-parse error.
    let (coeff, body) = match raw.find('*') {
        Some(idx) => {
            let coeff_str = &raw[..idx];
            let body = &raw[idx + 1..];
            // The "*" is only treated as a separator if the left side
            // parses as f64.  Without this, "ZZ*1.5" would be split as
            // coeff="ZZ" and produce a confusing BadCoeff error.
            match coeff_str.parse::<f64>() {
                Ok(c) => (c, body),
                Err(_) => (1.0, raw),
            }
        }
        None => (1.0, raw),
    };
    if body.is_empty() {
        return Err(PauliArgError::Empty);
    }
    let mut terms = Vec::with_capacity(body.len());
    for (i, ch) in body.chars().enumerate() {
        let p = match ch {
            'I' => Pauli::I,
            'X' => Pauli::X,
            'Y' => Pauli::Y,
            'Z' => Pauli::Z,
            other => return Err(PauliArgError::InvalidChar { ch: other }),
        };
        terms.push((i as u32, p));
    }
    // PauliString::new sorts, dedupes (none possible here), drops I,
    // and rejects non-finite coefficient.
    Ok(PauliString::new(coeff, terms)?)
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum PauliArgError {
    #[error("empty Pauli string")]
    Empty,

    #[error("invalid character {ch:?} in Pauli string (allowed: I, X, Y, Z)")]
    InvalidChar { ch: char },

    #[error("could not parse coefficient {coeff:?} as f64: {source_msg}")]
    BadCoeff { coeff: String, source_msg: String },

    #[error("PauliString construction failed: {0}")]
    PauliConstruction(#[from] aleph_core::PauliError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_core::Pauli;
    use proptest::prelude::*;

    fn terms_only(ps: &PauliString) -> Vec<(u32, Pauli)> {
        ps.terms.clone()
    }

    #[test]
    fn parse_zz() {
        let ps = parse_pauli_arg("ZZ").unwrap();
        assert_eq!(ps.coefficient, 1.0);
        assert_eq!(terms_only(&ps), vec![(0, Pauli::Z), (1, Pauli::Z)]);
    }

    #[test]
    fn parse_with_coeff() {
        let ps = parse_pauli_arg("1.5*ZZ").unwrap();
        assert_eq!(ps.coefficient, 1.5);
        assert_eq!(terms_only(&ps), vec![(0, Pauli::Z), (1, Pauli::Z)]);
    }

    #[test]
    fn parse_negative_coeff() {
        let ps = parse_pauli_arg("-0.5*X").unwrap();
        assert_eq!(ps.coefficient, -0.5);
        assert_eq!(terms_only(&ps), vec![(0, Pauli::X)]);
    }

    #[test]
    fn parse_drops_identity() {
        // "IXZI" → X on q1, Z on q2; I terms are removed by PauliString::new.
        let ps = parse_pauli_arg("IXZI").unwrap();
        assert_eq!(ps.coefficient, 1.0);
        assert_eq!(terms_only(&ps), vec![(1, Pauli::X), (2, Pauli::Z)]);
    }

    #[test]
    fn parse_rejects_empty() {
        assert_eq!(parse_pauli_arg(""), Err(PauliArgError::Empty));
    }

    #[test]
    fn parse_rejects_invalid_char() {
        match parse_pauli_arg("ABC") {
            Err(PauliArgError::InvalidChar { ch: 'A' }) => {}
            other => panic!("expected InvalidChar('A'), got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_misplaced_coeff() {
        // "ZZ*1.5" — coefficient must precede '*'; the trailing '*1.5'
        // is interpreted as a Pauli body and rejected at the first
        // non-IXYZ char ('*').
        match parse_pauli_arg("ZZ*1.5") {
            Err(PauliArgError::InvalidChar { ch: '*' }) => {}
            other => panic!("expected InvalidChar('*'), got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_empty_after_coeff() {
        assert_eq!(parse_pauli_arg("1.5*"), Err(PauliArgError::Empty));
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

        /// Round-trip a Z-only Pauli string through display → parse →
        /// equal terms.  Pins the parser ↔ Display invariant so a
        /// future change to either surface is loud, not silent.
        #[test]
        fn z_only_round_trip(mask in any::<u32>(), n in 1u32..=6) {
            // Build "Z...Z..." string from low n bits of mask.
            let s: String = (0..n)
                .map(|q| if (mask >> q) & 1 == 1 { 'Z' } else { 'I' })
                .collect();
            // All-identity is a legal Pauli body that the parser accepts;
            // PauliString::new() returns terms=[] in that case.  Both
            // sides agree.
            let ps = parse_pauli_arg(&s).unwrap();
            for (q, p) in &ps.terms {
                prop_assert!(*p == Pauli::Z);
                prop_assert!(*q < n);
            }
            // Reconstruct the string from parsed terms and parse again.
            let mut chars = vec!['I'; n as usize];
            for (q, _) in &ps.terms {
                chars[*q as usize] = 'Z';
            }
            let reparsed = parse_pauli_arg(&chars.iter().collect::<String>()).unwrap();
            prop_assert_eq!(reparsed.terms, ps.terms);
        }
    }
}
