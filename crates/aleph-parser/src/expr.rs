//! Expression sub-grammar for gate parameters.
//!
//! Supports: `pi`, float literals, `+ - * /`, unary minus, parens.
//! Evaluates to `f64` at parse time per spec § 9. Division by zero
//! and non-finite intermediate results are surfaced as a structured
//! error string that the caller wraps in `ParseErrorKind::BadExpression`.

use nom::branch::alt;
use nom::character::complete::char as ch;
use nom::IResult;
use nom::Parser;

use crate::lexer::{float, ident, skip_ws, Span};

/// Parse a full expression and evaluate to `f64`. Returns either the
/// finite result or a string describing the failure (the caller turns
/// this into `ParseErrorKind::BadExpression`).
pub fn expr(input: Span<'_>) -> IResult<Span<'_>, Result<f64, String>> {
    add(input)
}

fn add(input: Span<'_>) -> IResult<Span<'_>, Result<f64, String>> {
    let (mut input, mut acc) = mul(input)?;
    loop {
        let saved = input;
        let (next, _) = skip_ws(input)?;
        let op_result: IResult<Span<'_>, char> = alt((ch('+'), ch('-'))).parse(next);
        let Ok((next, op)) = op_result else {
            input = saved;
            break;
        };
        let (next, rhs) = mul(next)?;
        acc = match (acc, rhs) {
            (Ok(a), Ok(b)) => {
                let v = if op == '+' { a + b } else { a - b };
                finite(v)
            }
            (Err(e), _) | (_, Err(e)) => Err(e),
        };
        input = next;
    }
    Ok((input, acc))
}

fn mul(input: Span<'_>) -> IResult<Span<'_>, Result<f64, String>> {
    let (mut input, mut acc) = unary(input)?;
    loop {
        let saved = input;
        let (next, _) = skip_ws(input)?;
        let op_result: IResult<Span<'_>, char> = alt((ch('*'), ch('/'))).parse(next);
        let Ok((next, op)) = op_result else {
            input = saved;
            break;
        };
        let (next, rhs) = unary(next)?;
        acc = match (acc, rhs) {
            (Ok(a), Ok(b)) => {
                if op == '/' {
                    if b == 0.0 {
                        Err("division by zero".to_string())
                    } else {
                        finite(a / b)
                    }
                } else {
                    finite(a * b)
                }
            }
            (Err(e), _) | (_, Err(e)) => Err(e),
        };
        input = next;
    }
    Ok((input, acc))
}

fn unary(input: Span<'_>) -> IResult<Span<'_>, Result<f64, String>> {
    let (input, _) = skip_ws(input)?;
    let (input, sign) = match ch::<_, nom::error::Error<Span<'_>>>('-').parse(input) {
        Ok((rest, _)) => (rest, true),
        Err(_) => (input, false),
    };
    let (input, v) = atom(input)?;
    let v = match v {
        Ok(x) if sign => finite(-x),
        other => other,
    };
    Ok((input, v))
}

fn atom(input: Span<'_>) -> IResult<Span<'_>, Result<f64, String>> {
    let (input, _) = skip_ws(input)?;
    // Try paren-wrapped expression first.
    if let Ok((rest, _)) = ch::<_, nom::error::Error<Span<'_>>>('(').parse(input) {
        let (rest, v) = expr(rest)?;
        let (rest, _) = skip_ws(rest)?;
        let (rest, _) = ch::<_, nom::error::Error<Span<'_>>>(')').parse(rest)?;
        return Ok((rest, v));
    }
    // Try `pi`.
    if let Ok((rest, name)) = ident(input) {
        if *name.fragment() == "pi" {
            return Ok((rest, Ok(std::f64::consts::PI)));
        }
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Tag,
        )));
    }
    // Fall through to float.
    let (input, v) = float(input)?;
    Ok((input, finite(v)))
}

fn finite(v: f64) -> Result<f64, String> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!("non-finite expression result ({v})"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(s: &str) -> Result<f64, String> {
        let (_, v) = expr(Span::new(s)).unwrap();
        v
    }

    #[test]
    fn literal_float() {
        assert_eq!(eval("1.5"), Ok(1.5));
    }

    #[test]
    fn pi_substituted() {
        assert!((eval("pi").unwrap() - std::f64::consts::PI).abs() < 1e-15);
    }

    #[test]
    fn pi_div_two() {
        assert!((eval("pi/2").unwrap() - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
    }

    #[test]
    fn precedence_mul_before_add() {
        assert_eq!(eval("2 + 3 * 4"), Ok(14.0));
    }

    #[test]
    fn parens_override() {
        assert_eq!(eval("(2 + 3) * 4"), Ok(20.0));
    }

    #[test]
    fn unary_minus() {
        assert_eq!(eval("-1.5"), Ok(-1.5));
    }

    #[test]
    fn unary_minus_with_paren() {
        assert_eq!(eval("-(1 + 2)"), Ok(-3.0));
    }

    #[test]
    fn left_assoc_subtract() {
        assert_eq!(eval("10 - 3 - 2"), Ok(5.0));
    }

    #[test]
    fn division_by_zero_errors() {
        assert_eq!(eval("1/0"), Err("division by zero".to_string()));
    }

    #[test]
    fn division_by_computed_zero_errors() {
        assert_eq!(eval("1/(2-2)"), Err("division by zero".to_string()));
    }

    #[test]
    fn whitespace_tolerated() {
        assert_eq!(eval(" 2  *  3 "), Ok(6.0));
    }

    #[test]
    fn scientific_literal() {
        assert!((eval("1.5e-3").unwrap() - 1.5e-3).abs() < 1e-15);
    }
}
