//! Lexical primitives: `Span` alias, whitespace/comment skipper, and
//! small token combinators (idents, integers, floats, punctuation,
//! string literals). All combinators operate on
//! [`nom_locate::LocatedSpan<&str>`] so the parser keeps line/col info
//! and can build [`crate::error::ParseError`]s with accurate positions.

use nom::IResult;
use nom::Parser;
use nom::branch::alt;
use nom::bytes::complete::{is_not, tag, take_until};
use nom::character::complete::{multispace1, satisfy};
use nom::combinator::{opt, recognize, value};
use nom::multi::many0;
use nom::bytes::complete::take_while1;
use nom::character::complete::char as ch;
use nom::sequence::{pair, tuple};
use nom_locate::LocatedSpan;

pub type Span<'a> = LocatedSpan<&'a str>;

/// Skip ASCII whitespace, `// line comments`, and `/* block comments */`
/// (non-nesting). Returns success even if nothing was consumed.
pub fn skip_ws(input: Span<'_>) -> IResult<Span<'_>, ()> {
    value(
        (),
        many0(alt((
            value((), multispace1),
            value((), line_comment),
            value((), block_comment),
        ))),
    )
    .parse(input)
}

fn line_comment(input: Span<'_>) -> IResult<Span<'_>, ()> {
    let (input, _) = tag("//").parse(input)?;
    let (input, _) = opt(is_not("\n\r")).parse(input)?;
    Ok((input, ()))
}

fn block_comment(input: Span<'_>) -> IResult<Span<'_>, ()> {
    let (input, _) = tag("/*").parse(input)?;
    let (input, _) = take_until("*/").parse(input)?;
    let (input, _) = tag("*/").parse(input)?;
    Ok((input, ()))
}

/// Wrap a parser so it skips leading whitespace/comments.
pub fn ws<'a, F, O>(mut inner: F) -> impl FnMut(Span<'a>) -> IResult<Span<'a>, O>
where
    F: Parser<Span<'a>, O, nom::error::Error<Span<'a>>>,
{
    move |input| {
        let (input, _) = skip_ws(input)?;
        inner.parse(input)
    }
}

/// Recognise (but don't consume) the start of an identifier.
pub(crate) fn ident_start(input: Span<'_>) -> IResult<Span<'_>, char> {
    satisfy(|c: char| c.is_ascii_alphabetic() || c == '_').parse(input)
}

/// Recognise (but don't consume) an identifier continuation char.
pub(crate) fn ident_cont(input: Span<'_>) -> IResult<Span<'_>, char> {
    satisfy(|c: char| c.is_ascii_alphanumeric() || c == '_').parse(input)
}

/// Parse an identifier: `[A-Za-z_][A-Za-z0-9_]*`. Returns the span.
pub fn ident(input: Span<'_>) -> IResult<Span<'_>, Span<'_>> {
    recognize(pair(ident_start, many0(ident_cont))).parse(input)
}

/// Parse a non-negative decimal integer; returns the parsed `u32`.
/// The caller is responsible for context-bounded ranges; this is the
/// raw lexer primitive.
pub fn uint(input: Span<'_>) -> IResult<Span<'_>, u32> {
    let (input, digits) = take_while1(|c: char| c.is_ascii_digit()).parse(input)?;
    let value: u32 = digits.fragment().parse().map_err(|_| {
        nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Digit))
    })?;
    Ok((input, value))
}

/// Parse a floating-point literal: `123`, `123.456`, `.5`, `1e10`,
/// `1.5e-3`. Returns the parsed `f64`.
pub fn float(input: Span<'_>) -> IResult<Span<'_>, f64> {
    let (rest, lit) = recognize(tuple((
        opt(alt((ch('+'), ch('-')))),
        alt((
            // forms like 1.5 or 1.5e-3 or 1.
            recognize(tuple((
                take_while1(|c: char| c.is_ascii_digit()),
                ch('.'),
                opt(take_while1(|c: char| c.is_ascii_digit())),
                opt(exponent),
            ))),
            // forms like .5 or .5e10
            recognize(tuple((
                ch('.'),
                take_while1(|c: char| c.is_ascii_digit()),
                opt(exponent),
            ))),
            // forms like 1 or 1e10 (no dot)
            recognize(tuple((
                take_while1(|c: char| c.is_ascii_digit()),
                exponent,
            ))),
        )),
    )))
    .parse(input)?;
    let value: f64 = lit.fragment().parse().map_err(|_| {
        nom::Err::Error(nom::error::Error::new(rest, nom::error::ErrorKind::Float))
    })?;
    Ok((rest, value))
}

fn exponent(input: Span<'_>) -> IResult<Span<'_>, Span<'_>> {
    recognize(tuple((
        alt((ch('e'), ch('E'))),
        opt(alt((ch('+'), ch('-')))),
        take_while1(|c: char| c.is_ascii_digit()),
    )))
    .parse(input)
}

/// Parse a `"..."` string literal; returns the inner text (no escape handling).
pub fn string_literal(input: Span<'_>) -> IResult<Span<'_>, Span<'_>> {
    let (input, _) = ch('"').parse(input)?;
    let (input, body) = take_until("\"").parse(input)?;
    let (input, _) = ch('"').parse(input)?;
    Ok((input, body))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(s: &str) -> Span<'_> {
        Span::new(s)
    }

    #[test]
    fn skip_ws_consumes_spaces_tabs_newlines() {
        let (rest, _) = skip_ws(span(" \t\n  x")).unwrap();
        assert_eq!(*rest.fragment(), "x");
    }

    #[test]
    fn skip_ws_consumes_line_comment() {
        let (rest, _) = skip_ws(span("// blah\nx")).unwrap();
        assert_eq!(*rest.fragment(), "x");
    }

    #[test]
    fn skip_ws_consumes_block_comment() {
        let (rest, _) = skip_ws(span("/* multi\nline */x")).unwrap();
        assert_eq!(*rest.fragment(), "x");
    }

    #[test]
    fn skip_ws_consumes_mixed() {
        let (rest, _) = skip_ws(span("  // x\n  /* y */\n  z")).unwrap();
        assert_eq!(*rest.fragment(), "z");
    }

    #[test]
    fn skip_ws_succeeds_on_empty_input() {
        let (rest, _) = skip_ws(span("x")).unwrap();
        assert_eq!(*rest.fragment(), "x");
    }

    #[test]
    fn ident_parses_simple() {
        let (rest, name) = ident(span("foo_bar123 q")).unwrap();
        assert_eq!(*name.fragment(), "foo_bar123");
        assert_eq!(*rest.fragment(), " q");
    }

    #[test]
    fn ident_rejects_leading_digit() {
        assert!(ident(span("1foo")).is_err());
    }

    #[test]
    fn uint_parses() {
        let (rest, n) = uint(span("42 rest")).unwrap();
        assert_eq!(n, 42);
        assert_eq!(*rest.fragment(), " rest");
    }

    #[test]
    fn uint_zero_works() {
        let (_, n) = uint(span("0;")).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn float_parses_decimal() {
        let (_, v) = float(span("1.5;")).unwrap();
        assert_eq!(v, 1.5);
    }

    #[test]
    fn float_parses_scientific() {
        let (_, v) = float(span("1.5e-3;")).unwrap();
        assert!((v - 1.5e-3).abs() < 1e-12);
    }

    #[test]
    fn float_parses_leading_dot() {
        let (_, v) = float(span(".5;")).unwrap();
        assert_eq!(v, 0.5);
    }

    #[test]
    fn float_parses_no_dot_with_exponent() {
        let (_, v) = float(span("1e10;")).unwrap();
        assert_eq!(v, 1e10);
    }

    #[test]
    fn float_parses_signed() {
        let (_, v) = float(span("-1.5;")).unwrap();
        assert_eq!(v, -1.5);
    }

    #[test]
    fn string_literal_parses_simple() {
        let (rest, body) = string_literal(span(r#""stdgates.inc";"#)).unwrap();
        assert_eq!(*body.fragment(), "stdgates.inc");
        assert_eq!(*rest.fragment(), ";");
    }
}
