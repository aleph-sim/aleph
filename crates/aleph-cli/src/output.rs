//! Formatters for counts / statevector / expectation / bench.  See
//! spec §4.3 and §6.  Each formatter writes to `&mut dyn Write` so
//! unit tests can capture the exact bytes.

use std::collections::BTreeMap;
use std::io::{self, Write};
use std::time::Duration;

use aleph_core::Complex;

/// Histogram `shots` and print ascending by basis index. Zero-count
/// outcomes are elided so a uniform 10-qubit distribution doesn't
/// dump 1024 lines.  `seed_label` is either "seed=N" (with --seed)
/// or "seed=entropy" (without).
pub fn format_counts<W: Write>(
    out: &mut W,
    shots: &[u64],
    total: u32,
    num_qubits: u32,
    seed_label: &str,
) -> io::Result<()> {
    let width = num_qubits as usize;
    let mut hist: BTreeMap<u64, u64> = BTreeMap::new();
    for s in shots {
        *hist.entry(*s).or_insert(0) += 1;
    }
    writeln!(out, "counts ({total} shots, {seed_label}):")?;
    let total_f = total as f64;
    for (idx, count) in &hist {
        let prob = *count as f64 / total_f;
        let idx_us = *idx as usize;
        writeln!(out, "  |{idx_us:0width$b}⟩  {count}  ({prob:.4})", width = width)?;
    }
    Ok(())
}

/// Print every amplitude one per line with `{:+.16e}` precision.
/// Caller has already verified `num_qubits <= 10` (or that the user
/// passed --force-statevector).
pub fn format_statevector<W: Write>(
    out: &mut W,
    amps: &[Complex],
    num_qubits: u32,
) -> io::Result<()> {
    let width = num_qubits as usize;
    let dim = amps.len();
    writeln!(out, "statevector ({num_qubits} qubits, {dim} amplitudes):")?;
    for (i, a) in amps.iter().enumerate() {
        let p = a.norm_sqr();
        writeln!(
            out,
            "  |{i:0width$b}⟩   {:+.16e} {:+.16e}i  |a|² = {p:.6}",
            a.re,
            a.im,
            width = width,
        )?;
    }
    Ok(())
}

/// Print a single expectation value, echoing the raw `--expectation`
/// argument so the user can grep for it.
pub fn format_expectation<W: Write>(out: &mut W, pauli_raw: &str, value: f64) -> io::Result<()> {
    writeln!(out, "  {pauli_raw}    {value:+.16}")
}

/// Bench section header — emit once per bench invocation before the
/// per-phase lines.
pub fn format_bench_header<W: Write>(
    out: &mut W,
    qasm_name: &str,
    num_qubits: u32,
) -> io::Result<()> {
    writeln!(out, "bench {qasm_name} (n={num_qubits}):")
}

/// Per-phase line.  Use `Duration`'s Debug formatter, which
/// auto-selects µs / ms / s.
pub fn format_bench_phase<W: Write>(out: &mut W, name: &str, d: Duration) -> io::Result<()> {
    writeln!(out, "  {name:<13} {d:.1?}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capture(f: impl FnOnce(&mut Vec<u8>) -> io::Result<()>) -> String {
        let mut buf = Vec::new();
        f(&mut buf).unwrap();
        String::from_utf8(buf).unwrap()
    }

    #[test]
    fn format_counts_zero_count_elided() {
        // 4 shots, only outcomes 0 and 3 produced — outcomes 1 and 2
        // must NOT appear in output.
        let s = capture(|out| format_counts(out, &[0, 0, 3, 3], 4, 2, "seed=0"));
        assert!(s.contains("|00⟩  2"));
        assert!(s.contains("|11⟩  2"));
        assert!(!s.contains("|01⟩"));
        assert!(!s.contains("|10⟩"));
    }

    #[test]
    fn format_counts_ordering_ascending() {
        let s = capture(|out| format_counts(out, &[3, 3, 0, 0, 1], 5, 2, "seed=7"));
        let pos_00 = s.find("|00⟩").unwrap();
        let pos_01 = s.find("|01⟩").unwrap();
        let pos_11 = s.find("|11⟩").unwrap();
        assert!(pos_00 < pos_01 && pos_01 < pos_11);
    }

    #[test]
    fn format_counts_header_contains_seed_label() {
        let s = capture(|out| format_counts(out, &[0], 1, 1, "seed=entropy"));
        assert!(s.contains("counts (1 shots, seed=entropy):"));
    }

    #[test]
    fn format_statevector_two_qubits_bell() {
        let inv = std::f64::consts::FRAC_1_SQRT_2;
        let amps = [
            Complex::new(inv, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(inv, 0.0),
        ];
        let s = capture(|out| format_statevector(out, &amps, 2));
        assert!(s.contains("statevector (2 qubits, 4 amplitudes):"));
        assert!(s.contains("|00⟩"));
        assert!(s.contains("|11⟩"));
        // |a|² = 0.500000 on the populated basis states.
        assert!(s.contains("|a|² = 0.500000"));
    }

    #[test]
    fn format_expectation_emits_pauli_raw() {
        let s = capture(|out| format_expectation(out, "ZZ", 1.0));
        assert!(s.contains("ZZ"));
        assert!(s.contains("+1.0000000000000000"));
    }

    #[test]
    fn format_bench_phase_includes_name_and_duration() {
        let s = capture(|out| format_bench_phase(out, "parse", Duration::from_micros(45)));
        assert!(s.contains("parse"));
        assert!(s.contains("µs") || s.contains("us"));
    }

    #[test]
    fn format_bench_header_includes_qasm_and_n() {
        let s = capture(|out| format_bench_header(out, "bell.qasm", 2));
        assert!(s.contains("bench bell.qasm"));
        assert!(s.contains("n=2"));
    }
}
