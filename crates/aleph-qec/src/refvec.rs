//! Binary `.ref` companion-file codec for the on-silicon LER campaigns (Q7-06 AC-2, Q7-07).
//!
//! A campaign ships two blobs: `<prefix>.syn` (raw syndrome words the DMA streams into the PL) and
//! `<prefix>.ref` (what the software golden decided, for the host to compare against). v1 held two
//! u16 per shot — `true_obs`, `sw_obs` — and nothing about *validity*, so the board could not check
//! its own `valid_flag` against the golden's. v2 adds a third u16 and a magic header.
//!
//! The header exists because #478 cost a full campaign re-run: a golden silently paired with a
//! bitstream built at another `p`. A file that cannot be identified must not be guessed at.

use std::io::{Read, Write};

/// First header word. Chosen above `0x0FFF` so it can never be a legacy v1 file's leading
/// `true_obs` (the gross code has 12 observables), making v1 detectable rather than misparsed.
pub const REF_MAGIC: u16 = 0xA1E7;
/// Current format version.
pub const REF_VERSION: u16 = 2;
/// u16 words per shot in the payload: `true_obs`, `sw_obs`, `meta`.
pub const REF_WORDS_PER_SHOT: u16 = 3;

const HEADER_WORDS: usize = 4;
/// `REF_WORDS_PER_SHOT` as a `usize`, for use as a const-generic chunk width.
const WORDS_PER_SHOT: usize = REF_WORDS_PER_SHOT as usize;

/// One shot's software-golden record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RefRecord {
    /// Truth observable-flip mask sampled with the shot.
    pub true_obs: u16,
    /// The software golden's predicted observable-flip mask.
    pub sw_obs: u16,
    /// Whether some relay-BP leg found a syndrome-valid `ê` — the software twin of the RTL
    /// `valid_flag` (`hw/bp_relay_banked.sv:968`).
    pub valid: bool,
    /// 1-based global iteration index where a first-valid stop would land, or the full schedule
    /// length if none converged (`FixedRelayBp::iters_to_valid`).
    pub iters: u16,
}

fn invalid(msg: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg.to_string())
}

/// Write the v2 header followed by one 3-word record per shot.
pub fn write_ref<W: Write>(w: &mut W, recs: &[RefRecord]) -> std::io::Result<()> {
    for word in [REF_MAGIC, REF_VERSION, REF_WORDS_PER_SHOT, 0] {
        w.write_all(&word.to_le_bytes())?;
    }
    for r in recs {
        let meta = (u16::from(r.valid) << 15) | (r.iters & 0x7FFF);
        for word in [r.true_obs, r.sw_obs, meta] {
            w.write_all(&word.to_le_bytes())?;
        }
    }
    Ok(())
}

/// Read a v2 file. Rejects a missing/unknown magic (i.e. a legacy v1 file), an unknown version,
/// an unexpected record width, and a truncated payload — never guesses.
pub fn read_ref<R: Read>(r: &mut R) -> std::io::Result<Vec<RefRecord>> {
    let mut bytes = Vec::new();
    r.read_to_end(&mut bytes)?;
    if bytes.len() % 2 != 0 {
        return Err(invalid("ref: odd byte length, not a u16 stream"));
    }
    // Odd lengths are rejected above, so `as_chunks` leaves no remainder here.
    let words: Vec<u16> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect();
    if words.len() < HEADER_WORDS {
        return Err(invalid("ref: shorter than the header"));
    }
    if words[0] != REF_MAGIC {
        return Err(invalid(
            "ref: bad magic — this is a legacy v1 file (no header); regenerate it with silvectors",
        ));
    }
    if words[1] != REF_VERSION {
        return Err(invalid("ref: unsupported version"));
    }
    if words[2] != REF_WORDS_PER_SHOT {
        return Err(invalid("ref: unexpected words-per-shot"));
    }
    let payload = &words[HEADER_WORDS..];
    // `as_chunks` hands back the trailing partial record, which is exactly the truncation signal.
    let (records, tail) = payload.as_chunks::<WORDS_PER_SHOT>();
    if !tail.is_empty() {
        return Err(invalid("ref: truncated payload"));
    }
    Ok(records
        .iter()
        .map(|c| RefRecord {
            true_obs: c[0],
            sw_obs: c[1],
            valid: c[2] >> 15 == 1,
            iters: c[2] & 0x7FFF,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<RefRecord> {
        vec![
            RefRecord {
                true_obs: 0x000,
                sw_obs: 0x000,
                valid: true,
                iters: 3,
            },
            RefRecord {
                true_obs: 0x001,
                sw_obs: 0x801,
                valid: false,
                iters: 60,
            },
            RefRecord {
                true_obs: 0xFFF,
                sw_obs: 0xFFF,
                valid: true,
                iters: 1,
            },
        ]
    }

    #[test]
    fn test_round_trip_preserves_every_field() {
        let recs = sample();
        let mut buf = Vec::new();
        write_ref(&mut buf, &recs).expect("write");
        let back = read_ref(&mut buf.as_slice()).expect("read");
        assert_eq!(back, recs);
    }

    #[test]
    fn test_header_is_four_words_and_payload_is_three_per_shot() {
        let mut buf = Vec::new();
        write_ref(&mut buf, &sample()).expect("write");
        assert_eq!(buf.len(), 2 * (4 + 3 * 3));
        assert_eq!(u16::from_le_bytes([buf[0], buf[1]]), REF_MAGIC);
        assert_eq!(u16::from_le_bytes([buf[2], buf[3]]), REF_VERSION);
    }

    #[test]
    fn test_legacy_v1_file_is_rejected_not_misparsed() {
        // v1 layout: two u16 per shot, no header. First word is a 12-bit observable mask.
        let legacy: Vec<u8> = [0x0001u16, 0x0001, 0x0002, 0x0002]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let err = read_ref(&mut legacy.as_slice()).expect_err("legacy must be rejected");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_unknown_version_is_rejected() {
        let mut buf = Vec::new();
        write_ref(&mut buf, &sample()).expect("write");
        buf[2..4].copy_from_slice(&99u16.to_le_bytes());
        let err = read_ref(&mut buf.as_slice()).expect_err("bad version must be rejected");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_payload_truncated_by_whole_words_is_rejected() {
        // Dropping one whole u16 keeps the byte length even, so this is the only case that
        // reaches the partial-record check rather than the odd-byte guard below it.
        let mut buf = Vec::new();
        write_ref(&mut buf, &sample()).expect("write");
        buf.truncate(buf.len() - 2);
        let err = read_ref(&mut buf.as_slice()).expect_err("partial record must be rejected");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_truncated_payload_is_rejected() {
        let mut buf = Vec::new();
        write_ref(&mut buf, &sample()).expect("write");
        buf.truncate(buf.len() - 3);
        let err = read_ref(&mut buf.as_slice()).expect_err("truncated must be rejected");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }
}
