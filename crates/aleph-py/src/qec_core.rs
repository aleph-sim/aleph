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
    (
        "relay-bp",
        &["legs", "alpha", "gamma_min", "gamma_max", "seed"],
    ),
    (
        "relay-bp-osd",
        &[
            "legs",
            "alpha",
            "gamma_min",
            "gamma_max",
            "seed",
            "osd_order",
        ],
    ),
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
#[derive(Debug)]
pub enum AnyDecoder {
    Mwpm(MwpmDecoder),
    UnionFind(UnionFindDecoder),
    Bp(BpDecoder),
    BpOsd(OsdDecoder),
    Relay(RelayBpDecoder),
    // Boxed: RelayBpOsdDecoder is far larger than the other variants (it embeds a full
    // RelayBpDecoder plus an OsdDecoder), and clippy::large_enum_variant flags the bloat this
    // adds to every `AnyDecoder` value.
    RelayOsd(Box<RelayBpOsdDecoder>),
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
        // Every decoder packs its prediction into a u64 observable mask; past 64 the extra
        // observables would be dropped (BP/UF) or aliased (MWPM shift wrap), silently.
        if dem.observables > 64 {
            return Err(format!(
                "aleph decoders support at most 64 logical observables, got {}",
                dem.observables
            ));
        }
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
            "union-find" => {
                AnyDecoder::UnionFind(UnionFindDecoder::new(dem).map_err(|e| e.to_string())?)
            }
            "union-find-weighted" => AnyDecoder::UnionFind(
                UnionFindDecoder::new_weighted(dem).map_err(|e| e.to_string())?,
            ),
            "bp" => AnyDecoder::Bp(BpDecoder::with_params(
                dem,
                max_iter,
                finite("alpha", p.alpha, BP_ALPHA)?,
            )),
            "bp-osd" => AnyDecoder::BpOsd(OsdDecoder::with_params(
                dem,
                max_iter,
                finite("alpha", p.alpha, OSD_ALPHA)?,
                p.osd_order.unwrap_or(0),
            )),
            "relay-bp" => AnyDecoder::Relay(relay()?),
            "relay-bp-osd" => {
                let alpha = finite("alpha", p.alpha, RELAY_ALPHA)?;
                AnyDecoder::RelayOsd(Box::new(RelayBpOsdDecoder::with_parts(
                    relay()?,
                    OsdDecoder::with_params(dem, DEFAULT_MAX_ITER, alpha, p.osd_order.unwrap_or(0)),
                )))
            }
            other => {
                let names: Vec<&str> = DECODER_NAMES.iter().map(|(n, _)| *n).collect();
                return Err(format!(
                    "unknown decoder `{other}`; valid: {}",
                    names.join(", ")
                ));
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
            AnyDecoder::RelayOsd(d) => d.as_ref(),
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
    if detectors == 0 {
        // rayon's `zip` truncates to the shorter side; an empty `bits` slice (0 detectors) would
        // silently drop every shot instead of decoding `shots` empty syndromes.
        out.par_chunks_mut(observables).for_each(|o| {
            let c = dec.decode(&Syndrome {
                detectors: 0,
                fired: Vec::new(),
            });
            for (dst, &f) in o.iter_mut().zip(&c.observable_flips) {
                *dst = f as u8;
            }
        });
        return out;
    }
    out.par_chunks_mut(observables)
        .zip(bits.par_chunks(detectors))
        .for_each(|(o, row)| {
            let fired = (0..detectors as u32)
                .filter(|&d| row[d as usize] != 0)
                .collect();
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
    if detectors == 0 {
        // Same truncation hazard as `decode_rows`: `ib == 0`, so `packed` is empty and a zip
        // against it would decode zero shots instead of `shots`.
        out.par_chunks_mut(ob).for_each(|o| {
            let c = dec.decode(&Syndrome {
                detectors: 0,
                fired: Vec::new(),
            });
            for (i, &f) in c.observable_flips.iter().enumerate() {
                o[i / 8] |= (f as u8) << (i % 8);
            }
        });
        return out;
    }
    out.par_chunks_mut(ob)
        .zip(packed.par_chunks(ib))
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
            assert!(
                AnyDecoder::build(&dem(), name, &DecoderParams::default()).is_ok(),
                "{name}"
            );
        }
    }

    #[test]
    fn unknown_name_lists_valid_ones() {
        let err = AnyDecoder::build(&dem(), "nope", &DecoderParams::default()).unwrap_err();
        assert!(err.contains("relay-bp") && err.contains("mwpm"), "{err}");
    }

    #[test]
    fn more_than_64_observables_is_rejected() {
        // Every decoder packs observables into a u64; L64 would be silently lost/aliased.
        let ok = DetectorErrorModel::parse("error(0.1) D0 L63\n").unwrap();
        assert_eq!(ok.observables, 64);
        let wide = DetectorErrorModel::parse("error(0.1) D0 L64\n").unwrap();
        assert_eq!(wide.observables, 65);
        for (name, _) in DECODER_NAMES {
            assert!(AnyDecoder::build(&ok, name, &DecoderParams::default()).is_ok());
            let err = AnyDecoder::build(&wide, name, &DecoderParams::default()).unwrap_err();
            assert!(err.contains("at most 64") && err.contains("65"), "{err}");
        }
    }

    #[test]
    fn non_finite_float_param_is_rejected() {
        let p = DecoderParams {
            alpha: Some(f64::NAN),
            ..Default::default()
        };
        assert!(AnyDecoder::build(&dem(), "bp", &p).is_err());
        let p = DecoderParams {
            gamma_max: Some(f64::INFINITY),
            ..Default::default()
        };
        assert!(AnyDecoder::build(&dem(), "relay-bp", &p).is_err());
    }

    #[test]
    fn rows_match_single_decode() {
        let d = AnyDecoder::build(&dem(), "mwpm", &DecoderParams::default()).unwrap();
        // all 8 syndromes over 3 detectors
        let bits: Vec<u8> = (0..8u8)
            .flat_map(|s| (0..3).map(move |i| (s >> i) & 1))
            .collect();
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
        let text = (0..11)
            .map(|i| format!("error(0.05) D{i} D{}\n", (i + 1) % 11))
            .collect::<String>()
            + "error(0.05) D0 L0\nerror(0.05) D5 L1\n";
        let dem = DetectorErrorModel::parse(&text).unwrap();
        let d = AnyDecoder::build(&dem, "bp-osd", &DecoderParams::default()).unwrap();
        let shots = 64;
        let mut bits = vec![0u8; shots * 11];
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        for b in bits.iter_mut() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *b = x.is_multiple_of(5) as u8;
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

    #[test]
    fn zero_detector_model() {
        let dem = DetectorErrorModel::parse("error(0.1) L0\n").unwrap();
        assert_eq!(dem.detectors, 0);
        assert_eq!(dem.observables, 1);
        let d = AnyDecoder::build(&dem, "mwpm", &DecoderParams::default()).unwrap();

        let shots = 5;
        let bits: Vec<u8> = Vec::new();
        let rows = decode_rows(d.get(), &bits, shots, 0, 1);
        assert_eq!(rows.len(), shots);

        let packed: Vec<u8> = Vec::new();
        let pk = decode_packed(d.get(), &packed, shots, 0, 1);
        assert_eq!(pk.len(), shots * packed_len(1));
    }
}
