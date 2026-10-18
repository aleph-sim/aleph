//! Cross-version exact DEM/syndrome/prediction corpus, outside timing. Usage: NEW_DIRECTORY.
//! Uses only APIs present in the original #511 baseline. Integer/float bits are explicit LE.
use aleph_qec::{build_dem, Decoder, MwpmDecoder, SurfaceCode, Syndrome};
use std::io::Write;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = std::env::args().nth(1).ok_or("need new output directory")?;
    std::fs::create_dir(&directory)?;
    println!("distance,rate,detectors,observables,shots");
    for d in [7usize, 9, 11, 13, 15, 17] {
        for p in [0.03, 0.003] {
            let exp = SurfaceCode::new(d).memory_z_experiment(d);
            let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p))?;
            let decoder = MwpmDecoder::new(&dem)?;
            let artifact =
                |suffix: &str| std::path::Path::new(&directory).join(format!("d{d}-p{p}.{suffix}"));
            let mut model = std::fs::File::create(artifact("dem.bin"))?;
            for n in [dem.detectors, dem.observables, dem.errors.len()] {
                model.write_all(&(n as u64).to_le_bytes())?;
            }
            for error in &dem.errors {
                model.write_all(&error.prob.to_bits().to_le_bytes())?;
                for items in [&error.dets, &error.obs] {
                    model.write_all(&(items.len() as u64).to_le_bytes())?;
                    for &v in items {
                        model.write_all(&(v as u64).to_le_bytes())?;
                    }
                }
            }
            let mut input = std::fs::File::create(artifact("syndromes.bin"))?;
            let mut output = std::fs::File::create(artifact("corrections.bin"))?;
            for shot in 0..256u64 {
                let mut z =
                    (0x5EED ^ d as u64).wrapping_add(shot.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let mut det = vec![false; dem.detectors];
                for error in &dem.errors {
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                    z ^= z >> 31;
                    if (z >> 11) as f64 / ((1u64 << 53) as f64) < error.prob {
                        for &u in &error.dets {
                            det[u as usize] ^= true;
                        }
                    }
                }
                let syn = Syndrome::from_bits(&det);
                input.write_all(&(syn.fired.len() as u64).to_le_bytes())?;
                for &v in &syn.fired {
                    input.write_all(&v.to_le_bytes())?;
                }
                for flip in decoder.decode(&syn).observable_flips {
                    output.write_all(&[u8::from(flip)])?;
                }
            }
            println!("{d},{p},{},{},256", dem.detectors, dem.observables);
        }
    }
    Ok(())
}
