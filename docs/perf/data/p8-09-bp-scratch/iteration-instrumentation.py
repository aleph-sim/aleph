import pathlib,json,subprocess,hashlib
repo=pathlib.Path('/tmp/aleph-559-iterations')
f=repo/'crates/aleph-cuda/src/qec/bp.cu';s=f.read_text().replace('    const double INF = pos_inf();','    u32 completed_iterations = 0;\n    const double INF = pos_inf();').replace('    for (u32 it = 0; it < max_iter; ++it) {','    for (u32 it = 0; it < max_iter; ++it) {\n        completed_iterations = it + 1;').replace('    out_mask[shot] = mask;','    out_mask[shot] = ((u64)completed_iterations << 32) | mask;');f.write_text(s)
p=repo/'crates/aleph-cuda/examples/bp_layout_probe.rs';s=p.read_text();needle='    gpu.synchronize()?;';assert needle in s;s=s.replace(needle,needle+'''\n    eprintln!("iteration_counts={:?}", reference.iter().map(|x| (x >> 32) as u32).collect::<Vec<_>>());
    eprintln!("correction_checksum={}", reference.iter().fold(0u64, |acc, x| acc.wrapping_mul(31).wrapping_add(x & 0xffffffff)));''',1);p.write_text(s)
