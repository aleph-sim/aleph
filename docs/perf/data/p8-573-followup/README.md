# #573 CPU follow-up receipts

Final result: `bounded-campaign/clean-summary.csv`, `clean-criterion.json`
(all 84 accepted sample/estimate records), `clean-events.jsonl` (every attempt),
and `attempt-logs.txt` (original per-attempt stdout with filename headers).
`provenance.json` records matching measured source and ELF SHA256s.
`bounded-all-tests.log` is the final 518-test AVX512 run; `bounded-oracles.log`
is the final 28-test focused subset. `bounded-allocations.csv` is the warmed
public gate allocation check; `bounded-budget.csv` is the final noisy heap plan
check. Baseline negative allocation output is retained separately.

Sources: baseline main 64b729f6d7f000d0640439bc4c80ae23cf9a1405, with the same
fused_validation.rs harness and its Cargo bench registration added. Candidate is
integration d367ef54cceaa4eccf0a50f9ef5ffeed2ca434a9 plus bounded.patch. Both builds
use fresh independent target directories. No native flags or extra features are
used in Criterion; internal-bench enables SIMD oracle access in release tests.

Reproduce with cargo bench --locked -p aleph-sv --bench fused_validation --no-run
in each source; copy the emitted executables to a fresh artifact directory as
bench-base and bench-candidate. Copy affinity.txt / core-policy.json from the
final campaign, then run:

    python3 scripts/python/bench_noisy_trajectories_guard.py /absolute/artifacts --suite fused
    cargo test --locked --release -p aleph-sv --features internal-bench --lib --tests
    cargo run --locked --release -p aleph-sv --example trajectory_budget -- --fused-kernels
    cargo run --locked --release -p aleph-sv --example trajectory_budget

All intermediate campaigns remain: stack-campaign (stack scratch only),
final-campaign (index insertion with runtime-width FP32 scratch), sized-campaign
(fixed FP32 input width), bitcache-campaign (exact bitwise cache comparison),
and bounded-campaign (final exact matrix-array bound). sized-probe and
bounded-probe are targeted diagnostics and do not replace complete matrices.
Each candidate patch is against the same integration commit; patch names match
stage names except final-campaign=index.patch and sized-campaign=sized.patch.
The final kernel/query/validator fingerprints equal the checked-out source.

The host is shared; the core/SMT overlap policy is not global isolation and can
miss brief activity/cache/frequency effects. Do not advertise sub-5% speedups.
Bounds are requested heap/buffer accounting, not RSS. Every raw attempt's metadata
is retained, including rejected attempts; accepted repeats are chosen solely by
the fixed CPU-overlap/exit-status policy. Log trailing blank lines and CRLF were
normalized; attempt log files are concatenated without dropping their content.
