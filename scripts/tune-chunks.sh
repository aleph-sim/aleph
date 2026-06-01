# scripts/tune-chunks.sh — P2-04 chunk-size grid sweep.
# Usage: ALEPH_CPU_MODEL=epyc ./scripts/tune-chunks.sh 2>&1 | tee tune-$(hostname).log
# Run ONLY on a verified-idle box (uptime ~0; no cargo bench / bencher run).
#
# Each grid point is a separate `cargo bench` invocation (a NEW process) because
# tuning::resolve_policy caches ALEPH_PAR_MIN_AMPS / ALEPH_PAR_GRAIN in OnceLocks
# that are read exactly once per process. Varying the knobs within one process
# would not take effect after the first read — separate invocations guarantee
# each grid point sees its own fresh env values.
set -euo pipefail

GATES=("h" "zdiag" "cnot" "cphase")          # high-traffic Tier-1 classes
TARGETS=(1 12 24)                            # Low / Mid / High position buckets
MIN_AMPS=(65536 131072 262144 524288 1048576)
GRAINS=(16 32 64 128 256 512)
N="${ALEPH_TUNE_N:-25}"

echo "# host=$(hostname) cpu_model=${ALEPH_CPU_MODEL:-auto} n=$N"
echo "# gate target min_amps grain median"
for g in "${GATES[@]}"; do
  for t in "${TARGETS[@]}"; do
    for ma in "${MIN_AMPS[@]}"; do
      for gr in "${GRAINS[@]}"; do
        # Tolerate a single bad cell (compile/panic/no-match) so it can't
        # abort the whole multi-hundred-cell sweep under `set -e`.
        out=$(ALEPH_TUNE_GATE="$g" ALEPH_TUNE_TARGET="$t" ALEPH_TUNE_N="$N" \
              ALEPH_PAR_MIN_AMPS="$ma" ALEPH_PAR_GRAIN="$gr" \
              RUSTFLAGS="-C target-cpu=native" \
              cargo bench -p aleph-sv --features internal-bench --bench chunk_tune \
                -- --warm-up-time 1 --measurement-time 3 --noplot 2>/dev/null) \
          || { echo "$g $t $ma $gr FAILED"; continue; }
        # criterion line: "time:   [<lo> <unit> <median> <unit> <hi> <unit>]"
        # after grep, awk fields: $2=[<lo> $3=<unit> $4=<median> $5=<unit> ...
        med=$(echo "$out" | grep -oE 'time:[[:space:]]*\[[^]]+\]' | head -1 | awk '{print $4, $5}') || true
        med=${med:-NA}
        echo "$g $t $ma $gr $med"
      done
    done
  done
done
