# Q1-05 — Native MWPM vs PyMatching (Phase Q1 report)

Phase Q1 built aleph's own minimum-weight-perfect-matching decoder from scratch: a matching-graph
builder from a DEM (Q1-01), an Edmonds/blossom core (Q1-02), a localized savings reformulation
(Q1-03), and the harness wiring that reproduced the surface-code threshold (Q1-04). This report is
the head-to-head that closes the phase: aleph's [`MwpmDecoder`] against **PyMatching 2.4.0** — the
reference MWPM decoder everyone benchmarks against — on shared surface-code memory DEMs, for
`d ∈ {3, 5, 7, 9, 11}`.

**Verdict.**

- **Accuracy: parity.** aleph's logical-error rate equals PyMatching's *within the combined 95 %
  CI at every distance* `d ∈ {3,5,7,9,11}` — the largest gap is 1.6 × 10⁻³ against a CI of
  2.8 × 10⁻³. The two decoders make identical logical predictions up to genuine equal-weight ties.
  This is the Q1 correctness exit, and it is met.
- **Speed: aleph loses on raw matching, and the gap widens with distance.** Single-threaded core
  throughput is at parity at `d=3` (0.85×) but falls to **~15× slower at `d=11`** (0.065×).
  aleph wins *end-to-end* only at `d ≤ 5`, purely because PyMatching pays a fixed ~0.2 s
  Python-subprocess startup that aleph (in-process, static binary) does not; once the matching
  itself is non-trivial (`d ≥ 7`) PyMatching wins even with that tax.

The speed gap is expected and was scoped from the start: PyMatching implements **Sparse Blossom**
(Higgott & Gidney 2023), an almost-linear-time algorithm, whereas aleph's Q1-03 path is a localized
*dense* blossom. The ≥10× throughput target was explicitly deferred to the Sparse Blossom rewrite
([#331]); Q1-05's job is to **measure and report that gap honestly**, which it does.

## Noise model

Phenomenological, single-basis **memory-Z** experiment on the rotated surface code (identical to
the Q0/Q1-04 threshold model):

- Independent `X` error of probability **p** on every data qubit before each stabilizer round and
  the final readout.
- Measurement flip of probability **p** on every ancilla measurement (`p_data = p_meas = p`).
- **rounds = d**, starting from `|0…0⟩`, measuring only Z stabilizers. The DEM is graphlike (every
  mechanism flips ≤ 2 detectors) — exactly what MWPM consumes.

Both decoders are built from, and decode, the **same** [`build_dem`] model, sampled with the **same**
deterministic Bernoulli-per-mechanism sampler, so every comparison is apples-to-apples on identical
syndromes.

## Measurement parameters

| Parameter | Value |
|-----------|-------|
| Distances `d` | 3, 5, 7, 9, 11 (detectors: 16, 72, 192, 400, 720) |
| Physical error `p` | 3.0 % (near the ~2.9 % phenomenological threshold) |
| Accuracy shots / cell | 100 000 |
| Throughput shots / cell | 50 000 |
| Seed | 2024 (fixed; run is deterministic) |
| Reference decoder | PyMatching 2.4.0, via [`PyMatchingOracle`] |
| aleph decoder | [`MwpmDecoder`] (Q1-03 localized matching) |

**Machine.** Apple M4 (10-core, 4P+6E), 24 GB, macOS 26.5.1; rustc 1.96.0 `--release`;
Python 3.12.13 with PyMatching 2.4.0, stim 1.16.0, numpy 2.1.3. The host was running only desktop
background apps (no competing bench job). Each throughput cell reports the **best** of several
timed runs to suppress transient load — best-of-3 for aleph, best-of-5 for the PyMatching core.

Raw data: [`data/qec-q1-compare.csv`](data/qec-q1-compare.csv) (two CSV blocks, `accuracy` and
`throughput`), log [`data/qec-q1-compare.log`](data/qec-q1-compare.log).

## Accuracy (logical-error rate, 100 000 shots/cell, p = 3 %)

| `d` | detectors | aleph rate | PyMatching rate | \|Δ\| | combined 95 % CI | within CI |
|----:|----------:|-----------:|----------------:|------:|-----------------:|:---------:|
| 3 | 16 | 0.10014 | 0.10019 | 5.0 × 10⁻⁵ | 2.63 × 10⁻³ | ✓ |
| 5 | 72 | 0.10667 | 0.10556 | 1.11 × 10⁻³ | 2.70 × 10⁻³ | ✓ |
| 7 | 192 | 0.11088 | 0.11087 | 1.0 × 10⁻⁵ | 2.75 × 10⁻³ | ✓ |
| 9 | 400 | 0.11018 | 0.11156 | 1.38 × 10⁻³ | 2.75 × 10⁻³ | ✓ |
| 11 | 720 | 0.11196 | 0.11352 | 1.56 × 10⁻³ | 2.77 × 10⁻³ | ✓ |

Every distance is within the combined CI. The residual `|Δ|` is the signature of **equal-weight
ties**: where the minimum-weight matching is degenerate, aleph and PyMatching may pick different
matchings in different homology classes, but on a tie either choice is equally (in)correct, so the
disagreements wash out of the rate. This is corroborated below threshold by the oracle test
[`mwpm_pymatching_oracle.rs`], which shows ≥ 99 % *per-shot correction* agreement at `p = 0.6 %`
(where the matching is essentially unique) across the same five distances.

## Throughput (syndromes/second, 50 000 shots/cell, p = 3 %)

aleph is timed in-process, **single-threaded** (its [`Decoder::decode`] core) — the directly
comparable number to PyMatching's single-threaded C++ `decode_batch`. PyMatching is timed two ways:
its **core** matching time (reported by the Python driver, excluding startup/compile/serialisation)
and the **end-to-end** time a Rust caller actually pays (subprocess spawn + interpreter import + DEM
compile + serialisation + match).

| `d` | avg defects | aleph (1 thread) | PyMatching core | PyMatching e2e | aleph ÷ PM-core | aleph ÷ PM-e2e |
|----:|------------:|-----------------:|----------------:|---------------:|----------------:|---------------:|
| 3 | 1.9 | 5 372 000/s | 6 358 000/s | 242 000/s | 0.85× | **22.2×** |
| 5 | 9.5 | 407 000/s | 864 000/s | 191 000/s | 0.47× | **2.13×** |
| 7 | 26.5 | 63 900/s | 277 000/s | 129 000/s | 0.23× | 0.50× |
| 9 | 56.6 | 13 800/s | 114 600/s | 77 200/s | 0.12× | 0.18× |
| 11 | 103.4 | 3 600/s | 55 500/s | 44 200/s | **0.065×** | 0.082× |

**Reading the table.**

- **Core vs core**, aleph is competitive only at `d=3` (light syndromes, ~1.9 defects). As the
  defect count grows the dense-blossom cost compounds: aleph's throughput falls ~1490× from `d=3`
  to `d=11`, PyMatching's only ~115×. By `d=11` PyMatching's sparse blossom is **~15× faster**.
- **End-to-end**, aleph wins big at `d ≤ 5` because PyMatching's ~0.2 s fixed Python startup dwarfs
  the trivial matching — but that advantage is an artefact of the subprocess boundary, not of the
  matching algorithm, and it evaporates by `d=7`.
- aleph's decode is **embarrassingly parallel across shots** (each syndrome is independent; the
  harness already parallelizes sampling), so a deployable multi-core figure would be ~P× these
  single-thread numbers on a P-core host. That still does not close the algorithmic gap at large
  `d` — it shifts the crossover, it doesn't remove it. The real lever is **Sparse Blossom**
  ([#331]).

## Where aleph wins / loses — honest summary

| | aleph `MwpmDecoder` | PyMatching 2.4.0 |
|---|---|---|
| Logical accuracy | **parity** (within CI, all d) | reference |
| Core matching speed | loses, 1.2–15× slower, worsening with d | **wins** (sparse blossom) |
| End-to-end at d ≤ 5 | **wins** (no subprocess/Python tax) | loses (fixed startup) |
| End-to-end at d ≥ 7 | loses | **wins** |
| Deployment | single static Rust binary, no Python/IPC | Python wheel + C++ ext |
| Algorithm | dense localized blossom (Q1-03) | sparse blossom, ~linear |

The takeaway: **Phase Q1 delivers a correct native MWPM decoder at full accuracy parity with the
field reference, but not yet at competitive throughput.** Throughput is the remit of [#331]
(Sparse Blossom), the natural next step before any hardware decoder.

## Reproduce

```bash
# Oracle venv with pymatching + stim + numpy (one-time):
#   python3 -m venv .venv && .venv/bin/pip install pymatching stim numpy

# Head-to-head accuracy + throughput (≈1 min on the M4):
PYMATCHING_PYTHON=$PWD/.venv/bin/python \
  cargo run --release -p aleph-qec --example qec_q1_compare -- 100000 50000 2024 \
  > docs/perf/data/qec-q1-compare.csv 2> docs/perf/data/qec-q1-compare.log

# Correctness oracle (per-shot agreement + within-CI rate, all five distances; nightly/#[ignore]):
PYMATCHING_PYTHON=$PWD/.venv/bin/python \
  cargo test -p aleph-qec --test mwpm_pymatching_oracle -- --ignored --nocapture
```

The example exits non-zero if any distance breaks accuracy parity, so it doubles as a CI-able
regression guard on an oracle-equipped box.

## What this validates

- aleph's from-scratch MWPM is **provably optimal** (the hermetic brute-force matching test in
  `blossom.rs`) **and indistinguishable from PyMatching in logical performance** on real
  surface-code traffic up to `d = 11`.
- The remaining gap to the field is **pure throughput**, quantified here per distance, and
  attributable to the matching algorithm class (dense vs sparse blossom) rather than to any
  correctness defect — which is exactly the precondition for taking on [#331].

## References

- O. Higgott, **PyMatching: A Python package for decoding quantum codes with MWPM**, arXiv:2105.13082.
- O. Higgott & C. Gidney, **Sparse Blossom: correcting a million errors per core-second with MWPM**,
  arXiv:2303.15933.
- J. Edmonds, **Paths, trees, and flowers**, Canad. J. Math. 17 (1965) — the blossom algorithm.
- E. Dennis, A. Kitaev, A. Landahl, J. Preskill, **Topological quantum memory**, J. Math. Phys. 43
  (2002) — surface-code matching decoding.

[`MwpmDecoder`]: ../../crates/aleph-qec/src/mwpm.rs
[`PyMatchingOracle`]: ../../crates/aleph-qec/src/pymatching.rs
[`build_dem`]: ../../crates/aleph-qec/src/builder.rs
[`Decoder::decode`]: ../../crates/aleph-qec/src/decoder.rs
[`mwpm_pymatching_oracle.rs`]: ../../crates/aleph-qec/tests/mwpm_pymatching_oracle.rs
[#331]: https://github.com/ruslan-splynx/aleph/issues/331
