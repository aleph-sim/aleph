# Q7-07 — non-convergence rate, attributable ceiling, and the chosen fallback policy

**Status: both ACs MET.** AC-1 — the non-convergence rate `r(p) = P(valid_flag = 0)` is quantified at
every shipped operating point on both decoder paths, 10⁶ shots per block point. AC-2 — the chosen
policy is **do-nothing-but-flag**, selected under the pre-registered decision rule and with its LER
impact measured: it is the *only* affordable option, because every candidate fallback measured here
makes LER **worse**, by +5.4 % to +39.9 % depending on point and candidate. Closes #458.

The headline number is not the rate. It is the **attributable fraction**
`A(p) = 0.9960 – 1.0000`: at the shipped block operating points, essentially **every logical error is a
non-converged shot**. Non-convergence is not a tail nuisance — it *is* the error budget. And nothing
in the affordable candidate ladder can collect any of it.

## What the ACs asked for

From `docs/qec/BACKLOG.md` § Q7-07:

- AC-1: non-convergence rate quantified per operating point.
- AC-2: fallback policy chosen with data (incl. do-nothing-but-flag if rates are negligible); the LER
  impact of the chosen policy measured in software.

## Why the measurement is conditional, not a straight A/B

The natural reading of AC-2 — run the campaign with and without a fallback and compare LER — is
statistically hopeless. At p=0.003 the campaign LER is 8.32e-4 ± 5.65e-5 at 10⁶ shots
(`qec-q7-nonconv-block.csv`) while `r` is 1.17e-3. A fallback acts on ~0.1 % of shots, so its effect
on overall LER sits well under the campaign CI; resolving it directly needs ~10⁸ shots per arm.

So Q7-07 measures in three levels and propagates analytically
(`crates/aleph-qec/examples/qec_q7_nonconv.rs`):

1. **Rate.** `r(p)` with a Wald 95 % CI, plus the iteration-count distribution from `iters_to_valid`.
2. **Attributable fraction.** Split the logical errors by the flag. `A(p)` = errors with `valid=0`
   over all errors. **This is a hard ceiling**: a fallback only ever touches `valid=0` shots, so even
   a perfect oracle reduces LER by exactly `A(p)` and no more.
3. **Conditional rescue.** Retain only the `valid=0` shots as a fixed on-disk corpus and run every
   candidate against that dense subset, paired-tested (McNemar) against the baseline. Propagate with
   `ΔLER(p) = r(p) · [P(err|v=0) − P(err|v=0, fallback)]`.

**Licence for measuring in software at all:** Q7-06 AC-2 proved the RTL bit-exact to the software
`FixedRelayBp` golden at 10⁶ × 3 rates, so the golden's answer *is* the silicon's answer
(`docs/qec/q7-06-ac1-batched-dma.md`). Q7-07 extends that licence to the flag itself — see
*Board confirmation* below.

### Campaign sizing — 10⁶, not 10⁷, and why

The design assumed `r ≈ 1e-4` and therefore sized the block campaign at 10⁷ shots per point to reach
the 10³-retained corpus target. The pilot **measured** `r` an order of magnitude higher
(1.05e-3 / 9.0e-3 / 3.54e-2 at p = 0.003 / 0.005 / 0.007), so 10⁶ shots already fill the corpus at
every point — 1168 non-converged shots at the worst-case point p=0.003, against a target of 1000 —
and pin `r` to ±6.7e-5. 10⁷ would have bought nothing but 10× the runtime. Run at **10⁶**.

## AC-1 — non-convergence rate, block path (primary)

Gross `[[144,12,12]]`, circuit-level rounds=1 (the Q7-06 on-silicon campaign vehicle), shipped
schedule 6 legs × 10 iters, Q4.3 fixed point, seed 2024. 10⁶ shots per point.
Source: `docs/perf/data/qec-q7-nonconv-block.csv`.

| p | shots | `r` = P(valid=0) | 95 % CI | LER | iters mean | p50 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| 0.003 | 10⁶ | **1.1680e-3** | ±6.70e-5 | 8.3200e-4 | 3.29 | 2 | 20 | 60 |
| 0.005 | 10⁶ | **8.4730e-3** | ±1.80e-4 | 7.0520e-3 | 5.96 | 4 | 48 | 60 |
| 0.007 | 10⁶ | **3.2591e-2** | ±3.48e-4 | 2.8774e-2 | 9.71 | 6 | 60 | 60 |

Cross-check: the LER column reproduces Q7-06 AC-2's campaign exactly (8.3200e-4 / 7.0520e-3 /
2.8774e-2) — same sampler, same seed, same decoder, so the two campaigns are the same experiment
observed through different instrumentation.

`iters` is the first iteration index at which the decision satisfies the syndrome, over the 6×10 = 60
budget. The p99 hitting 60 at p=0.007 is the non-converged tail: those shots never satisfy it.
The median stays at 2–6 iterations, which is why early-exit is worth 163× (Q7-06 AC-1).

## AC-1 — non-convergence rate, window path (secondary)

`HwSlidingWindowBp`, M9b's frozen W=6/C=2 configuration, rounds=12, 20 000 shots, seed 2024,
7 windows per shot. Source: `docs/perf/data/qec-q7-nonconv-window.csv`.

Both normalisations are reported, because the flag counts **windows** and a shot spans many of them.

| p | shots | windows | `r` per window | `r` per shot (≥1 bad window) | LER | dirty-commit frac |
|---|---|---|---|---|---|---|
| 0.001 | 20 000 | 140 000 | **1.826e-2** | **0.1182** | 2.000e-3 ± 6.19e-4 | 0.0033 |
| 0.003 | 20 000 | 140 000 | **1.571e-1** | **0.6682** | 5.840e-2 ± 3.25e-3 | 0.0892 |
| 0.005 | 20 000 | 140 000 | **4.240e-1** | **0.9602** | 4.1575e-1 ± 6.83e-3 | 0.5102 |

The per-shot column reproduces M9b's 11.82 / 66.82 / 96.01 % headline to the digit (0.1182 / 0.6682 /
0.9602 here), which validates the instrumentation against an independently-derived reference. The
per-window rate is 3–6× lower and is the honest per-decode figure: at p=0.001, 11.8 % of *shots* see
a bad window but only 1.83 % of *windows* are bad, and only 0.33 % of shots actually discard a
non-empty commit region. M9b's argument that discarded bits (`commit_clean` / `residual`) is the
sharper health signal survives: `dirty_commit_frac` tracks LER far more closely than either rate does.

The window path is reported for completeness — it is the deployable multi-round decoder — but it is
**not** the path the policy is chosen on. The AC's `Depends on: Q7-02` names the block decoder, and
that is what Q7-06 qualified on silicon.

## The ceiling — A(p), and what it forbids

Source: `qec-q7-nonconv-block.csv`, columns `p_err_given_nonconv`, `p_err_given_conv`,
`attributable`.

| p | P(err \| valid=0) | P(err \| valid=1) | **A(p)** | LER floor a perfect oracle could reach |
|---|---|---|---|---|
| 0.003 | 0.7123 | **0.0** (0 / 998 832) | **1.0000** | **0** |
| 0.005 | 0.8300 | 1.916e-5 | **0.9973** | 1.90e-5 |
| 0.007 | 0.8794 | 1.178e-4 | **0.9960** | 1.14e-4 |

Stated in words, the ticket's central result:

> **At p = 0.003, A = 100.0 %.** Every one of the 832 logical errors in 10⁶ shots was a shot the
> decoder had already flagged `valid=0`. The converged-and-wrong population was *empty*: 998 832
> converged shots, zero logical errors. At p = 0.005 and 0.007 the picture is the same to within a
> fifth of a percent (A = 99.73 % / 99.60 %).

Two consequences, and they point opposite ways.

**The optimistic reading.** `valid_flag` is an almost perfect error detector on this decoder. A
converged relay-BP decision is essentially never wrong: `P(err | valid=1)` is 0, 1.9e-5, 1.2e-4 at the
three points — two to three orders of magnitude below the campaign LER. That makes `valid_flag` a
genuinely useful *heralding* signal, not just telemetry, and it is worth exporting for that reason
alone (it already is — `hw/bp_relay_banked.sv:968`, status word bit 19).

**The pessimistic reading, which is the operative one.** The headroom for a fallback is
*unconstrained by the ceiling* — `A ≈ 1` means an oracle could in principle remove the entire LER —
so this ticket does **not** get to dismiss fallbacks by arithmetic the way the design anticipated.
The rejection has to be earned on measured data. It is, below, and decisively.

## AC-2 — candidate evaluation on the retained corpora

1000 non-converged shots per operating point, retained from the 10⁶-shot streams, re-decoded by each
candidate. The corpus loader asserts on read that every retained shot is *still* non-converged under
the reconstructed DEM and budget, so a drifted corpus cannot silently produce meaningless rows.
McNemar is paired (b = baseline wrong & candidate right, c = the reverse), 1 dof,
continuity-corrected; χ² > 3.84 is significant at 0.05.
Source: `docs/perf/data/qec-q7-nonconv-candidates.csv`.

| p | candidate | solves/shot | errors / 1000 | P(err\|v=0) | rescued | broke | χ² | verdict |
|---|---|---|---|---|---|---|---|---|
| 0.003 | **baseline** (flag only) | 0 | **712** | 0.7123 | — | — | — | reference |
| 0.003 | osd-0 | 1 | 996 | 0.996 | 3 | 287 | 276.17 | **worse**, significant |
| 0.003 | osd-2 | 4 | 943 | 0.943 | 28 | 259 | 184.32 | **worse**, significant |
| 0.003 | osd-4 | 16 | 772 | 0.772 | 108 | 168 | 12.61 | **worse**, significant |
| 0.003 | osd-resid-2 | 4 | 939 | 0.939 | 29 | 256 | 179.21 | **worse**, significant |
| 0.003 | osd-resid-4 | 16 | 765 | 0.765 | 116 | 169 | 9.49 | **worse**, significant |
| 0.005 | **baseline** | 0 | **826** | 0.8300 | — | — | — | reference |
| 0.005 | osd-0 | 1 | 996 | 0.996 | 4 | 174 | 160.46 | **worse**, significant |
| 0.005 | osd-2 | 4 | 968 | 0.968 | 17 | 159 | 112.96 | **worse**, significant |
| 0.005 | osd-4 | 16 | 891 | 0.891 | 62 | 127 | 21.67 | **worse**, significant |
| 0.005 | osd-resid-2 | 4 | 962 | 0.962 | 18 | 154 | 105.96 | **worse**, significant |
| 0.005 | osd-resid-4 | 16 | 871 | 0.871 | 75 | 120 | 9.93 | **worse**, significant |
| 0.007 | **baseline** | 0 | **865** | 0.8794 | — | — | — | reference |
| 0.007 | osd-0 | 1 | 996 | 0.996 | 2 | 133 | 125.19 | **worse**, significant |
| 0.007 | osd-2 | 4 | 974 | 0.974 | 13 | 122 | 86.40 | **worse**, significant |
| 0.007 | osd-4 | 16 | 929 | 0.929 | 42 | 106 | 26.82 | **worse**, significant |
| 0.007 | osd-resid-2 | 4 | 971 | 0.971 | 14 | 120 | 82.28 | **worse**, significant |
| 0.007 | osd-resid-4 | 16 | 925 | 0.925 | 44 | 104 | 23.52 | **worse**, significant |

(The 1000-shot corpus baseline error counts differ slightly from `P(err|v=0)` measured over the full
non-converged population — 712/1000 vs 0.7123 at p=0.003, 826/1000 vs 0.8300 at p=0.005 — because the
corpus is the first 1000 retained shots, not all 1168 / 8473 / 32591. The agreement to three digits
says the corpus is unbiased.)

**Every candidate loses, at every operating point, significantly.** Not one row has χ² in the
candidate's favour. The direction is uniform: the best candidate, residual-restricted OSD-4, rescues
116 shots at p=0.003 and breaks 169 — it is a coin flip biased the wrong way. OSD-0 is the extreme
case: it rescues 3 shots and breaks 287, converting a 71 % conditional error rate into 99.6 %.

This is not a surprise. It reproduces, at the *shipped* operating points and with 5× the statistics,
the Q7-02 M5-followup finding already recorded at `docs/perf/qec-q7-fixed-bp.md:587–640` (OSD-0 makes
circuit-level LER worse; order-4 roughly break-even; only intractable order-12 wins). Q7-07's
contribution is that the rejection is now current, paired, and measured on the exact corpus the
policy would act on. The residual-restricted variant — the one genuinely new candidate, and the
literal reading of the backlog's "OSD-lite on the residual" — is the best of the ladder and still
loses: it buys a higher effective order for the same 2^w solve budget (116 rescued vs 108 for
unrestricted OSD-4 at p=0.003), but not enough to change the sign.

**Why OSD hurts here.** Relay-BP's best-kept decision is selected across 6 disordered legs by
weight; when it violates the syndrome it is still, empirically, close. OSD replaces it wholesale with
a syndrome-*satisfying* solution built from the most-reliable basis — and at these rates the
BP posteriors that order that basis are exactly the ones that just failed to converge, so the
resulting solution satisfies the syndrome while landing in the wrong logical class. Forcing
`H ê = s` is not the same as being right.

### LER impact of every candidate, propagated

`ΔLER(p) = r(p) · [P(err|v=0) − P(err|v=0, fallback)]`, `r` and the baseline LER from the block CSV,
conditional rates from the candidates CSV. Negative ΔLER = the fallback increases LER.

| p | baseline LER | osd-0 | osd-2 | osd-4 | osd-resid-2 | **osd-resid-4** (best) |
|---|---|---|---|---|---|---|
| 0.003 | 8.320e-4 | 1.164e-3 (+39.9 %) | 1.102e-3 (+32.4 %) | 9.021e-4 (+8.4 %) | 1.097e-3 (+31.9 %) | 8.939e-4 (**+7.4 %**) |
| 0.005 | 7.052e-3 | 8.492e-3 (+20.4 %) | 8.255e-3 (+17.1 %) | 7.603e-3 (+7.8 %) | 8.204e-3 (+16.3 %) | 7.433e-3 (**+5.4 %**) |
| 0.007 | 2.877e-2 | 3.304e-2 (+14.8 %) | 3.233e-2 (+12.3 %) | 3.086e-2 (+7.2 %) | 3.223e-2 (+12.0 %) | 3.073e-2 (**+6.8 %**) |

The chosen policy's LER impact, which is what AC-2 asks for, is therefore **exactly zero by
construction** — do-nothing-but-flag *is* the baseline column — and the measured cost of every
alternative is between +5.4 % and +39.9 % LER.

### Latency, and a caveat about the parallel timing column

The `us_per_shot` column in `qec-q7-nonconv-candidates.csv` (79–91 µs) is **wall-clock over a
rayon-parallel loop divided by the shot count** on a 32-core EPYC. It understates single-thread cost
by roughly the core count and is **not** a valid latency-budget number. It is kept in the CSV for
relative comparison between candidates only.

The budget-relevant measurement is the same run pinned to one thread
(`RAYON_NUM_THREADS=1`, `docs/perf/data/qec-q7-nonconv-candidates-1thread.csv`, p=0.005, idle box):

| candidate | µs/shot, 1 thread (EPYC 8124P) | vs the 1 µs/round budget |
|---|---|---|
| osd-0 | 1460.4 | 1460× over |
| osd-2 | 1609.7 | 1610× over |
| osd-4 | 1626.6 | 1627× over |
| osd-resid-2 | 1629.4 | 1629× over |
| osd-resid-4 | 1643.7 | 1644× over |

For scale: the whole early-exit hardware decode is **1.81 µs/shot** on silicon (Q7-06 AC-1), and
Q7-01 targets 1 µs/round. A PS-side OSD tail costs ~1.6 ms *per shot it fires on*, on a 3 GHz-class
x86 core — the KV260's 1.33 GHz Cortex-A53 would be several times slower again. Real-time QEC is
governed by worst case, not mean: a tail that fires on 0.12 % of shots still sets the worst-case
latency at ~1.6 ms, which is ~900× the entire hardware decode and ~1600× the per-round budget.

So even a candidate that *won* on LER would be **rejected-on-latency** here. None won, so the
rejection is over-determined.

## Board confirmation — the flag itself is bit-exact on silicon

Q7-06 proved the RTL's *observables* bit-exact to the software golden. Q7-07 needs one more claim:
that the RTL's `valid_flag` agrees with the software golden's `valid` shot for shot — otherwise the
policy is chosen on a signal the silicon does not actually reproduce.

`.ref` was extended to v2 (magic `0xA1E7`, version, 3 u16/shot: `true_obs`, `sw_obs`,
`(valid << 15) | iters`) with a version gate so a stale v1 file cannot be silently misread — the same
footgun class as #478. `hw/sw/bp_stream_banked_ler_kv260.py` now captures status-word bit 19 and
reports `valid_mismatch`. **No RTL change, no new bitstream.**

Run on the KV260 (`root@192.168.88.174`) against the matched-prior `bp_p005.bit` overlay from the
Q7-06 re-run, 10⁵ fresh shots at p=0.005 (`docs/perf/data/qec-q7-nonconv-board.csv`):

| overlay | n | sw LER | RTL LER | obs divergence | rtl_nonconv | sw_nonconv | **valid_mismatch** | verdict |
|---|---|---|---|---|---|---|---|---|
| `bp_p005.bit` | 100 000 | 7.1200e-3 | 7.1200e-3 | **0 / 100 000** | 857 (0.857 %) | 857 | **0** | **PASS** |

Driver output, verbatim:

```
  point        n        sw_ler       rtl_ler       |diff|     comb_ci   divergence  verdict
  p005v2      100000  7.1200e-03  7.1200e-03  0.000e+00  1.042e-03       0/100000    PASS
           (2.31 s, 23.10 us/shot; rtl_err=712 sw_err=712)
           (valid: rtl_nonconv=857 (0.8570%), sw_nonconv=857, mismatch=0; cycles mean=2085.0 max=2085)
AC-2 RESULT: PASS (RTL LER within CI of software golden; valid_flag matches at every point)
```

**Gate met: `mismatch=0`.** The hardware flag is not merely statistically consistent with the golden
— it is identical on every one of 100 000 shots. That is the licence for the whole software analysis
above. (The 0.857 % on-silicon rate at p=0.005 sits just under the 0.847 % ± 0.018 % campaign figure,
as expected for a different 10⁵ sample.)

## The verdict, and the gap in the pre-registered rule

The decision rule was fixed before the data was seen (design § *Pre-registered decision rule*):

> - `A(p) < 5 %` at every operating point → **do-nothing-but-flag**.
> - `A(p) ≥ 5 %` **and** some candidate shows a significant conditional rescue (paired McNemar on the
>   non-converged subset) **and** its worst-case latency fits the 1 µs/round budget → that candidate
>   is chosen and implemented.
> - Wins on LER, breaks latency → **rejected-on-latency**, arithmetic recorded.

**The data landed in a combination the rule does not have a branch for**, and this is worth saying
plainly rather than quietly filing under the nearest clause. The measurement is `A(p) ≈ 1.0` **and no
candidate wins** — the second branch's first condition holds and its second fails, and the first
branch's condition fails. The complement of "some candidate wins" resolves to **do-nothing-but-flag**,
so the verdict is unambiguous. But it is reached for a reason precisely opposite to the one the rule's
first branch anticipated:

- The rule expected do-nothing-but-flag to mean *"non-convergence is negligible, so ignore it."*
- What was actually measured is *"non-convergence is **nearly the entire logical-error budget**, and
  nothing in the affordable candidate ladder can rescue any of it — every candidate makes it worse,
  and would blow the latency budget by ~1600× even if it did not."*

Those are very different engineering situations that happen to share an action. Filing this result
under "rates were negligible" would misrepresent it: the rate is not negligible, the *ceiling* is not
the constraint, and the ceiling is not what rejects the fallbacks. **Measured LER regression is.**

### Chosen policy

**Do-nothing-but-flag.** The decoder emits its best-kept decision and sets `valid_flag = 0`, exactly
as `hw/bp_relay_banked.sv:968`/`:956` and `FixedRelayBp::decode_fixed` already do. No RTL change, no
PS-side tail, no re-synthesis. `valid_flag` ships as a **heralding and telemetry** signal:

- **Heralding.** `P(err | valid=1)` is 0 / 1.9e-5 / 1.2e-4 at p = 0.003 / 0.005 / 0.007. A converged
  decode is right essentially always, so a consumer that can afford to discard or escalate flagged
  shots gets a post-selected LER two to three orders of magnitude below the raw campaign LER. That is
  a far larger win than any fallback measured here, and it is available for free.
- **Telemetry.** `r(p)` is monotone and steep in p (1.17e-3 → 8.47e-3 → 3.26e-2 over a 2.3× range in
  physical error rate), which makes the flag rate a sensitive live estimator of the device's actual
  operating point — useful for detecting prior mismatch, the exact failure mode that caused #478.

### What would change this verdict

Not a cheaper OSD. The ladder fails on *accuracy*, not cost — order-4 already costs 16 solves/shot and
still regresses. Two directions remain open, both out of Q7-07's scope:

- **More relay legs / fresh disorder on `!valid_flag`** — explicitly out of scope here (design
  § *Candidates*), and a hardware-side policy: it would change the RTL schedule, so per the design's
  implementation boundary it returns for a scope decision rather than being decided by this ticket.
  It is the more promising direction precisely because it attacks the decoder that failed rather than
  post-processing its failure.
- **Escalate-to-host** on the 0.12–3.3 % flagged shots, accepting the latency for a non-real-time
  deployment. `A ≈ 1` says the entire LER is reachable that way; the 1.6 ms tail says it is not
  reachable in real time.

## Reproduce

```
# Block path — rate, attributable fraction, and the retained corpora (10^6 shots x 3 points).
cargo run --release -p aleph-qec --example qec_q7_nonconv -- block 1 1000000 2024 q7-07 \
  > docs/perf/data/qec-q7-nonconv-block.csv

# Candidate ladder on each corpus (headers deduped when concatenating the three runs).
for f in q7-07-p003.corpus q7-07-p005.corpus q7-07-p007.corpus; do
  cargo run --release -p aleph-qec --example qec_q7_nonconv -- candidates "$f"
done > docs/perf/data/qec-q7-nonconv-candidates.csv

# Budget-relevant single-thread latency (idle box; the parallel column is NOT a latency number).
RAYON_NUM_THREADS=1 cargo run --release -p aleph-qec --example qec_q7_nonconv -- \
  candidates q7-07-p005.corpus > docs/perf/data/qec-q7-nonconv-candidates-1thread.csv

# Window path — per-window and per-shot rates, rounds=12, W=6/C=2.
cargo run --release -p aleph-qec --example qec_q7_nonconv -- window 12 20000 2024 \
  > docs/perf/data/qec-q7-nonconv-window.csv
```

```
# Board confirmation of the flag (KV260, matched-prior p=0.005 overlay, .ref v2).
cargo run --release -p aleph-qec --example qec_q7_bp_graph -- silvectors 1 0.005 100000 2024 p005v2 0.005
scp p005v2.syn p005v2.ref hw/sw/bp_stream_banked_ler_kv260.py root@192.168.88.174:~/q7stream/
ssh root@192.168.88.174 'cd ~/q7stream && sudo env XILINX_XRT=/usr \
  /usr/local/share/pynq-venv/bin/python3 bp_stream_banked_ler_kv260.py bp_p005.bit p005v2'
# Gate: mismatch=0.
```

```
# Off-board regression gates (unchanged; bpbanked-highweight already compares valid_flag):
make -C hw bpbanked-highweight
```

Campaigns ran on the EPYC bench box (`root@195.154.249.85`, 32 cores), verified idle
(`uptime` load 0.02, no `cargo bench` / `Runner.Worker`) before measuring.

## Data

| file | contents |
|---|---|
| `docs/perf/data/qec-q7-nonconv-block.csv` | block path: `r`, CI, LER, `P(err\|v)` both ways, `A`, iteration percentiles, retained |
| `docs/perf/data/qec-q7-nonconv-candidates.csv` | candidate ladder × 3 points: errors, conditional rate, solves/shot, parallel µs/shot, McNemar |
| `docs/perf/data/qec-q7-nonconv-candidates-1thread.csv` | the same ladder at p=0.005, single-threaded — the latency-valid timing |
| `docs/perf/data/qec-q7-nonconv-window.csv` | window path: per-window and per-shot rates, `commit_clean`/`residual` fractions |
| `docs/perf/data/qec-q7-nonconv-board.csv` | KV260 `valid_flag` confirmation, 10⁵ shots, `mismatch=0` |
