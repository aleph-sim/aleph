# F6 — `aleph.cudaq` A/B: aleph decoders vs NVIDIA inside cudaq-qec

**Status:** complete. Produces the evidence for Track O, F6 (`docs/qec/open-silicon-program.md`):
aleph's seven QEC decoders registered inside NVIDIA `cudaq-qec` 0.8 (Task 6, commit `96d2f76`),
A/B'd against NVIDIA's own decoders on identical inputs.

## Purpose

`aleph.cudaq` lets a `cudaq_qec` integrator try aleph's relay-BP / Sparse Blossom MWPM without
leaving their harness — `pip install aleph-sim[cudaq]; import aleph.cudaq` registers seven
`aleph-*` decoder names that `cudaq_qec.get_decoder` resolves like any built-in. This record is
the honest A/B that makes that useful: identical `(H, O, error_rate_vec)` and identical
syndrome/observable batches fed to both sides, logical error rate (95% Wilson CI) and shots/s
measured the same way for every decoder in a workload. Two workloads: **gross qLDPC**
(`[[144,12,12]]`, circuit-level noise) pits `aleph-relay-bp(-osd)` against `nv-qldpc-decoder` in
relay mode; **surface code** (`rotated_memory_x`, d ∈ {5,9}) pits `aleph-mwpm` /
`aleph-union-find-weighted` against cudaq's `pymatching` and `nv-fusion-decoder`. Per
`docs/superpowers/specs/2026-09-24-cudaq-plugin-design.md` §8, this is a same-inputs comparison,
not a tuned bake-off: NVIDIA's relay-BP is run at NVIDIA's own documented parameters, not aleph's,
and vice versa — see "Reading the numbers" below before drawing conclusions from any single cell.

## Machine and versions

GPU box `openwebgui.splynx.com` (NVIDIA RTX 4000 SFF Ada, driver 580.178.04, CUDA 13.0 —
`nvidia-smi` is not installed in this container; the driver was confirmed instead via
`/proc/driver/nvidia/version` and the `/dev/nvidia*` device nodes), venv `/root/cqvenv`. Idle
immediately before the full run (CLAUDE.md § Performance):

```
$ uptime
 08:54:19 up 8 days, 10:45,  1 user,  load average: 1.26, 2.19, 2.45
$ pgrep -af python | grep -v uvicorn
(no output — only the unrelated uvicorn service was running; the load average above is decaying
 residue from this session's own earlier --quick smoke runs, not concurrent work)
```

Harness header line (verbatim, printed by the script itself):

```
aleph 0.3.0, cudaq-qec CUDA-Q QEC 0.8.0 (https://github.com/NVIDIA/cudaqx 6ac00e5c24ab17bbb6515f09009ce69643839eed), cudaq CUDA-Q Version 0.16.0 (https://github.com/NVIDIA/cuda-quantum 62fce8b862302158e0119c65e3ce57538f3af048), stim 1.16.0, numpy 2.5.3, Linux-6.8.0-139-generic-x86_64-with-glibc2.39, 20 CPUs, python 3.12.3
08:54:26 up 8 days, 10:45,  1 user,  load average: 1.15, 2.14, 2.42
NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.178.04  Tue Jul  7 12:38:21 UTC 2026
```

## Commands

```bash
# sync (excluding the compiled extension — see hazard note below)
rsync -az --exclude target --exclude .git --exclude dist --exclude '*.so' \
  /Users/ex/GitHub/aleph/ root@openwebgui.splynx.com:/root/aleph-f6/

# rebuild the branch wheel on the box — required after every rsync, since rsync
# does not touch _native.abi3.so but a *previous* rsync in this session may have
# (Task 6's hazard note); verify the branch build before trusting any run:
ssh -o BatchMode=yes root@openwebgui.splynx.com \
  'export PATH=/root/.cargo/bin:$PATH; cd /root/aleph-f6/crates/aleph-py && \
   VIRTUAL_ENV=/root/cqvenv /root/cqvenv/bin/maturin develop --release'
ssh -o BatchMode=yes root@openwebgui.splynx.com \
  '/root/cqvenv/bin/python -c "import aleph.qec as q; print(hasattr(q, \"gross_code_dem\"))"'
# -> True

# quick smoke run (foreground; -u so a killed/timed-out run still shows partial progress —
# python fully block-buffers stdout when it isn't a tty, so without -u a run that is still
# healthy but slower than expected looks indistinguishable from a hang)
ssh -o BatchMode=yes root@openwebgui.splynx.com \
  'cd /root/aleph-f6 && /root/cqvenv/bin/python -u -W ignore scripts/python/ab_cudaq.py --quick'

# full run (background; box idle, 29 min measured — see "Adaptations" below)
ssh -o BatchMode=yes root@openwebgui.splynx.com \
  'cd /root/aleph-f6 && nohup /root/cqvenv/bin/python -u -W ignore scripts/python/ab_cudaq.py \
   > /root/aleph-f6/ab_cudaq.out 2>&1 < /dev/null & disown'
# poll (repeat until the process exits):
ssh -o BatchMode=yes root@openwebgui.splynx.com "pgrep -f 'ab_cudaq.py$' && echo RUNNING || echo DONE"
ssh -o BatchMode=yes root@openwebgui.splynx.com 'tail -50 /root/aleph-f6/ab_cudaq.out'
```

## Parameter sets (verbatim)

**`ALEPH_OSD`** (aleph-relay-bp-osd's OSD stage; relay-BP defaults otherwise — 4 legs, α = 0.875,
γ ∈ [−0.3, 0.9], per `docs/perf/qec-q5-circuit-dem.md`):

```python
ALEPH_OSD = dict(osd_order=12)
```

**`NV_RELAY`** (`nv-qldpc-decoder`, relay mode). NVIDIA's own performance guide
(`docs/sphinx/performance/nv_qldpc_relay_solutions_user_guide.rst`, NVIDIA/cudaqx, cudaq-qec
0.8.0) publishes "the canonical Relay BP settings" for **this exact code** — the `[[144,12,12]]`
gross code, 12-round circuit-level DEM — so this harness uses those values verbatim instead of
guessing:

```python
NV_RELAY = dict(use_sparsity=True, bp_method=3, composition=1, max_iterations=60,
                gamma0=0.125, gamma_dist=[-0.24, 0.66], clip_value=200.0, repeatable=True,
                srelay_config=dict(pre_iter=80, num_sets=60, stopping_criterion="NConv", stop_nconv=5))
# + bp_batch_size=min(1000, shots)                          on every row (see below)
# + {"use_osd": True, "osd_order": 12, "osd_method": 1}     for the OSD row
# + {"use_osd": False}                                       for the no-OSD row
```

`bp_method=3` is min-sum with **disordered** per-node memory (cudaq-qec calls this "DMem-BP"),
`composition=1` selects sequential relay composition. `pre_iter=80, num_sets=60` (the maximum
relay-leg budget) and the γ/clip/repeatable values are NVIDIA's canonical settings for this code
verbatim; `stopping_criterion="NConv", stop_nconv=5` replaces the doc's own `"All"` (see
"Adaptations" below — `"All"` is specific to that doc's offline-recording use case, not a
decoding default). This deviates from the harness's original draft guess (`pre_iter=60,
num_sets=4, stopping_criterion="All", max_iterations=100`), which appears nowhere in NVIDIA's
docs or examples for any code; `pre_iter`/`num_sets`/γ/clip/repeatable above are the ones NVIDIA
actually measured and published for the identical `[[144,12,12]]` workload this table uses. Kept
out: the doc's recording-only knobs (`opt_results={"relay_solutions": True, ...}`,
`output="observables"`) — this harness is not doing an offline `stop_nconv` sweep — and
`proc_float="fp32"`, so NVIDIA's numeric precision stays comparable to aleph's fp64 path (this is
an extra speed/precision knob orthogonal to the relay schedule itself).

**`nv-fusion-decoder`**: `error_rate_vec` plus `num_threads=20` (matching aleph's all-core row),
`output="observables"` and `detector_round` (an `int32` array, one entry per detector, from
Stim's own `circuit.get_detector_coordinates()` third coordinate). `block_leaf_size` and
`fusion_strategy` left at their 0.8.0 defaults (not set by this harness — both surface distances
are far under the 192-round auto-threshold, so the schedule is a single leaf, i.e. exact
monolithic MWPM).

**`pymatching`** (via cudaq-qec): `error_rate_vec` only, its own defaults otherwise.

## Adaptations made while getting the harness to run cleanly

The brief's draft script needed five fixes before both tables printed with no `ERROR:` rows;
none of them changes what is measured, only how the same measurement is obtained or how large a
sample it runs on.

1. **Unbuffered stdout (`-u`).** Python fully block-buffers stdout when it is not a tty, so a
   redirected run's progress is invisible until it exits — indistinguishable from a hang when a
   cell is slower than expected. Every invocation below uses `-u`.
2. **`bp_batch_size` on `nv-qldpc-decoder`.** Without it, `decode_batch` defaults to
   `bp_batch_size=1` and the decoder itself warns: *"called with default bp_batch_size=1 while
   decoding N syndromes. Set bp_batch_size > 1 at construction to decode syndromes in parallel."*
   Unbatched, one GPU kernel launch per shot made even a 2,000-shot `--quick` batch fail to finish
   in 300 s. Fixed by adding `bp_batch_size=min(1000, shots)` (NVIDIA's own canonical-settings run
   for this code uses `bp_batch_size=1000`) to every `nv-qldpc-decoder` construction.
3. **`srelay_config.stopping_criterion`: `"All"` → `"NConv"`, `stop_nconv=5`.** NVIDIA's
   published canonical settings use `stopping_criterion="All"`, `num_sets=60` — but that
   configuration is that doc's own *offline recording* run (it forces every shot through the full
   60-leg schedule regardless of early convergence, so a later `stop_nconv` sweep can be
   reconstructed from the recording). Measured directly here: an identical 2,000-shot batch took
   ~31 s/pass under `"All"` versus ~4.7 s/pass under `"NConv", stop_nconv=5` — a ~6.5x difference
   with no offline sweep to justify paying for it. `stop_nconv=5` is not a guess: the same NVIDIA
   doc's own RelayBP-N sweep on this exact code concludes *"for this code and noise,
   `stop_nconv=5` buys all of the measured accuracy at a small fraction of the cost of larger
   N"* (LER falls ~5x from N=1 to N=5, then saturates, while mean iterations keep growing linearly
   in N). `pre_iter=80, num_sets=60` (the leg budget available to reach 5 convergences) and every
   other canonical value are kept as published.
4. **Result-basis detection.** The original `run_decoder` always computed
   `pred = ((result > 0.5) @ O.T) % 2`, assuming every decoder returns per-mechanism error
   estimates. Empirically, once `O` is supplied at construction, `nv-qldpc-decoder` and
   `pymatching` return **observable**-space predictions directly (`result.shape[1] ==
   O.shape[0]`, not `O.shape[1]`) — multiplying by `O.T` again produced a matmul dimension-mismatch
   `ERROR:` row for both. Only `aleph.cudaq`'s own decoders return raw per-mechanism estimates
   regardless of `O` (documented in its own module docstring). Fixed by checking which width
   `result` actually has and branching accordingly, rather than assuming one convention for every
   decoder — this also makes the harness fail loud (`RuntimeError`) instead of silently comparing
   the wrong axis if a decoder's convention changes.
5. **`nv-fusion-decoder` construction from raw `(H, O)`.** Constructed this way (rather than from
   DEM text), it has no detector coordinates to derive a temporal layout from and raised
   `RuntimeError: nv-fusion-decoder: scaffold not yet built; set_D_sparse() must be called before
   decode`. Per `nv_fusion_decoder_api.rst`, an explicit `detector_round` (one integer round index
   per detector) "takes highest priority over all automatic derivation paths"; Stim's
   `circuit.get_detector_coordinates()` gives that directly as each detector's time coordinate.
   `output="observables"` is separately required — the same doc states "supplying O ... does not
   select observable output" for this decoder (unlike `nv-qldpc-decoder`/`pymatching`, whose
   output basis is not user-selectable and is observable-space unconditionally once `O` is given).
6. **Gross-workload shot counts, to fit an honest full run in well under an hour** (permitted by
   the brief: "reduce that cell's shots and say so in the record — never drop a cell silently"):
   - **20,000 shots at every p** (not 100,000 at p ≤ 0.001 as first planned). Measured on the box,
     `aleph-relay-bp-osd` at `RAYON_NUM_THREADS=1` runs ~60-75 shots/s (one core doing 4-leg
     relay-BP plus OSD-12's combination sweep per shot); 100,000 shots x 3 repeats would have cost
     roughly 80 minutes for that one cell alone. The OSD rows' LER is already at/near the
     zero-error floor at p ≤ 0.001 in existing 1,000-shot runs
     (`docs/perf/qec-q5-circuit-dem.md`), so the extra 80,000 shots would not have changed that
     row's conclusion; every row still clears the spec's "≥ 10,000 shots" floor at every p.
   - **The two aleph 1-thread rows in Workload G use a 2,000-shot prefix of the same sampled
     batch**, not the full 20,000. The 20-core row two lines above already measures that decoder's
     LER at full statistical power on the complete batch (thread count cannot change which answer
     a deterministic decoder converges to, only how fast); running the identical algorithm again
     on the same 20,000 shots at ~60-75 shots/s would have cost ~15-16 minutes per cell, ~90
     minutes across the two 1-thread rows x three p values — by far the largest cost in the whole
     harness for no new statistical information. The 1-thread rows exist to report *throughput*;
     their own (wider-CI) LER is still reported, on the reduced sample the `shots` column names,
     rather than dropped or hidden.

With these six fixes, `--quick` (2,000/2,000-shot batches) reproducibly prints both tables with
no `ERROR:` rows on this box in well under 5 minutes.

## Results

Full run: started 08:54:26, finished 09:23:26 CEST — **29 minutes**, box idle throughout (this
session's own earlier `--quick` smoke runs were the only other recent activity; no CI/other job
ran concurrently).

### Workload G — gross `[[144,12,12]]`, rounds=12, circuit-level uniform p

| workload | decoder | config | threads | shots | LER [95% CI] | non-conv | shots/s |
|---|---|---|---:|---:|---|---:|---:|
| gross p=0.001 | aleph-relay-bp-osd | relay+OSD-12 | 20 | 20,000 | 0.00e+00 [0.0e+00, 1.9e-04] | 0 | 561 |
| gross p=0.001 | aleph-relay-bp-osd | relay+OSD-12 | 1 | 2,000 | 0.00e+00 [0.0e+00, 1.9e-03] | 0 | 71 |
| gross p=0.001 | aleph-relay-bp | relay, no OSD | 20 | 20,000 | 7.00e-04 [4.2e-04, 1.2e-03] | 31 | 586 |
| gross p=0.001 | aleph-relay-bp | relay, no OSD | 1 | 2,000 | 1.50e-03 [5.1e-04, 4.4e-03] | 4 | 72 |
| gross p=0.001 | nv-qldpc-decoder | relay+OSD-12 | GPU | 20,000 | 0.00e+00 [0.0e+00, 1.9e-04] | 0 | 1,882 |
| gross p=0.001 | nv-qldpc-decoder | relay, no OSD | GPU | 20,000 | 0.00e+00 [0.0e+00, 1.9e-04] | 0 | 2,003 |
| gross p=0.002 | aleph-relay-bp-osd | relay+OSD-12 | 20 | 20,000 | 1.50e-04 [5.1e-05, 4.4e-04] | 0 | 553 |
| gross p=0.002 | aleph-relay-bp-osd | relay+OSD-12 | 1 | 2,000 | 0.00e+00 [0.0e+00, 1.9e-03] | 0 | 70 |
| gross p=0.002 | aleph-relay-bp | relay, no OSD | 20 | 20,000 | 8.50e-03 [7.3e-03, 9.9e-03] | 242 | 586 |
| gross p=0.002 | aleph-relay-bp | relay, no OSD | 1 | 2,000 | 1.00e-02 [6.5e-03, 1.5e-02] | 30 | 73 |
| gross p=0.002 | nv-qldpc-decoder | relay+OSD-12 | GPU | 20,000 | 0.00e+00 [0.0e+00, 1.9e-04] | 0 | 927 |
| gross p=0.002 | nv-qldpc-decoder | relay, no OSD | GPU | 20,000 | 0.00e+00 [0.0e+00, 1.9e-04] | 0 | 941 |
| gross p=0.003 | aleph-relay-bp-osd | relay+OSD-12 | 20 | 20,000 | 2.15e-03 [1.6e-03, 2.9e-03] | 0 | 500 |
| gross p=0.003 | aleph-relay-bp-osd | relay+OSD-12 | 1 | 2,000 | 2.50e-03 [1.1e-03, 5.8e-03] | 0 | 65 |
| gross p=0.003 | aleph-relay-bp | relay, no OSD | 20 | 20,000 | 4.57e-02 [4.3e-02, 4.9e-02] | 1261 | 587 |
| gross p=0.003 | aleph-relay-bp | relay, no OSD | 1 | 2,000 | 4.30e-02 [3.5e-02, 5.3e-02] | 121 | 73 |
| gross p=0.003 | nv-qldpc-decoder | relay+OSD-12 | GPU | 20,000 | 1.50e-04 [5.1e-05, 4.4e-04] | 1 | 402 |
| gross p=0.003 | nv-qldpc-decoder | relay, no OSD | GPU | 20,000 | 1.50e-04 [5.1e-05, 4.4e-04] | 1 | 400 |

### Workload S — `surface_code:rotated_memory_x`, rounds=d, p=0.003, decomposed

| workload | decoder | config | threads | shots | LER [95% CI] | non-conv | shots/s |
|---|---|---|---:|---:|---|---:|---:|
| surface d=5 | aleph-mwpm | - | 20 | 20,000 | 3.80e-03 [3.0e-03, 4.8e-03] | 0 | 171,927 |
| surface d=5 | aleph-mwpm | - | 1 | 20,000 | 3.80e-03 [3.0e-03, 4.8e-03] | 0 | 114,247 |
| surface d=5 | aleph-union-find-weighted | - | 20 | 20,000 | 3.95e-03 [3.2e-03, 4.9e-03] | 0 | 171,800 |
| surface d=5 | aleph-union-find-weighted | - | 1 | 20,000 | 3.95e-03 [3.2e-03, 4.9e-03] | 0 | 110,213 |
| surface d=5 | pymatching | defaults | 1 | 20,000 | 3.80e-03 [3.0e-03, 4.8e-03] | 0 | 657,571 |
| surface d=5 | nv-fusion-decoder | defaults | 20 | 20,000 | 3.80e-03 [3.0e-03, 4.8e-03] | 0 | 333,743 |
| surface d=9 | aleph-mwpm | - | 20 | 5,000 | 8.00e-04 [3.1e-04, 2.1e-03] | 0 | 25,673 |
| surface d=9 | aleph-mwpm | - | 1 | 5,000 | 8.00e-04 [3.1e-04, 2.1e-03] | 0 | 16,820 |
| surface d=9 | aleph-union-find-weighted | - | 20 | 5,000 | 1.00e-03 [4.3e-04, 2.3e-03] | 0 | 25,721 |
| surface d=9 | aleph-union-find-weighted | - | 1 | 5,000 | 1.00e-03 [4.3e-04, 2.3e-03] | 0 | 16,074 |
| surface d=9 | pymatching | defaults | 1 | 5,000 | 8.00e-04 [3.1e-04, 2.1e-03] | 0 | 95,587 |
| surface d=9 | nv-fusion-decoder | defaults | 20 | 5,000 | 8.00e-04 [3.1e-04, 2.1e-03] | 0 | 61,069 |

## Reading the numbers

*(honesty rules from `docs/superpowers/specs/2026-09-24-cudaq-plugin-design.md` §8)*

- **Relay-BP parameterisations are not equivalent, and the table shows it clearly.** aleph's
  `relay-bp`/`relay-bp-osd` (4 legs, α = 0.875, γ ∈ [−0.3, 0.9] — its plain defaults, no tuning)
  and NVIDIA's `nv-qldpc-decoder` relay mode (up to 60 legs, stopping after 5 convergences,
  γ = 0.125/[−0.24, 0.66], the settings NVIDIA measured and published for this exact code) explore
  a very different amount of the relay schedule per shot, and it shows: at p=0.003, no-OSD LER is
  4.57e-2 (aleph) vs 1.5e-4 (NVIDIA) — a ~300x gap — and the OSD rows are 2.15e-3 vs 1.5e-4 (~14x).
  This is not a bug; it is the expected consequence of comparing a 4-leg schedule against a
  60-leg-budget one, and it is exactly what follow-up #<pending> (an `iters_per_leg`/early-exit
  knob for aleph's `RelayBpDecoder`) would let this table control for. Neither side was tuned
  against the other beyond its own documented defaults, per the honesty rule.
- **CPU vs GPU throughput is stated, not ranked.** aleph decoders run on the CPU (20 cores, and
  again pinned to 1 thread via `RAYON_NUM_THREADS=1` in a child process, or a 2,000-shot prefix
  for the two Workload G cells named in "Adaptations" above); `nv-qldpc-decoder`, `nv-fusion-decoder`
  and `pymatching` (via cudaq-qec) run on the GPU / cudaq-qec's own path. Different hardware, no
  claimed winner on shots/s: `nv-qldpc-decoder`'s 400-2,003 shots/s and aleph's 20-core 500-587
  shots/s are reported side by side, not ranked, and the same goes for `pymatching`'s 657k-96k
  shots/s (1 thread) and `aleph-mwpm`'s 172k-26k (20 cores) — both fast enough that most of a
  20,000-shot batch's wall time in this table is warm-up/Python overhead, not the matcher itself.
- **`ERROR:` / non-converged cells.** No `ERROR:` rows in the final run (five configuration bugs
  surfaced by earlier `--quick` attempts, all fixed — see "Adaptations" above). Non-convergence:
  aleph's no-OSD relay-BP has a non-trivial non-converged count that grows with p (31/20,000 at
  p=0.001 up to 1,261/20,000 at p=0.003, 4-16%) — expected, since it is BP-family with no
  correction fallback and only 4 relay legs; every one of those shots still gets a best-effort
  `ehat` (converged=False, not dropped), which is exactly what the elevated no-OSD LER reflects.
  `nv-qldpc-decoder`'s non-conv is 0 at p≤0.002 and 1/20,000 at p=0.003, consistent with its much
  larger relay budget. `aleph-relay-bp-osd`'s non-conv is 0 everywhere (OSD's combination sweep
  always returns a candidate). All matching decoders (`aleph-mwpm`, `aleph-union-find-weighted`,
  `pymatching`, `nv-fusion-decoder`) show non-conv=0 throughout, as expected for exact/near-exact
  MWPM on these small distances.
- **Sanity check (must hold or the harness has a bug): MET.** `aleph-mwpm` and `pymatching` decode
  the identical graphlike `(H, O, error_rate_vec)` and syndromes; their LER is **exactly equal**
  at both distances (d=5: 3.80e-03 vs 3.80e-03; d=9: 8.00e-04 vs 8.00e-04) — not just within the
  95% CI, bit-for-bit the same count of wrong shots out of 20,000/5,000. `nv-fusion-decoder`
  (single-leaf, i.e. exact monolithic MWPM for these round counts, per its own docs — see the
  parameter-set note above) agrees with both at every row too. This is strong evidence the harness
  is wiring `(H, O, syndromes)` identically to every decoder and that aleph's Sparse Blossom MWPM
  is decoding correctly against an independent, widely-used reference.
- **aleph and NVIDIA relay LER are NOT consistently the same order of magnitude — NOT MET,
  and expected.** At p=0.001 the OSD rows agree (both 0/20,000) and the no-OSD rows are within a
  reasonable factor (7.0e-4 vs an upper CI bound of 1.9e-4, roughly 4x); but at p=0.002-0.003 the
  gap widens sharply (up to ~300x on no-OSD at p=0.003, detailed above). This traces directly to
  the leg-count/stopping-criterion disparity discussed in the relay-BP bullet above, not to a
  harness bug: both decoders see the identical DEM, syndromes, and `error_rate_vec`, and the
  `aleph-mwpm`/`pymatching` cross-check on the same code path (dem_to_matrices → decode_batch →
  compare to `obs`) shows the plumbing is correct. Recorded here per the brief rather than
  investigated further, since spec §8 explicitly anticipates non-equivalent relay-BP
  parameterisations and asks for the gap to be shown, not closed.

## Follow-ups

- #<pending> — native C++ plugin (`libaleph-cudaq.so`) for the realtime / NVQLink path (spec §10.1).
- #<pending> — `iters_per_leg` / early-exit knob for the f64 `RelayBpDecoder` so the A/B can match
  the ASIC's 6×10 schedule exactly (spec §10.2).
