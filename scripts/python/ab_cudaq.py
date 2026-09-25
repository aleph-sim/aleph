"""A/B of aleph decoders vs NVIDIA's inside cudaq-qec: logical error rate + shots/s.

Run on the GPU box only:
  /root/cqvenv/bin/python scripts/python/ab_cudaq.py [--quick]
Prints Markdown tables; paste into docs/perf/f6-cudaq-plugin.md with the header block.

Workload G: gross [[144,12,12]] circuit-level DEM (aleph.qec.gross_code_dem), rounds=12,
  p in {0.001, 0.002, 0.003}; aleph-relay-bp(-osd) vs nv-qldpc-decoder in relay mode (+/-OSD).
Workload S: stim surface_code:rotated_memory_x d in {5, 9}, rounds=d, p=0.003, decomposed;
  aleph-mwpm / aleph-union-find-weighted vs cudaq pymatching vs nv-fusion-decoder.
Every decoder in a workload sees the identical (H, O, error_rate_vec) and the identical
syndrome/observable arrays. LER = shots with any wrong observable / shots, 95% Wilson CI.
"""
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np
import stim

import cudaq_qec as cq
import aleph
import aleph.qec as aq
import aleph.cudaq as ac

QUICK = "--quick" in sys.argv
SEED = 20260924

# NVIDIA relay-BP mode for the [[144,12,12]] gross code: the "canonical Relay BP settings for
# that code" from docs/sphinx/performance/nv_qldpc_relay_solutions_user_guide.rst (NVIDIA/cudaqx,
# 0.8.0) — bp_method=3 (disordered-memory min-sum), sequential relay composition (composition=1),
# sparse kernels, gamma0/gamma_dist/clip_value tuned for this code, pre_iter=80 / num_sets=60
# relay legs with stopping_criterion="All". We drop that doc's recording-only knobs
# (opt_results={"relay_solutions": True, ...}, output="observables", bp_batch_size) since we are
# not doing an offline stop_nconv sweep here, and we drop proc_float="fp32" to keep NVIDIA's
# numeric precision comparable to aleph's fp64 path (this doc's own default omits it too; fp32 is
# an extra speed knob orthogonal to the relay schedule). The brief's original guess
# (pre_iter=60, num_sets=4, stopping_criterion="All", max_iterations=100) appears nowhere in
# NVIDIA's docs/examples for this code; these documented values replace it.
NV_RELAY = dict(use_sparsity=True, bp_method=3, composition=1, max_iterations=60,
                gamma0=0.125, gamma_dist=[-0.24, 0.66], clip_value=200.0, repeatable=True,
                srelay_config=dict(pre_iter=80, num_sets=60, stopping_criterion="NConv", stop_nconv=5))
# stopping_criterion="NConv", stop_nconv=5 (rather than the doc's "All", which is only meant for
# an offline stop_nconv *recording* run) is NVIDIA's own published conclusion for this exact code:
# nv_qldpc_relay_solutions_user_guide.rst measures the RelayBP-N sweep and states "for this code
# and noise, stop_nconv=5 buys all of the measured accuracy at a small fraction of the cost of
# larger N" (LER falls ~5x from N=1 to N=5 then saturates, while mean iterations keep growing
# linearly in N). Confirmed empirically here too: "All" (running the full 60-leg schedule on every
# shot regardless of convergence) measured ~6.5x slower than NConv=5 on an identical 2000-shot
# batch with no LER-relevant difference expected per the doc's own sweep.
# bp_batch_size batches shots through one GPU dispatch instead of one kernel launch per shot (the
# decoder itself warns "called with default bp_batch_size=1 ... Set bp_batch_size > 1 at
# construction to decode syndromes in parallel"); NVIDIA's canonical-settings run for this code
# uses bp_batch_size=1000, capped to the batch's own shot count for --quick's smaller batches.
NV_BATCH = 1000
ALEPH_OSD = dict(osd_order=12)   # docs/perf/qec-q5-circuit-dem.md settings; relay defaults otherwise


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def sample(dem_text, shots, seed):
    s = stim.DetectorErrorModel(dem_text).compile_sampler(seed=seed)
    dets, obs, _ = s.sample(shots)
    return dets.astype(np.uint8), obs.astype(np.uint8)


def run_decoder(name, H, O, rates, dets, obs, params, threads=None):
    """(LER, lo, hi, shots/s, nonconverged) for cudaq decoder `name` on this batch."""
    if threads is not None:
        return _run_in_child(name, H, O, rates, dets, obs, params, threads)
    d = cq.get_decoder(name, H, O=O, error_rate_vec=rates.tolist(), **params)
    rows = dets.astype(np.float64).tolist()
    d.decode_batch(rows[:200])                       # warm-up
    best = 0.0
    for _ in range(3):
        t = time.perf_counter()
        br = d.decode_batch(rows)
        best = max(best, len(rows) / (time.perf_counter() - t))
    # Result basis differs by decoder even though every decoder here is constructed with the same
    # O: aleph's own decode_batch (aleph.cudaq.AlephDecoder) always returns per-mechanism error
    # estimates regardless of O (per its docstring), while cudaq-qec's native decoders
    # (nv-qldpc-decoder, pymatching) switch to observable-space output unconditionally once O is
    # supplied at construction, and nv-fusion-decoder switches on the explicit output="observables"
    # kwarg this harness passes. Detect which basis came back from the width actually returned
    # rather than hardcoding it per decoder name, so a decoder changing its convention fails loud
    # (RuntimeError) instead of silently comparing the wrong axis.
    r = np.asarray(br.result)
    if r.ndim == 1:
        r = r.reshape(len(rows), -1)
    # Tripwire: the branch below assumes O.shape[0] (observables) and O.shape[1] (mechanisms)
    # are distinguishable widths. Every workload in this harness has far fewer observables than
    # mechanisms, but if that ever stopped holding, silently picking the wrong branch would
    # compare the wrong axis without either branch raising — fail loud instead.
    assert O.shape[0] != O.shape[1], (
        f"{name}: ambiguous result basis (O has {O.shape[0]} observables == {O.shape[1]} "
        "mechanisms; width alone can't tell errors-space from observables-space apart)")
    if r.shape[1] == O.shape[0]:
        pred = (r > 0.5).astype(np.uint8)
    elif r.shape[1] == O.shape[1]:
        ehat = (r > 0.5).astype(np.uint8)
        pred = (ehat @ O.T) % 2
    else:
        raise RuntimeError(f"{name}: result width {r.shape[1]} matches neither "
                           f"O's {O.shape[0]} observables nor its {O.shape[1]} mechanisms")
    wrong = int((pred != obs).any(axis=1).sum())
    nconv = int((~np.asarray(br.converged, dtype=bool)).sum())
    return (*wilson(wrong, len(rows)), best, nconv)


def _run_in_child(name, H, O, rates, dets, obs, params, threads):
    """aleph decoders size rayon's pool once per process: 1-thread numbers come from a child."""
    import json, tempfile
    with tempfile.TemporaryDirectory() as td:
        np.savez(f"{td}/in.npz", H=H, O=O, rates=rates, dets=dets, obs=obs)
        code = (
            "import json,sys,numpy as np,warnings; warnings.simplefilter('ignore');"
            "sys.path.insert(0, %r); import ab_cudaq as m; z=np.load(%r);"
            "print(json.dumps(m.run_decoder(%r, z['H'], z['O'], z['rates'], z['dets'], z['obs'], json.loads(%r))))"
            % (os.path.dirname(os.path.abspath(__file__)), f"{td}/in.npz", name, json.dumps(params))
        )
        out = subprocess.run([sys.executable, "-c", code], env={**os.environ, "RAYON_NUM_THREADS": str(threads)},
                             capture_output=True, text=True, check=True).stdout
        return tuple(json.loads(out.strip().splitlines()[-1]))


def row(workload, decoder, cfg, threads, shots, r):
    ler, lo, hi, rate, nconv = r
    return f"| {workload} | {decoder} | {cfg} | {threads} | {shots:,} | {ler:.2e} [{lo:.1e}, {hi:.1e}] | {nconv} | {rate:,.0f} |"


HEADER = "| workload | decoder | config | threads | shots | LER [95% CI] | non-conv | shots/s |\n|---|---|---|---:|---:|---|---:|---:|"


def workload_g():
    print("\n### Workload G — gross [[144,12,12]], rounds=12, circuit-level uniform p\n")
    print(HEADER)
    for p in ([0.003] if QUICK else [0.001, 0.002, 0.003]):
        dem = aq.gross_code_dem(12, p)
        H, O, rates = ac.dem_to_matrices(dem)
        # 20,000 shots at every p (not 100,000 at p<=0.001 as originally planned): measured on the
        # box, aleph-relay-bp-osd at RAYON_NUM_THREADS=1 runs ~60 shots/s, so 100,000 shots x 3
        # repeats would cost ~80 minutes for that single cell alone. The OSD rows' LER is already
        # at or near the floor (0 errors) at p=0.001 in 1,000-shot runs (docs/perf/qec-q5-circuit-
        # dem.md), so the extra 80,000 shots would not change that row's conclusion; the no-OSD
        # rows still get a full 20,000-shot Wilson CI, satisfying the spec's ">= 10,000 shots"
        # floor at every p.
        shots = 2000 if QUICK else 20_000
        dets, obs = sample(dem.to_dem_string(), shots, SEED)
        w = f"gross p={p}"
        # aleph's relay-bp(-osd) at RAYON_NUM_THREADS=1 measured ~60-75 shots/s on this box (a
        # single core doing 4-leg relay-BP + OSD-12's combination sweep per shot); at the full
        # 20,000-shot batch x 3 repeats that is ~15-16 minutes PER cell, ~90 minutes for the two
        # 1-thread rows across all three p values alone. The LER these two cells report is already
        # covered at full statistical power by the identical decoder's 20-core row two lines above
        # (same algorithm, same data — thread count does not change which answer it converges to,
        # only how fast); the 1-thread row exists to report *throughput*, so it uses a 2,000-shot
        # prefix of the same sampled batch instead, with its own (wider-CI) LER reported honestly
        # rather than dropped.
        n1 = shots if QUICK else min(shots, 2000)
        for cfg_name, name, params in [
            ("relay+OSD-12", "aleph-relay-bp-osd", ALEPH_OSD),
            ("relay, no OSD", "aleph-relay-bp", {}),
        ]:
            try:
                print(row(w, name, cfg_name, os.cpu_count(), shots, run_decoder(name, H, O, rates, dets, obs, params)))
            except Exception as e:  # report, never drop — same pattern as the NVIDIA rows below
                print(f"| {w} | {name} | {cfg_name} | {os.cpu_count()} | {shots:,} | ERROR: {str(e)[:120]} | | |")
            try:
                print(row(w, name, cfg_name, 1, n1, run_decoder(name, H, O, rates, dets[:n1], obs[:n1], params, threads=1)))
            except Exception as e:
                print(f"| {w} | {name} | {cfg_name} | 1 | {n1:,} | ERROR: {str(e)[:120]} | | |")
        nv_batch = min(NV_BATCH, shots)
        for cfg_name, params in [
            ("relay+OSD-12", {**NV_RELAY, "use_osd": True, "osd_order": 12, "osd_method": 1, "bp_batch_size": nv_batch}),
            ("relay, no OSD", {**NV_RELAY, "use_osd": False, "bp_batch_size": nv_batch}),
        ]:
            try:
                print(row(w, "nv-qldpc-decoder", cfg_name, "GPU", shots, run_decoder("nv-qldpc-decoder", H, O, rates, dets, obs, params)))
            except Exception as e:  # report, never drop
                print(f"| {w} | nv-qldpc-decoder | {cfg_name} | GPU | {shots:,} | ERROR: {str(e)[:120]} | | |")


def surface(d, p=0.003):
    return stim.Circuit.generated("surface_code:rotated_memory_x", distance=d, rounds=d,
                                  after_clifford_depolarization=p, before_round_data_depolarization=p,
                                  before_measure_flip_probability=p, after_reset_flip_probability=p)


def workload_s():
    print("\n### Workload S — surface_code:rotated_memory_x, rounds=d, p=0.003, decomposed\n")
    print(HEADER)
    for d, shots in ([(5, 2000)] if QUICK else [(5, 20_000), (9, 5_000)]):
        circ = surface(d)
        dem = circ.detector_error_model(decompose_errors=True)
        H, O, rates = ac.dem_to_matrices(dem)
        dets, obs = circ.compile_detector_sampler(seed=SEED).sample(shots, separate_observables=True)
        dets, obs = dets.astype(np.uint8), obs.astype(np.uint8)
        w = f"surface d={d}"
        for name in ["aleph-mwpm", "aleph-union-find-weighted"]:
            try:
                print(row(w, name, "-", os.cpu_count(), shots, run_decoder(name, H, O, rates, dets, obs, {})))
            except Exception as e:  # report, never drop — same pattern as the NVIDIA rows below
                print(f"| {w} | {name} | - | {os.cpu_count()} | {shots:,} | ERROR: {str(e)[:120]} | | |")
            try:
                print(row(w, name, "-", 1, shots, run_decoder(name, H, O, rates, dets, obs, {}, threads=1)))
            except Exception as e:
                print(f"| {w} | {name} | - | 1 | {shots:,} | ERROR: {str(e)[:120]} | | |")
        # nv-fusion-decoder is constructed from raw (H, O) rather than DEM text, so it has no
        # detector coordinates to derive a temporal layout from and needs one supplied explicitly
        # (docs/sphinx/api/qec/nv_fusion_decoder_api.rst: "detector_round ... Takes highest
        # priority over all automatic derivation paths"; without it or a DEM, construction is
        # rejected — the exact "scaffold not yet built" error hit here first). Stim's own detector
        # coordinates give the round directly: the circuit's third coordinate per detector.
        # output="observables" is likewise required — "supplying O ... does not select observable
        # output" per the same doc — otherwise decode_batch would return per-mechanism errors.
        detector_round = np.array([int(round(circ.get_detector_coordinates()[i][-1]))
                                    for i in range(H.shape[0])], dtype=np.int32)
        for name, params in [("pymatching", {}),
                             ("nv-fusion-decoder", {"num_threads": os.cpu_count(), "output": "observables",
                                                    "detector_round": detector_round})]:
            try:
                print(row(w, name, "defaults", "1" if name == "pymatching" else os.cpu_count(), shots,
                          run_decoder(name, H, O, rates, dets, obs, params)))
            except Exception as e:
                print(f"| {w} | {name} | defaults | | {shots:,} | ERROR: {str(e)[:120]} | | |")


if __name__ == "__main__":
    import cudaq
    print(f"aleph {aleph.version()}, cudaq-qec {cq.__version__}, cudaq {cudaq.__version__}, stim {stim.__version__}, "
          f"numpy {np.__version__}, {platform.platform()}, {os.cpu_count()} CPUs, python {platform.python_version()}")
    print(subprocess.run(["bash", "-c", "uptime; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || cat /proc/driver/nvidia/version | head -1"],
                         capture_output=True, text=True).stdout.strip())
    workload_g()
    workload_s()
