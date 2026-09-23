"""Decode throughput (shots/s) of aleph decoders vs pymatching / ldpc on stim DEMs.

Usage: python scripts/python/bench_qec.py  (prints a Markdown table)

aleph's batch decode runs on rayon's global pool, which is sized once at first use, so
every aleph decoder is measured twice: in this process on all cores, and in a child
process with RAYON_NUM_THREADS=1 (the per-core number, comparable to pymatching and
ldpc, which decode on one thread).
"""
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import stim

import aleph.qec as qec


def surface(d, p=0.003):
    return stim.Circuit.generated("surface_code:rotated_memory_x", distance=d, rounds=d,
                                  after_clifford_depolarization=p,
                                  before_round_data_depolarization=p,
                                  before_measure_flip_probability=p,
                                  after_reset_flip_probability=p)


def color(d, p=0.003):
    return stim.Circuit.generated("color_code:memory_xyz", distance=d, rounds=d,
                                  after_clifford_depolarization=p,
                                  before_measure_flip_probability=p)


# (label, circuit factory, decompose_errors, aleph decoder names, shot count).
# surface d=9's relay-bp-osd is slow enough that 20000 shots would take ~60s on this
# machine; that case runs at 5000 shots instead (still >=2000) and the table states
# the count per row.
CASES = [("surface d=5", lambda: surface(5), True, ["mwpm", "union-find-weighted", "bp-osd", "relay-bp-osd"], 20000),
         ("surface d=9", lambda: surface(9), True, ["mwpm", "union-find-weighted", "relay-bp-osd"], 5000),
         ("color d=5", lambda: color(5), False, ["bp-osd", "relay-bp", "relay-bp-osd"], 20000)]


def rate(fn, packed, shots):
    fn(packed[: min(200, shots)])  # warm-up
    t = time.perf_counter()
    fn(packed)
    return shots / (time.perf_counter() - t)


def sample(circ, shots):
    packed, _ = circ.compile_detector_sampler(seed=1).sample(shots, separate_observables=True, bit_packed=True)
    return packed


def aleph_rates():
    """{"label|name": shots/s} for every aleph decoder, on this process's rayon pool."""
    out = {}
    for label, make, decompose, names, shots in CASES:
        circ = make()
        dem = qec.DetectorErrorModel(circ.detector_error_model(decompose_errors=decompose))
        packed = sample(circ, shots)
        for n in names:
            out[f"{label}|{n}"] = rate(qec.Decoder(dem, n).decode_batch_bit_packed, packed, shots)
    return out


def main():
    all_threads = os.environ.get("RAYON_NUM_THREADS") or str(os.cpu_count())
    multi = aleph_rates()
    env = dict(os.environ, RAYON_NUM_THREADS="1")
    single = json.loads(subprocess.run([sys.executable, __file__, "--aleph-only"], env=env,
                                       check=True, capture_output=True, text=True).stdout)
    rows = []
    for label, make, decompose, names, shots in CASES:
        for n in names:
            rows.append((label, f"aleph {n}", all_threads, multi[f"{label}|{n}"], shots))
            rows.append((label, f"aleph {n}", "1", single[f"{label}|{n}"], shots))
        circ = make()
        sdem = circ.detector_error_model(decompose_errors=decompose)
        packed = sample(circ, shots)
        if decompose:
            try:
                import pymatching
                m = pymatching.Matching.from_detector_error_model(sdem)
                dense = np.unpackbits(packed, axis=1, bitorder="little", count=sdem.num_detectors)
                rows.append((label, "pymatching", "1", rate(m.decode_batch, dense, shots), shots))
            except ImportError:
                pass
        else:
            try:
                from ldpc.sinter_decoders import SinterBpOsdDecoder
                c = SinterBpOsdDecoder().compile_decoder_for_dem(dem=sdem)
                rows.append((label, "ldpc bp-osd", "1",
                             rate(lambda x: c.decode_shots_bit_packed(bit_packed_detection_event_data=x), packed, shots),
                             shots))
            except (ImportError, AttributeError, NotImplementedError):
                # Installed `ldpc` may not implement compile_decoder_for_dem
                # (inherited sinter.Decoder stub raises NotImplementedError) —
                # skip the row rather than fail the whole benchmark.
                pass
    print(f"Machine: {platform.platform()} / {platform.processor() or platform.machine()}"
          f" / {os.cpu_count()} logical CPUs\n")
    print("| DEM | decoder | threads | shots | shots/s |\n|---|---|---:|---:|---:|")
    for label, name, threads, r, shots in rows:
        print(f"| {label} | {name} | {threads} | {shots:,} | {r:,.0f} |")


if __name__ == "__main__":
    if "--aleph-only" in sys.argv:
        print(json.dumps(aleph_rates()))
    else:
        main()
