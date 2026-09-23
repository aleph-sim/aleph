"""Decode throughput (shots/s) of aleph decoders vs pymatching / ldpc on stim DEMs.

Usage: python scripts/python/bench_qec.py  (prints a Markdown table)
"""
import platform
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


def rate(fn, packed, shots):
    fn(packed[: min(200, shots)])  # warm-up
    t = time.perf_counter()
    fn(packed)
    return shots / (time.perf_counter() - t)


def main():
    rows = []
    # (label, circuit, decompose_errors, decoder names, shot count)
    # surface d=9's relay-bp-osd is slow enough that 20000 shots would take
    # ~60s on this machine; that case runs at 5000 shots instead (still >=2000)
    # and the table states the count per row.
    cases = [("surface d=5", surface(5), True, ["mwpm", "union-find-weighted", "bp-osd", "relay-bp-osd"], 20000),
             ("surface d=9", surface(9), True, ["mwpm", "union-find-weighted", "relay-bp-osd"], 5000),
             ("color d=5", color(5), False, ["bp-osd", "relay-bp", "relay-bp-osd"], 20000)]
    for label, circ, decompose, names, shots in cases:
        sdem = circ.detector_error_model(decompose_errors=decompose)
        packed, _ = circ.compile_detector_sampler(seed=1).sample(shots, separate_observables=True, bit_packed=True)
        dem = qec.DetectorErrorModel(sdem)
        for n in names:
            dec = qec.Decoder(dem, n)
            rows.append((label, f"aleph {n}", rate(dec.decode_batch_bit_packed, packed, shots), shots))
        if decompose:
            try:
                import pymatching
                m = pymatching.Matching.from_detector_error_model(sdem)
                dense = np.unpackbits(packed, axis=1, bitorder="little", count=sdem.num_detectors)
                rows.append((label, "pymatching", rate(m.decode_batch, dense, shots), shots))
            except ImportError:
                pass
        else:
            try:
                from ldpc.sinter_decoders import SinterBpOsdDecoder
                c = SinterBpOsdDecoder().compile_decoder_for_dem(dem=sdem)
                rows.append((label, "ldpc bp-osd",
                             rate(lambda x: c.decode_shots_bit_packed(bit_packed_detection_event_data=x), packed, shots),
                             shots))
            except (ImportError, AttributeError, NotImplementedError):
                # Installed `ldpc` may not implement compile_decoder_for_dem
                # (inherited sinter.Decoder stub raises NotImplementedError) —
                # skip the row rather than fail the whole benchmark.
                pass
    print(f"Machine: {platform.platform()} / {platform.processor() or platform.machine()}\n")
    print("| DEM | decoder | shots | shots/s |\n|---|---|---:|---:|")
    for label, name, r, shots in rows:
        print(f"| {label} | {name} | {shots:,} | {r:,.0f} |")


if __name__ == "__main__":
    main()
