#!/usr/bin/env python3
"""Compare native split columns to the existing matrix round trip on real Stim shots.

Run with an installed current wheel plus numpy/stim. Emits CSV; no
performance measurements. Reuses the test suite's circuit fixture. Missing oracle
packages, projection failures or reference differences fail rather than skip.
"""
import argparse
import csv
import sys

import numpy as np
import test_qec as t


def mismatches(a, b):
    return int(np.count_nonzero(np.any(a != b, axis=1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shots', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=518)
    parser.add_argument('--probability', type=float, default=0.003)
    args = parser.parse_args()
    if args.shots <= 0:
        parser.error('--shots must be positive')
    if not (np.isfinite(args.probability) and 0 < args.probability < 0.5):
        parser.error('--probability must be finite and between 0 and 0.5')
    if not (t.HAVE_ALEPH and t.HAVE_STIM):
        raise RuntimeError('requires installed current aleph and stim')
    writer = csv.writer(sys.stdout)
    writer.writerow(['distance', 'decoder', 'shots', 'probability', 'original_columns', 'split_columns',
                     'native_reference_estimate_disagreement', 'syndrome_projection_failures',
                     'observable_vs_decode_disagreement', 'reference_observable_vs_decode_disagreement',
                     'original_decode_disagreement'])
    for d in [3, 5]:
        circ = t.surface(d, args.probability)
        dem = t.qec.DetectorErrorModel(circ.detector_error_model(decompose_errors=True))
        split = dem.split_components()
        H, O, rates = t.qec.dem_to_matrices(dem)
        refdem = t.qec.dem_from_matrices(H, O, rates)
        dets, _ = circ.compile_detector_sampler(seed=args.seed).sample(args.shots, separate_observables=True)
        dets = dets.astype(np.uint8)
        for name in sorted(t.MATCHING):
            dec, ref = t.qec.Decoder(split, name), t.qec.Decoder(refdem, name)
            estimate, conv = dec.decode_batch_errors(dets)
            expected, expected_conv = ref.decode_batch_errors(dets)
            pred, refpred = dec.decode_batch(dets), ref.decode_batch(dets)
            original = t.qec.Decoder(dem, name).decode_batch(dets)
            estimate_diff = mismatches(estimate, expected)
            syndrome_diff = mismatches(estimate @ H.T % 2, dets)
            observable_diff = mismatches((estimate @ O.T % 2).astype(bool), pred)
            reference_observable_diff = mismatches((expected @ O.T % 2).astype(bool), refpred)
            original_diff = mismatches(pred, original)
            if (estimate_diff or syndrome_diff or original_diff or
                    observable_diff != reference_observable_diff or
                    (name != 'mwpm' and observable_diff) or
                    not conv.all() or not np.array_equal(conv, expected_conv)):
                raise AssertionError((d, name, 'native split contract/reference mismatch'))
            writer.writerow([d, name, args.shots, args.probability, dem.num_errors, split.num_errors,
                             estimate_diff, syndrome_diff, observable_diff,
                             reference_observable_diff, original_diff])


if __name__ == '__main__':
    main()
