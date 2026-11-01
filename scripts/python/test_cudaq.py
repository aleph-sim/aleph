"""aleph.cudaq: aleph decoders registered as cudaq-qec decoders.

Skips unless cudaq_qec is importable (it is not in CI; run on the GPU box:
  /root/cqvenv/bin/python -m unittest scripts.python.test_cudaq -v).
"""
import unittest

import numpy as np

try:
    import cudaq_qec as cq
    import aleph.qec as aq
    import aleph.cudaq as ac
    HAVE = True
except ImportError:
    HAVE = False

try:
    import stim
    HAVE_STIM = True
except ImportError:
    HAVE_STIM = False

DEM = "error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1\n"
H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
O = np.array([[1, 0, 0]], dtype=np.uint8)
RATES = [0.1, 0.1, 0.1]


@unittest.skipUnless(HAVE, "needs cudaq_qec + aleph")
class TestRegistration(unittest.TestCase):
    def test_all_seven_resolve(self):
        self.assertEqual(sorted(ac.NAMES), sorted(aq.decoder_names()))
        for name in ac.NAMES:
            d = cq.get_decoder(f"aleph-{name}", H, O=O, error_rate_vec=RATES)
            self.assertEqual((d.get_block_size(), d.get_syndrome_size()), (3, 2), name)

    def test_decode_contract(self):
        for name in ac.NAMES:
            d = cq.get_decoder(f"aleph-{name}", H, O=O, error_rate_vec=RATES)
            r = d.decode([1.0, 0.0])
            self.assertIsInstance(r, cq.DecoderResult)
            self.assertEqual(len(r.result), 3, name)
            self.assertTrue(r.converged, name)
            ehat = (np.asarray(r.result) > 0.5).astype(np.uint8)
            np.testing.assert_array_equal((H @ ehat) % 2, [1, 0], name)
            self.assertEqual(((O @ ehat) % 2).tolist(), [1], name)  # D0 alone -> boundary edge with L0

    def test_decode_batch_contract(self):
        dets = np.array([[1, 0], [0, 0], [1, 1], [0, 1]], dtype=np.uint8)
        ref = aq.Decoder(aq.dem_from_matrices(H, O, RATES), "mwpm").decode_batch(dets.astype(bool))
        d = cq.get_decoder("aleph-mwpm", H, O=O, error_rate_vec=RATES)
        br = d.decode_batch(dets.astype(np.float64).tolist())
        self.assertIsInstance(br, cq.BatchDecoderResult)
        self.assertEqual(br.result.shape, (4, 3))
        self.assertEqual(br.converged.tolist(), [True] * 4)
        ehat = (br.result > 0.5).astype(np.uint8)
        np.testing.assert_array_equal((ehat @ O.T % 2).astype(bool), ref)

    def test_numpy_layouts_and_thresholds_match_native_errors(self):
        values = np.array([[0.5, np.nextafter(0.5, 1.0)], [np.nan, np.inf],
                           [-np.inf, -1.0], [0.7, 0.3]])
        for name in ac.NAMES:
            decoder = cq.get_decoder(f"aleph-{name}", H, O=O, error_rate_vec=RATES)
            native = aq.Decoder(aq.dem_from_matrices(H, O, RATES), name)
            for rows in (values, values[::-1], values[:, ::-1], values.tolist(),
                         values > .5, (values > .5).astype(np.uint8) * 255):
                expected, conv = native.decode_batch_errors((np.asarray(rows) > .5).astype(np.uint8))
                result = decoder.decode_batch(rows)
                np.testing.assert_array_equal(result.result, expected)
                np.testing.assert_array_equal(result.converged, conv)
                self.assertEqual(result.result.dtype, np.float64)

    def test_empty_shape_validation(self):
        decoder = cq.get_decoder("aleph-mwpm", H, error_rate=.1)
        for bad in (np.empty((0, 3)), np.empty((1, 0)), np.empty((0, 2, 0))):
            with self.assertRaisesRegex(ValueError, "width"):
                decoder.decode_batch(bad)
        # Zero detectors still represent N shots, rather than an empty batch.
        rows = ac._to_bits(np.empty((3, 0)), 0)
        self.assertEqual(rows.shape, (3, 0))

    def test_decode_batch_empty(self):
        d = cq.get_decoder("aleph-bp", H, O=O, error_rate_vec=RATES)
        br = d.decode_batch([])
        # (0, E), not (0, 0): cudaq's own `(O @ err.T)` breaks on a (0, 0) result (Step-0 probe).
        self.assertEqual(br.result.shape, (0, 3))
        self.assertEqual(br.converged.shape, (0,))

    def test_soft_syndrome_thresholds(self):
        d = cq.get_decoder("aleph-mwpm", H, O=O, error_rate_vec=RATES)
        hard = np.asarray(d.decode([1.0, 0.0]).result)
        soft = np.asarray(d.decode([0.7, 0.3]).result)
        np.testing.assert_array_equal(hard, soft)

    def test_scalar_error_rate_and_missing_rates(self):
        d = cq.get_decoder("aleph-relay-bp", H, O=O, error_rate=0.1)
        self.assertEqual(len(d.decode([0.0, 1.0]).result), 3)
        with self.assertRaisesRegex(ValueError, "error_rate"):
            cq.get_decoder("aleph-relay-bp", H, O=O)

    def test_both_error_rate_forms_rejected(self):
        with self.assertRaisesRegex(ValueError, "not both"):
            cq.get_decoder("aleph-relay-bp", H, O=O, error_rate_vec=RATES, error_rate=0.1)

    def test_unknown_param_and_wrong_width(self):
        with self.assertRaisesRegex(ValueError, "osd_order"):
            cq.get_decoder("aleph-mwpm", H, O=O, error_rate_vec=RATES, osd_order=2)
        d = cq.get_decoder("aleph-mwpm", H, O=O, error_rate_vec=RATES)
        with self.assertRaisesRegex(ValueError, "2"):
            d.decode([1.0, 0.0, 0.0])

    def test_matching_decoder_rejects_hyperedge_column(self):
        H3 = np.array([[1], [1], [1]], dtype=np.uint8)
        for name in ("mwpm", "union-find", "union-find-weighted"):
            with self.assertRaisesRegex(ValueError, "graph-?like"):
                cq.get_decoder(f"aleph-{name}", H3, error_rate_vec=[0.1])
        cq.get_decoder("aleph-bp", H3, error_rate_vec=[0.1])  # BP is fine with hyperedges

    def test_matching_decoder_rejects_prior_above_half(self):
        for name in ("mwpm", "union-find", "union-find-weighted"):
            for probability in (np.nextafter(0.5, 1.0), 0.6, 1.0):
                for kwargs in ({"error_rate_vec": [0.1, probability, 0.1]},
                               {"error_rate": probability}):
                    with self.subTest(name=name, kwargs=kwargs):
                        with self.assertRaisesRegex(ValueError, r"matching prior at column [01].*0.5"):
                            cq.get_decoder(f"aleph-{name}", H, O=O, **kwargs)
        for name in ("bp", "bp-osd", "relay-bp", "relay-bp-osd"):
            cq.get_decoder(f"aleph-{name}", H, O=O, error_rate_vec=[0.1, 0.6, 0.1])

    def test_matching_prior_message_matches_native_python(self):
        rates = [0.1, 0.7, 0.1]
        dem = aq.dem_from_matrices(H, O, rates)
        for name in ("mwpm", "union-find", "union-find-weighted"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as native:
                    aq.Decoder(dem, name)
                with self.assertRaises(ValueError) as adapter:
                    cq.get_decoder(f"aleph-{name}", H, O=O, error_rate_vec=rates)
                self.assertEqual(str(adapter.exception), str(native.exception))

    def test_matching_boundary_priors_are_accepted(self):
        for name in ("mwpm", "union-find", "union-find-weighted"):
            for probability in (0.0, 0.5, np.nextafter(0.0, 1.0)):
                cq.get_decoder(f"aleph-{name}", H, O=O, error_rate=probability)
            # Independently constrained columns exercise zero and very large finite weights.
            for probability in (0.5, np.nextafter(0.0, 1.0)):
                dec = cq.get_decoder(f"aleph-{name}", np.ones((1, 1), dtype=np.uint8),
                                     O=np.ones((1, 1), dtype=np.uint8), error_rate=probability)
                result = dec.decode_batch(np.array([[0], [1]], dtype=float))
                self.assertTrue(np.asarray(result.converged).all())
                np.testing.assert_array_equal(np.asarray(result.result) > 0.5, [[0], [1]])

    def test_relay_budget_forwarding_and_typed_rejection(self):
        from test_qec import (RELAY_BUDGET_DEM, RELAY_BUDGET_SYNDROME,
                              RELAY_BUDGET_FULL, RELAY_BUDGET_FIRST)
        H, O, rates = aq.dem_to_matrices(RELAY_BUDGET_DEM)
        for name in ("relay-bp", "relay-bp-osd"):
            for iterations, early, expected in [(8, False, RELAY_BUDGET_FULL),
                                                 (8, True, RELAY_BUDGET_FIRST),
                                                 (1, False, RELAY_BUDGET_FIRST),
                                                 (0, False, RELAY_BUDGET_FIRST)]:
                with self.subTest(name=name, iterations=iterations, early=early):
                    for source, priors in [(H, {"O": O, "error_rate_vec": rates}),
                                            (RELAY_BUDGET_DEM, {})]:
                        dec = cq.get_decoder(f"aleph-{name}", source, **priors,
                                             legs=3, iters_per_leg=iterations, early_exit=early, seed=7)
                        result = dec.decode_batch(RELAY_BUDGET_SYNDROME.astype(float))
                        self.assertTrue(np.asarray(result.converged).all())
                        np.testing.assert_array_equal(np.asarray(result.result) > 0.5, expected)
                        single = dec.decode(RELAY_BUDGET_SYNDROME[0].astype(float))
                        self.assertTrue(single.converged)
                        np.testing.assert_array_equal(np.asarray(single.result) > 0.5, expected[0])
            with self.assertRaises((ValueError, TypeError, OverflowError)):
                cq.get_decoder(f"aleph-{name}", H, error_rate=0.1, iters_per_leg=-1)
            with self.assertRaises((ValueError, TypeError)):
                cq.get_decoder(f"aleph-{name}", H, error_rate=0.1, early_exit="false")
        with self.assertRaisesRegex(ValueError, "iters_per_leg"):
            cq.get_decoder("aleph-bp", H, error_rate=0.1, iters_per_leg=10)

    def test_scipy_sparse_h(self):
        import scipy.sparse as sp
        d = cq.get_decoder("aleph-union-find", sp.csr_matrix(H), O=sp.csr_matrix(O), error_rate_vec=RATES)
        self.assertEqual(d.get_block_size(), 3)

    def test_dem_to_matrices_cancels_duplicate_targets(self):
        # A target listed twice within one mechanism/part is Stim's "no flip" (parity), not a
        # second flip. Needs no stim: dem_to_matrices accepts a raw DEM string directly.
        Hm, Om, rates = ac.dem_to_matrices("error(0.1) D0 D0 D1 L0 L0\n")
        np.testing.assert_array_equal(Hm[:, 0], [0, 1])
        self.assertEqual(Om.shape, (1, 1))
        self.assertEqual(Om[0, 0], 0)
        np.testing.assert_array_equal(rates, [0.1])

    def test_dem_to_matrices_splits_hat(self):
        Hm, Om, rates = ac.dem_to_matrices("error(0.2) D0 ^ D1 L0\n")
        self.assertEqual(Hm.shape[1], 2)
        np.testing.assert_array_equal(Hm[:, 0], [1, 0])
        np.testing.assert_array_equal(Hm[:, 1], [0, 1])
        np.testing.assert_array_equal(Om[0], [0, 1])
        np.testing.assert_array_equal(rates, [0.2, 0.2])

    def test_nan_priors_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            cq.get_decoder("aleph-mwpm", H, O=O, error_rate=float("nan"))
        with self.assertRaisesRegex(ValueError, r"\[1\]"):
            cq.get_decoder("aleph-mwpm", H, O=O, error_rate_vec=[0.1, float("nan"), 0.1])

    def test_decode_empty_syndrome_raises(self):
        d = cq.get_decoder("aleph-mwpm", H, O=O, error_rate_vec=RATES)
        with self.assertRaisesRegex(ValueError, "2"):
            d.decode([])

    def test_decode_accepts_single_row_2d(self):
        d = cq.get_decoder("aleph-mwpm", H, O=O, error_rate_vec=RATES)
        # r.result comes back as a numpy array (not the Python list AlephDecoder.decode
        # assigned), so compare with assert_array_equal rather than assertEqual.
        np.testing.assert_array_equal(
            d.decode(np.array([[1.0, 0.0]])).result, d.decode([1.0, 0.0]).result)


@unittest.skipUnless(HAVE and HAVE_STIM, "needs cudaq_qec + aleph + stim")
class TestStimDem(unittest.TestCase):
    def circuit(self):
        return stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3,
                                      after_clifford_depolarization=0.003, before_round_data_depolarization=0.003,
                                      before_measure_flip_probability=0.003, after_reset_flip_probability=0.003)

    def test_dem_string_path_for_bp_family(self):
        dem = self.circuit().detector_error_model(decompose_errors=True)
        d = cq.get_decoder("aleph-relay-bp-osd", str(dem))  # cudaq injects O and error_rate_vec
        self.assertEqual(d.get_syndrome_size(), dem.num_detectors)
        with self.assertRaisesRegex(ValueError, "graph-?like"):
            cq.get_decoder("aleph-mwpm", str(dem))  # cudaq keeps `^` parts as one column

    def test_dem_to_matrices_makes_mwpm_agree_with_aleph_qec(self):
        circ = self.circuit()
        dem = circ.detector_error_model(decompose_errors=True)
        Hm, Om, rates = ac.dem_to_matrices(dem)
        self.assertEqual(Hm.shape[0], dem.num_detectors)
        self.assertEqual(Om.shape[0], dem.num_observables)
        self.assertTrue((Hm.sum(axis=0) <= 2).all(), "columns must be graphlike after splitting ^")
        dets, obs = circ.compile_detector_sampler(seed=3).sample(500, separate_observables=True)
        d = cq.get_decoder("aleph-mwpm", Hm, O=Om, error_rate_vec=rates)
        br = d.decode_batch(dets.astype(np.float64).tolist())
        pred = ((br.result > 0.5).astype(np.uint8) @ Om.T % 2).astype(bool)
        ref = aq.Decoder(aq.DetectorErrorModel(dem), "mwpm").decode_batch(dets)
        # Same model, same matcher; equal up to genuine ties (a few % at most at p=0.003, d=3).
        self.assertGreater((pred == ref).all(axis=1).mean(), 0.97)
        np.testing.assert_array_equal(((br.result > 0.5).astype(np.uint8) @ Hm.T % 2).astype(bool), dets)
