"""Tests for aleph.qec (DEM parsing + decoders) against stim-generated models.

Run after installing the wheel:  python -m unittest discover -s scripts/python -v
Skips cleanly when stim / pymatching are absent.
"""
import unittest

import numpy as np

try:
    import aleph.qec as qec
    HAVE_ALEPH = True
except ImportError:
    HAVE_ALEPH = False

try:
    import stim
    HAVE_STIM = True
except ImportError:
    HAVE_STIM = False

try:
    import pymatching
    HAVE_PYMATCHING = True
except ImportError:
    HAVE_PYMATCHING = False

ALL = ["mwpm", "union-find", "union-find-weighted", "bp", "bp-osd", "relay-bp", "relay-bp-osd"]
MATCHING = {"mwpm", "union-find", "union-find-weighted"}


def surface(d, p, rounds=None):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=d,
        rounds=rounds or d,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def color(d, p):
    return stim.Circuit.generated(
        "color_code:memory_xyz", distance=d, rounds=d,
        after_clifford_depolarization=p, before_measure_flip_probability=p,
    )


def xor_reduce_dem_text(text):
    """Replace every `error(p) A ^ B ...` line of a flat DEM by its parity-reduced targets."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("error") and "^" in s:
            head, _, targets = s.partition(")")
            counts = {}
            for tok in targets.split():
                if tok != "^":
                    counts[tok] = counts.get(tok, 0) + 1
            kept = [t for t, c in counts.items() if c % 2 == 1]
            s = head + ") " + " ".join(kept)
        out.append(s)
    return "\n".join(out) + "\n"


@unittest.skipUnless(HAVE_ALEPH, "aleph not installed")
class TestApi(unittest.TestCase):
    DEM = "error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1\n"

    def test_names(self):
        self.assertEqual(sorted(qec.decoder_names()), sorted(ALL))

    def test_dem_properties(self):
        dem = qec.DetectorErrorModel(self.DEM)
        self.assertEqual((dem.num_detectors, dem.num_observables, dem.num_errors), (2, 1, 3))
        self.assertEqual(qec.DetectorErrorModel(dem.to_dem_string()).to_dem_string(), dem.to_dem_string())

    def test_bad_inputs_raise_value_error(self):
        dem = qec.DetectorErrorModel(self.DEM)
        with self.assertRaises(ValueError):
            qec.DetectorErrorModel("frobnicate D0\n")
        with self.assertRaisesRegex(ValueError, "relay-bp"):
            qec.Decoder(dem, "nope")
        with self.assertRaisesRegex(ValueError, "osd_order"):
            qec.Decoder(dem, "mwpm", osd_order=2)
        with self.assertRaises(ValueError):
            qec.Decoder(dem, "bp", alpha=float("nan"))
        wide = qec.DetectorErrorModel("error(0.1) D0 L64\n")  # 65 observables
        for name in ALL:
            with self.assertRaisesRegex(ValueError, "at most 64"):
                qec.Decoder(wide, name)
        dec = qec.Decoder(dem, "mwpm")
        with self.assertRaises(ValueError):
            dec.decode_batch(np.zeros((4, 3), dtype=bool))  # wrong detector count
        with self.assertRaises(ValueError):
            dec.decode_batch_bit_packed(np.zeros((4, 2), dtype=np.uint8))  # needs 1 byte/row

    def test_decode_shapes_and_dtypes(self):
        dec = qec.Decoder(qec.DetectorErrorModel(self.DEM), "mwpm")
        self.assertEqual(dec.name, "mwpm")
        one = dec.decode(np.array([1, 0], dtype=np.uint8))
        self.assertEqual((one.dtype, one.shape), (np.bool_, (1,)))
        self.assertTrue(one[0])  # D0 alone -> boundary edge carrying L0
        batch = dec.decode_batch(np.array([[1, 0], [0, 0], [1, 1]], dtype=bool))
        self.assertEqual(batch.shape, (3, 1))
        self.assertEqual(batch[:, 0].tolist(), [True, False, False])
        packed = dec.decode_batch_bit_packed(np.array([[1], [0], [3]], dtype=np.uint8))
        self.assertEqual((packed.dtype, packed.shape), (np.uint8, (3, 1)))
        self.assertEqual(packed[:, 0].tolist(), [1, 0, 0])

    def test_dem_from_matrices_accepts_dense_int64_and_sparse(self):
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int64)
        O = np.array([[1, 0, 0]], dtype=np.int64)
        dem = qec.dem_from_matrices(H, O, [0.1, 0.2, 0.3])
        self.assertEqual((dem.num_detectors, dem.num_observables, dem.num_errors), (2, 1, 3))
        self.assertEqual(dem.to_dem_string(), qec.DetectorErrorModel("error(0.1) D0 L0\nerror(0.2) D0 D1\nerror(0.3) D1\n").to_dem_string())
        try:
            import scipy.sparse as sp
        except ImportError:
            return
        coo = sp.coo_matrix(([1, 1, 1, 0, 1], ([0, 0, 1, 1, 1], [0, 1, 1, 2, 2])), shape=(2, 3))  # explicit zero + duplicate at (1,2)
        dem2 = qec.dem_from_matrices(coo, sp.csr_matrix(O), np.array([0.1, 0.2, 0.3]))
        self.assertEqual(dem2.to_dem_string(), dem.to_dem_string())
        self.assertEqual(qec.dem_from_matrices(H, None, [0.1, 0.2, 0.3]).num_observables, 0)

    def test_dem_from_matrices_rejects_bad_input(self):
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "error_rate_vec"):
            qec.dem_from_matrices(H)
        with self.assertRaisesRegex(ValueError, "3 columns"):
            qec.dem_from_matrices(H, None, [0.1, 0.2])
        with self.assertRaisesRegex(ValueError, "column 1"):
            qec.dem_from_matrices(H, None, [0.1, float("nan"), 0.3])
        with self.assertRaisesRegex(ValueError, "at most 64"):
            qec.dem_from_matrices(H, np.ones((65, 3), dtype=np.uint8), [0.1, 0.2, 0.3])

    def test_decode_batch_errors(self):
        dem = qec.DetectorErrorModel(self.DEM)  # D0 L0 | D0 D1 | D1
        for name in ALL:
            dec = qec.Decoder(dem, name)
            self.assertEqual(dec.num_errors, 3)
            dets = np.array([[1, 0], [0, 0], [1, 1], [0, 1]], dtype=bool)
            ehat, conv = dec.decode_batch_errors(dets)
            self.assertEqual((ehat.dtype, ehat.shape, conv.dtype, conv.shape), (np.uint8, (4, 3), np.bool_, (4,)))
            # H ê = s (all converge on this tiny model) and O ê = decode_batch.
            H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
            self.assertTrue(conv.all(), name)
            np.testing.assert_array_equal((ehat @ H.T) % 2, dets.astype(np.uint8), name)
            O = np.array([[1, 0, 0]], dtype=np.uint8)
            np.testing.assert_array_equal(((ehat @ O.T) % 2).astype(bool), dec.decode_batch(dets), name)

    def test_decode_batch_errors_empty_batch(self):
        dec = qec.Decoder(qec.DetectorErrorModel(self.DEM), "mwpm")
        ehat, conv = dec.decode_batch_errors(np.zeros((0, 2), dtype=bool))
        self.assertEqual((ehat.shape, conv.shape), ((0, 3), (0,)))

    def test_gross_code_dem(self):
        dem = qec.gross_code_dem(2, 0.003)
        self.assertEqual(dem.num_observables, 12)
        self.assertGreater(dem.num_detectors, 100)
        with self.assertRaises(ValueError):
            qec.gross_code_dem(0, 0.003)
        with self.assertRaises(ValueError):
            qec.gross_code_dem(2, float("nan"))


@unittest.skipUnless(HAVE_ALEPH and HAVE_STIM, "needs aleph + stim")
class TestStimDems(unittest.TestCase):
    def test_compressed_equals_flattened(self):
        for circ, decompose in ((surface(3, 0.003), True), (color(3, 0.003), False)):
            dem = circ.detector_error_model(decompose_errors=decompose)
            a = qec.DetectorErrorModel(dem)
            b = qec.DetectorErrorModel(dem.flattened())
            self.assertEqual(a.to_dem_string(), b.to_dem_string())
            self.assertEqual(a.num_detectors, dem.num_detectors)
            self.assertEqual(a.num_observables, dem.num_observables)

    def test_every_decoder_on_decomposed_surface_code(self):
        circ = surface(3, 0.003)
        dem = qec.DetectorErrorModel(circ.detector_error_model(decompose_errors=True))
        dets, _ = circ.compile_detector_sampler(seed=7).sample(2000, separate_observables=True)
        packed = np.packbits(dets, axis=1, bitorder="little")
        for name in ALL:
            with self.subTest(name=name):
                dec = qec.Decoder(dem, name)
                batch = dec.decode_batch(dets)
                again = dec.decode_batch(dets)
                np.testing.assert_array_equal(batch, again)  # deterministic
                single = np.array([dec.decode(row) for row in dets[:200]])
                np.testing.assert_array_equal(batch[:200], single)
                pk = dec.decode_batch_bit_packed(packed)
                np.testing.assert_array_equal(
                    np.unpackbits(pk, axis=1, bitorder="little", count=batch.shape[1]).astype(bool),
                    batch,
                )

    def test_bp_family_uses_parity_reduced_view(self):
        # Spec §1.2: BP-family decoders see each `^` mechanism as the symmetric difference
        # of its parts, so a decomposed DEM must decode exactly like its XOR-reduced twin.
        circ = surface(5, 0.005)
        sdem = circ.detector_error_model(decompose_errors=True)
        self.assertIn("^", str(sdem.flattened()))
        reduced = qec.DetectorErrorModel(xor_reduce_dem_text(str(sdem.flattened())))
        decomposed = qec.DetectorErrorModel(sdem)
        dets, obs = circ.compile_detector_sampler(seed=5).sample(5000, separate_observables=True)
        for name in ("bp", "bp-osd", "relay-bp", "relay-bp-osd"):
            with self.subTest(name=name):
                a = qec.Decoder(decomposed, name).decode_batch(dets)
                b = qec.Decoder(reduced, name).decode_batch(dets)
                np.testing.assert_array_equal(a, b)

    def test_hypergraph_dem(self):
        circ = color(3, 0.003)
        dem = qec.DetectorErrorModel(circ.detector_error_model())
        for name in MATCHING:
            with self.assertRaisesRegex(ValueError, "graph"):
                qec.Decoder(dem, name)
        dets, obs = circ.compile_detector_sampler(seed=3).sample(5000, separate_observables=True)
        raw = int(np.any(obs, axis=1).sum())
        for name in ("bp-osd", "relay-bp-osd"):
            pred = qec.Decoder(dem, name).decode_batch(dets)
            errs = int(np.any(pred != obs, axis=1).sum())
            self.assertLess(errs, raw, f"{name}: {errs} >= undecoded {raw}")


@unittest.skipUnless(HAVE_ALEPH and HAVE_STIM and HAVE_PYMATCHING, "needs pymatching")
class TestOracle(unittest.TestCase):
    def test_mwpm_matches_pymatching(self):
        circ = surface(5, 0.005)
        sdem = circ.detector_error_model(decompose_errors=True)
        dets, obs = circ.compile_detector_sampler(seed=11).sample(20000, separate_observables=True)
        ours = qec.Decoder(qec.DetectorErrorModel(sdem), "mwpm").decode_batch(dets)
        theirs = pymatching.Matching.from_detector_error_model(sdem).decode_batch(dets).astype(bool)
        a = int(np.any(ours != obs, axis=1).sum())
        p = int(np.any(theirs != obs, axis=1).sum())
        self.assertLessEqual(abs(a - p), max(5, 0.1 * p), f"aleph {a} vs pymatching {p}")


if __name__ == "__main__":
    unittest.main()
