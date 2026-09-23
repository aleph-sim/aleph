"""aleph.sinter adapters: picklable, and usable by sinter.collect with worker processes."""
import pickle
import unittest

import numpy as np

try:
    import sinter
    import stim
    import aleph.sinter as asinter
    HAVE = True
except ImportError:
    HAVE = False


@unittest.skipUnless(HAVE, "needs aleph + stim + sinter")
class TestSinter(unittest.TestCase):
    def test_all_names_present_and_picklable(self):
        import aleph.qec as qec
        decs = asinter.decoders(params={"relay-bp": {"legs": 2}})
        self.assertEqual(sorted(decs), sorted(f"aleph-{n}" for n in qec.decoder_names()))
        for key, d in decs.items():
            back = pickle.loads(pickle.dumps(d))
            self.assertEqual((back.name, back.params), (d.name, d.params), key)
        self.assertEqual(decs["aleph-relay-bp"].params, {"legs": 2})

    def test_compiled_decoder_bit_packed(self):
        circ = stim.Circuit.generated("repetition_code:memory", distance=5, rounds=5,
                                      after_clifford_depolarization=0.01)
        dem = circ.detector_error_model(decompose_errors=True)
        dets, obs = circ.compile_detector_sampler(seed=5).sample(
            1000, separate_observables=True, bit_packed=True)
        comp = asinter.decoders()["aleph-mwpm"].compile_decoder_for_dem(dem=dem)
        pred = comp.decode_shots_bit_packed(bit_packed_detection_event_data=dets)
        self.assertEqual((pred.dtype, pred.shape), (np.uint8, obs.shape))

    def test_collect_with_workers(self):
        circ = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=3,
                                      after_clifford_depolarization=0.005)
        tasks = [sinter.Task(circuit=circ, json_metadata={"d": 3})]
        stats = sinter.collect(num_workers=2, tasks=tasks,
                               decoders=["aleph-mwpm", "aleph-relay-bp-osd"],
                               custom_decoders=asinter.decoders(),
                               max_shots=2000, max_errors=50)
        self.assertEqual(sorted(s.decoder for s in stats), ["aleph-mwpm", "aleph-relay-bp-osd"])
        for s in stats:
            self.assertGreater(s.shots, 0)


if __name__ == "__main__":
    unittest.main()
