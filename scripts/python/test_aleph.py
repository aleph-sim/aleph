"""Behaviour tests for the aleph Python bindings (P4-08).

Gated per-PR by the `test-python` CI job (P4-10). Also runnable from the
repo root after installing the wheel:

    python -m unittest discover -s scripts/python -v
"""
import math
import os
import unittest

import numpy as np

try:
    import aleph
    HAVE_ALEPH = True
except ImportError:
    HAVE_ALEPH = False

BELL_QASM = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
"""

BELL_FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "..", "oracle", "circuits", "bell_phi_plus.qasm"
)


@unittest.skipUnless(HAVE_ALEPH, "aleph extension module not installed")
class TestVersion(unittest.TestCase):
    def test_version_attribute_and_function_agree(self):
        # Both come from CARGO_PKG_VERSION — the same source maturin uses
        # for the wheel version.
        self.assertRegex(aleph.__version__, r"^\d+\.\d+\.\d+$")
        self.assertEqual(aleph.version(), aleph.__version__)


@unittest.skipUnless(HAVE_ALEPH, "aleph extension module not installed")
class TestCircuitBuilder(unittest.TestCase):
    def test_bell_counts(self):
        c = aleph.Circuit(2)
        c.h(0)
        c.cx(0, 1)
        r = aleph.run(c, shots=4096, seed=7)
        counts = r.counts()
        self.assertEqual(set(counts), {"00", "11"})
        self.assertEqual(sum(counts.values()), 4096)
        for v in counts.values():
            self.assertGreater(v, 1700)  # ~50/50 within generous slack

    def test_seed_reproducible(self):
        def go():
            c = aleph.Circuit(3)
            c.h(0)
            c.cx(0, 1)
            c.cx(1, 2)
            return aleph.run(c, shots=256, seed=42).counts()

        self.assertEqual(go(), go())

    def test_statevector_h(self):
        c = aleph.Circuit(1)
        c.h(0)
        amps = aleph.run(c, shots=1, seed=0).statevector()
        # P4-11: a numpy complex128 array, not a list of Python complex.
        self.assertIsInstance(amps, np.ndarray)
        self.assertEqual(amps.dtype, np.complex128)
        self.assertEqual(amps.shape, (2,))
        np.testing.assert_allclose(amps, [1 / math.sqrt(2), 1 / math.sqrt(2)], atol=1e-10)

    def test_statevector_bell_amplitudes(self):
        # |Φ+⟩ = (|00⟩ + |11⟩)/√2 — shape (4,), amps 0 and 3 are 1/√2.
        c = aleph.Circuit(2)
        c.h(0)
        c.cx(0, 1)
        amps = aleph.run(c, shots=1, seed=0).statevector()
        self.assertEqual(amps.shape, (2**2,))
        self.assertEqual(amps.dtype, np.complex128)
        np.testing.assert_allclose(
            amps, [1 / math.sqrt(2), 0.0, 0.0, 1 / math.sqrt(2)], atol=1e-10
        )

    def test_ghz_backends_agree(self):
        outcome_sets = {}
        for be in ("sv", "mps", "stab"):
            c = aleph.Circuit(4)
            c.h(0)
            for q in range(3):
                c.cx(q, q + 1)
            counts = aleph.run(c, shots=2048, seed=1, backend=be).counts()
            self.assertEqual(sum(counts.values()), 2048)
            outcome_sets[be] = set(counts)
        self.assertEqual(outcome_sets["sv"], {"0000", "1111"})
        self.assertEqual(outcome_sets["sv"], outcome_sets["mps"])
        self.assertEqual(outcome_sets["sv"], outcome_sets["stab"])

    def test_non_clifford_on_stab_raises(self):
        c = aleph.Circuit(1)
        c.t(0)
        with self.assertRaises(ValueError):
            aleph.run(c, backend="stab")

    def test_unknown_backend_raises(self):
        c = aleph.Circuit(1)
        with self.assertRaises(ValueError):
            aleph.run(c, backend="gpu")

    def test_statevector_unavailable_on_mps(self):
        c = aleph.Circuit(2)
        c.h(0)
        r = aleph.run(c, backend="mps", seed=0)
        with self.assertRaises(ValueError):
            r.statevector()

    def test_from_qasm_matches_builder(self):
        a = aleph.run(aleph.Circuit.from_qasm(BELL_QASM), shots=1024, seed=3).counts()
        c = aleph.Circuit(2)
        c.h(0)
        c.cx(0, 1)
        b = aleph.run(c, shots=1024, seed=3).counts()
        self.assertEqual(a, b)

    @unittest.skipUnless(os.path.exists(BELL_FIXTURE), "repo fixture not present")
    def test_from_qasm_file_fixture(self):
        c = aleph.Circuit.from_qasm_file(BELL_FIXTURE)
        self.assertEqual(c.num_qubits, 2)
        counts = aleph.run(c, shots=512, seed=5).counts()
        self.assertEqual(set(counts), {"00", "11"})

    def test_qasm_parse_error(self):
        with self.assertRaises(ValueError):
            aleph.Circuit.from_qasm("OPENQASM 3.0;\nqubit[2] q;\nnotagate q[0];\n")

    def test_duplicate_qubit_raises(self):
        c = aleph.Circuit(2)
        with self.assertRaises(ValueError):
            c.cx(0, 0)

    def test_out_of_range_raises(self):
        c = aleph.Circuit(2)
        with self.assertRaises(ValueError):
            c.h(5)

    def test_cp_pi_equals_cz(self):
        # cp(π) ≡ CZ: on (H⊗H)|00⟩ the |11⟩ amplitude flips sign.
        c = aleph.Circuit(2)
        c.h(0)
        c.h(1)
        c.cp(math.pi, 0, 1)
        amps = aleph.run(c, shots=1, seed=0).statevector()
        self.assertAlmostEqual(amps[0].real, 0.5, places=10)
        self.assertAlmostEqual(amps[3].real, -0.5, places=10)

    def test_num_gates_property(self):
        c = aleph.Circuit(2)
        c.h(0)
        c.cx(0, 1)
        c.measure(0, 0)
        c.barrier([0, 1])
        self.assertEqual(c.num_gates, 2)
        self.assertEqual(c.num_qubits, 2)
        self.assertEqual(c.num_clbits, 2)


@unittest.skipUnless(HAVE_ALEPH, "aleph extension module not installed")
class TestNoise(unittest.TestCase):
    def _bell(self):
        c = aleph.Circuit(2)
        c.h(0)
        c.cx(0, 1)
        return c

    def test_empty_model_matches_noiseless(self):
        # An empty NoiseModel must reproduce the noiseless distribution
        # (same seed). run() with noise= is per-shot Monte-Carlo, but with
        # no channels every trajectory equals the clean run.
        nm = aleph.NoiseModel()
        noisy = aleph.run(self._bell(), shots=4096, noise=nm, seed=11).counts()
        self.assertEqual(set(noisy), {"00", "11"})
        self.assertEqual(sum(noisy.values()), 4096)

    def test_depolarizing_on_x_spreads_distribution(self):
        # Strong depolarizing on X injects errors; a circuit that is otherwise
        # |0> on qubit 0 (two X gates = net identity) picks up "1" outcomes.
        c = aleph.Circuit(1)
        c.x(0)
        c.x(0)  # net identity -> noiseless is all "0"
        nm = aleph.NoiseModel()
        nm.add_all_qubit_quantum_error(aleph.depolarizing_error(0.5, 1), ["x"])
        counts = aleph.run(c, shots=8000, noise=nm, seed=3).counts()
        self.assertIn("1", counts)
        self.assertGreater(counts.get("1", 0), 200)

    def test_readout_error_flips_outcomes(self):
        # A near-certain |00> state with heavy readout error produces "1"s.
        c = aleph.Circuit(2)  # noiseless -> all "00"
        nm = aleph.NoiseModel()
        nm.add_readout_error([[0.7, 0.3], [0.3, 0.7]], 0)
        counts = aleph.run(c, shots=8000, noise=nm, seed=5).counts()
        flipped = sum(v for k, v in counts.items() if k.endswith("1"))
        self.assertGreater(flipped, 1500)  # ~0.3 of 8000, generous slack

    def test_bad_params_raise(self):
        with self.assertRaises(ValueError):
            aleph.depolarizing_error(1.5, 1)
        with self.assertRaises(ValueError):
            aleph.depolarizing_error(0.1, 3)
        with self.assertRaises(ValueError):
            aleph.amplitude_damping_error(-0.1)
        with self.assertRaises(ValueError):
            aleph.pauli_error([])

    def test_unknown_gate_name_raises(self):
        # aleph has no idle "id" gate; attaching to it is an explicit error.
        nm = aleph.NoiseModel()
        with self.assertRaises(ValueError):
            nm.add_all_qubit_quantum_error(aleph.depolarizing_error(0.01, 1), ["id"])

    def test_aer_mnemonic_maps_to_internal_name(self):
        # "cx" must reach the engine as "Cnot": attach 2q depol to cx and run
        # a Bell circuit without error.
        nm = aleph.NoiseModel()
        nm.add_all_qubit_quantum_error(aleph.depolarizing_error(0.02, 2), ["cx"])
        counts = aleph.run(self._bell(), shots=2048, noise=nm, seed=9).counts()
        self.assertEqual(sum(counts.values()), 2048)

    def test_noise_rejects_non_sv_backend(self):
        nm = aleph.NoiseModel()
        with self.assertRaises(ValueError):
            aleph.run(self._bell(), shots=64, backend="mps", noise=nm, seed=1)

    def test_noisy_result_has_no_statevector(self):
        nm = aleph.NoiseModel()
        res = aleph.run(self._bell(), shots=64, noise=nm, seed=1)
        with self.assertRaises(ValueError):
            res.statevector()
