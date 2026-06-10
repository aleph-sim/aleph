"""Behaviour tests for the aleph Python bindings (P4-08).

Not in CI (no maturin step there). Release gate — run from the repo root
after installing the wheel:

    python -m unittest discover -s scripts/python -v
"""
import math
import os
import unittest

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
        self.assertEqual(len(amps), 2)
        for a in amps:
            self.assertAlmostEqual(a.real, 1 / math.sqrt(2), places=10)
            self.assertAlmostEqual(a.imag, 0.0, places=10)

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
