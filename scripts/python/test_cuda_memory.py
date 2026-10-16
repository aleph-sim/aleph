"""Real CUDA memory-policy frontend tests. Explicit opt-in must execute hardware.
ALEPH_TEST_CUDA_MEMORY=1 python -m unittest discover -s scripts/python -p test_cuda_memory.py -v
"""
import os
import unittest
import numpy as np
import aleph
from aleph import _native


@unittest.skipUnless(os.environ.get('ALEPH_TEST_CUDA_MEMORY') == '1', 'requires explicit CUDA hardware opt-in')
class TestCudaMemory(unittest.TestCase):
    def test_reject_before_sv_initialization(self):
        for backend in ('cuda', 'cuda-f32'):
            for qubits in (40, 64):
                before = _native._gpu_backend_init_count()
                with self.assertRaisesRegex(ValueError, 'memory|overflow'):
                    aleph.run(aleph.Circuit(qubits), backend=backend, shots=1024)
                self.assertEqual(_native._gpu_backend_init_count(), before)

    def test_counts_zero_shots_and_lazy_export(self):
        for backend in ('cuda', 'cuda-f32'):
            c = aleph.Circuit(8)
            c.h(0)
            c.cx(0, 7)
            for shots in (0, 32):
                result = aleph.run(c, backend=backend, shots=shots, seed=554)
                self.assertEqual(sum(result.counts().values()), shots)
                amps = result.statevector()
                expected = np.zeros(256, dtype=complex)
                expected[[0, 129]] = 1 / np.sqrt(2)
                np.testing.assert_allclose(amps, expected, atol=1e-6)
                again = result.statevector()
                np.testing.assert_array_equal(amps, again)
