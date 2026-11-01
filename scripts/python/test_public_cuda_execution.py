"""Real public CUDA execution checks; ALEPH_TEST_PUBLIC_CUDA=1 must execute hardware.
Optional ALEPH_CUDA_CLI adds CLI frontend checks. No large states or OOM probes.
"""
import gc
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import aleph
from aleph import _native


def circuit(n=8):
    c=aleph.Circuit(n)
    for q in range(n): c.h(q)
    c.barrier(list(range(n)))
    for layer in range(3):
        for q in range(n): c.ry(.13+q*.07+layer*.03,q)
        for q in range(n-1): c.cx(q,q+1)
        for q in range(n): c.rz(.11+q*.02,q)
    return c


@unittest.skipUnless(os.environ.get('ALEPH_TEST_PUBLIC_CUDA')=='1','explicit CUDA hardware opt-in required')
class PublicCudaExecution(unittest.TestCase):
    def test_raw_optimized_precision_launches_and_lazy_transfer(self):
        c=circuit()
        expected=aleph.run(c,backend='sv',shots=0).statevector()
        for backend,tol in [('cuda',1e-10),('cuda-f32',1e-5)]:
            for optimize in [False,True]:
                before=_native._gpu_result_stats()
                r=aleph.run(c,backend=backend,shots=64,seed=552,cuda_optimize=optimize)
                after=_native._gpu_result_stats()
                self.assertEqual(after[0],before[0])
                self.assertEqual(after[3]-before[3],16+64*8)
                d=r.execution_stats()
                self.assertEqual(d['optimized'],float(optimize))
                self.assertEqual(d['tf32_effective'],0)
                self.assertEqual(d['tf32_launches_process_delta'],0)
                self.assertGreater(d['kernel_launches_process_delta'],0)
                if optimize:
                    self.assertGreater(d['fused_kq'],0)
                    self.assertGreater(d['layer_launches_process_delta'],0)
                    self.assertLess(d['output_instructions'],d['input_instructions'])
                np.testing.assert_allclose(r.statevector(),expected,atol=tol,rtol=0)
                again=aleph.run(c,backend=backend,shots=64,seed=552,cuda_optimize=optimize)
                self.assertEqual(r.counts(),again.counts())
                del r,again
                gc.collect()

    def test_fused_plan_with_large_state_sparse_shot_sampler(self):
        # n=18, shots=16 selects #555's hierarchical path, rather than the
        # small-state BlockCdf path used by the other frontend fixtures.
        c=aleph.Circuit(18)
        c.h(0)
        for q in range(17): c.cx(q,q+1)
        expected=aleph.run(c,backend='sv',shots=0).statevector()
        support={'0'*18,'1'*18}
        for backend,tol in [('cuda',1e-10),('cuda-f32',1e-5)]:
            for optimize in [False,True]:
                r=aleph.run(c,backend=backend,shots=16,seed=552,cuda_optimize=optimize)
                self.assertEqual(sum(r.counts().values()),16)
                self.assertTrue(set(r.counts()) <= support)
                np.testing.assert_allclose(r.statevector(),expected,atol=tol,rtol=0)
                again=aleph.run(c,backend=backend,shots=16,seed=552,cuda_optimize=optimize)
                self.assertEqual(r.counts(),again.counts())
                del r,again
                gc.collect()

    def test_diagonal_phase_and_nontrivial_preparation(self):
        c=aleph.Circuit(8)
        for q in range(8):c.ry(.3+.09*q,q)
        for i in range(8):
            c.h(i)
            for j in range(i+1,8):c.cp(.4/(2**(j-i)),j,i)
        expected=aleph.run(c,backend='sv',shots=0).statevector()
        for backend,tol in [('cuda',1e-10),('cuda-f32',1e-5)]:
            result=aleph.run(c,backend=backend,shots=0)
            self.assertGreater(result.execution_stats()['diagonal_polynomials'],0)
            np.testing.assert_allclose(result.statevector(),expected,atol=tol,rtol=0)

    def test_collapse_final_state_and_strict_fp32(self):
        c=aleph.Circuit.from_qasm('OPENQASM 3.0; include "stdgates.inc"; qubit[3] q; bit[1] c; h q[1]; measure q[0] -> c[0]; x q[2];')
        expected=np.zeros(8,dtype=complex);expected[[4,6]]=1/np.sqrt(2)
        for backend in ['cuda','cuda-f32']:
            r=aleph.run(c,backend=backend,shots=64,seed=5)
            self.assertTrue(set(r.counts()) <= {'100','110'})
            np.testing.assert_allclose(r.statevector(),expected,atol=1e-6)
        c=circuit()
        reference=aleph.run(c,backend='sv',shots=0).statevector()
        result=aleph.run(c,backend='cuda-f32',shots=0)
        d=result.execution_stats()
        self.assertEqual(d['fusion_width'],3)
        self.assertEqual(d['tf32_effective'],0)
        self.assertEqual(d['tf32_launches_process_delta'],0)
        np.testing.assert_allclose(result.statevector(),reference,atol=1e-5,rtol=0)

    def test_missing_tf32_module_keeps_strict_width_three(self):
        # Hide only the optional module's CUDA header path in a child process;
        # the ordinary NVRTC kernels and driver remain available.
        with tempfile.TemporaryDirectory() as tmp:
            env=dict(os.environ,CUDA_HOME=tmp,CUDA_PATH=tmp)
            code="import aleph,json; c=aleph.Circuit(8); c.h(0); r=aleph.run(c,backend='cuda-f32',shots=16,seed=552); print(json.dumps(r.execution_stats())); assert sum(r.counts().values())==16"
            p=subprocess.run([sys.executable,'-c',code],env=env,text=True,capture_output=True,check=True)
            d=json.loads(p.stdout)
            self.assertEqual(d['tf32_available'],0)
            self.assertEqual(d['tf32_effective'],0)
            self.assertEqual(d['fusion_width'],3)
            self.assertEqual(d['tf32_launches_process_delta'],0)

    def test_unsupported_operations_reject_before_backend_construction(self):
        c=aleph.Circuit.from_qasm('OPENQASM 3.0; include "stdgates.inc"; qubit[2] q; reset q[0];')
        for backend in ['cuda','cuda-f32']:
            before=_native._gpu_backend_init_count()
            with self.assertRaisesRegex(ValueError,'reset'):
                aleph.run(c,backend=backend)
            self.assertEqual(_native._gpu_backend_init_count(),before)
            with self.assertRaisesRegex(ValueError,'32 qubits'):
                aleph.run(aleph.Circuit(33),backend=backend)
            self.assertEqual(_native._gpu_backend_init_count(),before)
        with self.assertRaisesRegex(ValueError,'require a CUDA'):
            aleph.run(aleph.Circuit(2),backend='sv',cuda_optimize=False)

    def test_cli_uses_same_plan_and_final_state(self):
        binary=os.environ.get('ALEPH_CUDA_CLI')
        if not binary: self.skipTest('set ALEPH_CUDA_CLI for CLI route')
        source='OPENQASM 3.0; include "stdgates.inc"; qubit[3] q; bit[1] c; h q[1]; measure q[0] -> c[0]; x q[2];'
        with tempfile.TemporaryDirectory() as tmp:
            path=pathlib.Path(tmp)/'collapse.qasm';path.write_text(source)
            for backend in ['cuda','cuda-f32']:
                for flags in [[],['--cuda-raw']]:
                    p=subprocess.run([binary,'run',str(path),'--backend',backend,'--shots','16','--statevector','--seed','7','--cuda-diagnostics',*flags],text=True,capture_output=True,check=True)
                    d=json.loads(next(l.split('=',1)[1] for l in p.stderr.splitlines() if l.startswith('CUDA_EXECUTION=')))
                    self.assertEqual(d['optimized'],float(not flags))
                    self.assertEqual(d['tf32_launches_process_delta'],0)
                    self.assertIn('|100⟩',p.stdout)
                    self.assertIn('|110⟩',p.stdout)
                    self.assertIn('sampling_ms',d)


if __name__=='__main__':unittest.main()
