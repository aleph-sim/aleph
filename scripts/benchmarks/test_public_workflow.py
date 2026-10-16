"""Contract/failure tests plus an opt-in ordinary CLI route integration guard."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import public_workflow as w


def spec():
    return dict(schema_version=1, name='cpu-control', route='cpu-cli', comparison='same-algorithm',
        model='ghz', precision='f64', error_budget=dict(amplitude_atol=1e-10,norm_atol=1e-4), topology='one process, one host',
        stop_rule='stop on failures, contamination or inert requested counter; two pairs for CI',
        qubits=3, shots=16, pairs=2, threads=1, workers=1, seed=563, warmup_calls=0,
        max_load_per_cpu=1, max_gpu_utilization=5, bottleneck_fraction=.5, target_speedup=1,
        preflight=dict(kind='control', reason='identical reference: expected no gain'),
        arms=[dict(name=a, executable=sys.executable, revision='0'*40, optimizer=False, algorithm='statevector', build_command='test fixture', rustflags='unset', executable_sha256=w.sha256(sys.executable)) for a in ('a','b')])


def receipt(s):
    return dict(protocol=1, status='executed', route=s['route'], hardware_executed=s['route']=='cuda-python',
                warmup_calls=s['warmup_calls'], shots_observed=s['shots'], qubits=s['qubits'],
                counters={'path':1}, stages=dict.fromkeys(w.STAGES))


class Contracts(unittest.TestCase):
    def test_schema_and_amdahl_gate(self):
        for key, val in [('schema_version',2),('pairs',0),('threads',True),('workers',2),
                         ('target_speedup',3),('precision','tf32'),('warmup_calls',1)]:
            with self.subTest(key=key):
                s=spec(); s[key]=val
                with self.assertRaises(ValueError): w.validate_spec(s)
        s=spec(); s['arms'][1]['algorithm']='other'
        with self.assertRaises(ValueError): w.validate_spec(s)
        s['comparison']='algorithm-change'
        w.validate_spec(s)

    def test_receipt_missing_hardware_or_counters_is_failure(self):
        s=spec(); s['route']='cuda-python'; r=receipt(s)
        for key, value in [('hardware_executed',False),('status','skipped'),('counters',None),
                           ('shots_observed',0),('warmup_calls',1),('qubits',7)]:
            with self.subTest(key=key):
                bad=dict(r);bad[key]=value
                with self.assertRaises(ValueError):w.validate_receipt(bad,s)
        r['stages']['execution_ms']=float('nan')
        with self.assertRaises(ValueError): w.validate_receipt(r,s)

    def test_deterministic_workload_and_summary(self):
        s=spec();s['model']='brickwall'
        self.assertEqual(w.qasm(s),w.qasm(s))
        s['seed']+=1
        self.assertNotEqual(w.qasm(s),w.qasm(spec()))
        rows=[dict(phase='measurement',pair=i,arm=a,status='executed',caller_ms=t)
              for i in range(3) for a,t in [('a',20),('b',10)]]
        rows += [dict(phase='measurement',pair=5,arm='a',status='failed',caller_ms=.01)]
        summary=w.summarize(rows,563)
        self.assertEqual(summary['pairs'],3)
        self.assertEqual(summary['paired_bootstrap_95'],[2,2])

    def test_paired_order_and_inert_parameter_stops_before_measurement(self):
        s=spec();calls=[];preflights=[]
        def invoke(s, arm, *args, **kwargs):
            calls.append(arm['name'])
            preflights.append(kwargs['preflight'])
            return dict(status='executed',caller_ms=1,receipt=receipt(s))
        with tempfile.TemporaryDirectory() as d, patch.object(w,'invoke',side_effect=invoke):
            result=w.run_campaign(s,Path(d)/'paired')
            self.assertEqual(result['status'],'executed')
            self.assertEqual(w.validate_result(Path(d)/'paired')['status'],'executed')
            self.assertEqual(calls,['b','a','a','b','b','a'])
            self.assertEqual(preflights,[True,True,False,False,False,False])
            calls.clear();s['preflight']=dict(kind='counter-change',counter='path')
            result=w.run_campaign(s,Path(d)/'inert')
            self.assertEqual(result['status'],'failed')
            self.assertIn('inert',result['error'])
            self.assertEqual(len(calls),2)

    def test_skipped_hardware_never_passes(self):
        s=spec();s['route']='cuda-python'
        for arm in s['arms']:arm['executable']='/missing/sdk/python'
        with tempfile.TemporaryDirectory() as d, patch.object(w,'invoke') as invoke:
            result=w.run_campaign(s,Path(d)/'skip',skip_hardware=True)
            self.assertEqual(result['status'],'skipped')
            self.assertNotIn('summary',result)
            self.assertEqual(w.validate_result(Path(d)/'skip')['status'],'skipped')
            invoke.assert_not_called()

    def test_subprocess_failure_and_bad_receipt(self):
        s=spec()
        # Python executable is deliberately not the CLI; unknown run command
        # must remain a subprocess failure, not a skipped hardware success.
        with tempfile.TemporaryDirectory() as d, patch.object(w,'resource_ok',return_value=True):
            p=Path(d)/'x.qasm';p.write_text(w.qasm(s))
            r=w.invoke(s,s['arms'][0],Path(d),p)
            self.assertEqual(r['status'],'failed')
            self.assertNotEqual(r['returncode'],0)

    def test_changed_executable_rejected_before_launch(self):
        s=spec();s['arms'][0]['executable_sha256']='0'*64
        with tempfile.TemporaryDirectory() as d, patch.object(w.subprocess,'run') as run:
            r=w.invoke(s,s['arms'][0],Path(d),Path(d)/'x')
            self.assertEqual(r['status'],'failed')
            self.assertIn('changed',r['reason'])
            run.assert_not_called()

    def test_successful_subprocess_with_wrong_receipt_fails(self):
        s=spec()
        class Process:
            returncode=0
            stdout='counts (16 shots):\n  |000⟩  16  (1.0)'
            stderr='CPU_EXECUTION={"qubits":3}'
        with tempfile.TemporaryDirectory() as d, patch.object(w,'resource_ok',return_value=True), \
                patch.object(w,'resources',return_value={}), patch.object(w.subprocess,'run',return_value=Process()):
            r=w.invoke(s,s['arms'][0],Path(d),Path(d)/'x')
            self.assertEqual(r['status'],'failed')
            self.assertIn('invalid receipt',r['reason'])

    def test_contamination_aborts_without_launching(self):
        with tempfile.TemporaryDirectory() as d, patch.object(w,'resource_ok',return_value=False), \
                patch.object(w.subprocess,'run') as run:
            # resources itself calls subprocess, so replace it as well.
            with patch.object(w,'resources',return_value={}):
                r=w.invoke(spec(),spec()['arms'][0],Path(d),Path(d)/'x')
            self.assertEqual(r['status'],'contaminated')
            run.assert_not_called()

    @unittest.skipUnless(os.environ.get('ALEPH_BENCH_TEST_CLI'), 'set ALEPH_BENCH_TEST_CLI to execute public CLI guard')
    def test_real_cpu_cli_entrypoint(self):
        s=spec()
        for arm in s['arms']:arm['executable']=os.environ['ALEPH_BENCH_TEST_CLI']
        with tempfile.TemporaryDirectory() as d, patch.object(w,'resource_ok',return_value=True):
            # CI timing is not performance evidence; execute frontend contracts
            # without requiring an idle CI host or any wall-time threshold.
            result=w.run_campaign(s,Path(d)/'actual-cli')
            self.assertEqual(result['status'],'executed',result)
            rows=[json.loads(x) for x in (Path(d)/'actual-cli/runs.jsonl').read_text().splitlines()]
            for row in rows:
                r=row['receipt'];self.assertFalse(r['hardware_executed'])
                self.assertEqual(r['shots_observed'],16)
                self.assertGreaterEqual(r['stages']['execution_ms'],0)
                self.assertIn('--backend',row['command'])
                self.assertEqual(row['command'][1],'run')


if __name__=='__main__':unittest.main()
