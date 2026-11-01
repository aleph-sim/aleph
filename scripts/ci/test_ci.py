"""Routing, failure and argv coverage contracts; no GPU or shell interpreter needed."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import docs
import gate
import plan
import run

SHA = 'a' * 40


def identified(p):
    return dict(p, head_sha=SHA, base_sha=SHA, tested_sha=SHA)


def sample(paths, force=False):
    roots = {'crates/' + x: x for x in ('aleph-core', 'aleph-ir', 'aleph-sv', 'aleph-qec', 'aleph-metal', 'aleph-cuda', 'aleph-cli', 'aleph-py')}
    graph = {v: set() for v in roots.values()}
    graph['aleph-core'] = {'aleph-ir', 'aleph-sv', 'aleph-metal', 'aleph-cuda'}
    graph['aleph-ir'] = {'aleph-qec'}
    graph['aleph-metal'] = {'aleph-cli', 'aleph-py'}
    graph['aleph-cuda'] = {'aleph-cli', 'aleph-py'}
    graph['aleph-sv'] = {'aleph-cli', 'aleph-py'}
    graph['aleph-qec'] = {'aleph-cuda', 'aleph-py'}
    return identified(plan.classify(paths, roots, graph, force))


class Routing(unittest.TestCase):
    def test_docs_and_receipts_do_not_select_code(self):
        p = sample(['README.md', 'docs/perf/report.md', 'docs/perf/data/case/raw.jsonl', 'docs/perf/data/case/SHA256SUMS'])
        self.assertEqual(p['mode'], 'docs')
        self.assertEqual([k for k,v in p['jobs'].items() if v], ['docs'])
        plan.validate(p)

    def test_mixed_and_unknown_paths_fall_back(self):
        for name in ('Cargo.lock', 'crates/aleph-sv/Cargo.toml', 'crates/aleph-cuda/build.rs', 'crates/aleph-py/pyproject.toml', 'crates/aleph-metal/.cargo/config.toml', '.github/workflows/ci.yml', 'scripts/ci/plan.py', 'new-input.bin', 'docs/fixtures/model.json', 'docs/perf/source.py'):
            self.assertEqual(sample(['README.md', name])['mode'], 'full', name)
        self.assertNotEqual(sample(['README.md', 'crates/aleph-sv/src/lib.rs'])['mode'], 'docs')
        self.assertEqual(sample([])['mode'], 'full')
        self.assertEqual(sample(['README.md'], force=True)['mode'], 'full')

    def test_backend_consumers_and_platforms(self):
        metal = sample(['crates/aleph-metal/src/shaders/test.metal'])
        self.assertEqual(metal['packages'], ['aleph-cli', 'aleph-metal', 'aleph-py'])
        self.assertFalse(metal['jobs']['linux'])
        self.assertFalse(metal['jobs']['clippy'])
        self.assertFalse(metal['cuda'])
        self.assertEqual(metal['python_os'], 'macos-latest')
        self.assertEqual(metal['python_features'], 'python,metal')
        cuda = sample(['crates/aleph-cuda/src/kernel.cu'])
        self.assertTrue(cuda['jobs']['linux'])
        self.assertFalse(cuda['metal'])
        self.assertEqual(cuda['python_features'], 'python,cuda')
        self.assertFalse(cuda['jobs']['hardware'])

    def test_cpu_shared_qec_frontend_and_scripts(self):
        cpu = sample(['crates/aleph-sv/src/kernel.rs'])
        self.assertIn('aleph-py', cpu['packages'])
        self.assertTrue(cpu['metal'] and cpu['cuda'])
        shared = sample(['crates/aleph-core/src/lib.rs'])
        self.assertTrue(shared['jobs']['hardware'])
        qec = sample(['crates/aleph-qec/src/model.rs'])
        self.assertTrue(qec['jobs']['hardware'] and qec['cuda'])
        for path in ('crates/aleph-py/src/run.rs', 'crates/aleph-cli/src/exec.rs', 'scripts/python/test_gpu.py', 'scripts/benchmarks/cuda_worker.py'):
            p = sample([path])
            self.assertTrue(p['cuda'] and p['metal'], path)
        hw = sample(['hw/core.sv'])
        self.assertEqual([k for k,v in hw['jobs'].items() if v], ['docs', 'hardware'])

    def test_dependency_cycles_and_optional_dev_build_target_edges(self):
        self.assertEqual(plan.closure({'a'}, {'a': {'b'}, 'b': {'a', 'c'}, 'c': set()}), ['a','b','c'])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            names = ['aleph-seed', 'aleph-dev', 'aleph-build', 'aleph-optional', 'aleph-target']
            packages = []
            for n in names:
                dep = [] if n == names[0] else [{'name': names[0], 'path': str(root/names[0]), 'kind': 'dev' if n.endswith('dev') else 'build', 'optional': True, 'target': 'cfg(target_os = "macos")'}]
                packages.append({'name':n, 'id':n, 'manifest_path':str(root/n/'Cargo.toml'), 'dependencies':dep})
            metadata = json.dumps({'workspace_members':names,'packages':packages}).encode()
            with patch.object(subprocess, 'check_output', return_value=metadata) as command:
                _, reverse = plan.workspace(root)
                self.assertIn('--locked', command.call_args[0][0])
            self.assertEqual(plan.closure({names[0]}, reverse), sorted(names))

    def test_complete_git_diff_deletions_renames_spaces_and_more_than_300(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            def g(*argv):return subprocess.check_output(['git','-C',d,*argv],stderr=subprocess.PIPE).decode().strip()
            g('init','-q');g('config','user.email','test@example.invalid');g('config','user.name','CI Test')
            (root/'old.md').write_text('old')
            g('add','.');g('commit','-qm','base');base=g('rev-parse','HEAD')
            (root/'old.md').rename(root/'code.rs')
            for n in range(305):(root/f'doc {n}.md').write_text('x')
            (root/'zz-last.rs').write_text('code')
            g('add','.');g('commit','-qm','head');head=g('rev-parse','HEAD')
            event={'pull_request':{'base':{'sha':base},'head':{'sha':head}}}
            paths,h,b,t,f=plan.changed_paths(root,event,'pull_request')
            self.assertEqual((h,b,t,f),(head,base,head,False))
            self.assertEqual(len(paths),308)
            for p in ('old.md','code.rs','doc 304.md','zz-last.rs'):self.assertIn(p,paths)
            push=plan.changed_paths(root,{'before':base,'after':head},'push')
            self.assertEqual(push[0],paths)
            with self.assertRaises(ValueError):plan.changed_paths(root,{'before':'0'*40,'after':head},'push')
            (root/'old.md').symlink_to('/etc/passwd')
            g('add','.');g('commit','-qm','symlink');link_head=g('rev-parse','HEAD')
            self.assertTrue(plan.changed_paths(root,{'before':head,'after':link_head},'push')[-1])
            (root/'old.md').unlink();g('add','.');g('commit','-qm','delete symlink');deleted_head=g('rev-parse','HEAD')
            self.assertTrue(plan.changed_paths(root,{'before':link_head,'after':deleted_head},'push')[-1])

    def test_gate_rejects_all_non_success_and_bad_outputs(self):
        p = sample(['README.md'])
        needs={'plan':{'result':'success'},**{k:{'result':'success' if v else 'skipped'} for k,v in p['jobs'].items()}}
        self.assertTrue(gate.check(p,needs,SHA,SHA))
        for state in ('failure','cancelled','skipped','neutral',None):
            n=copy.deepcopy(needs);n['docs']['result']=state
            with self.assertRaises(ValueError):gate.check(p,n,SHA,SHA)
        for k in ('plan','docs','linux'):
            n=copy.deepcopy(needs);del n[k]
            with self.assertRaises(ValueError):gate.check(p,n,SHA,SHA)
        with self.assertRaises(ValueError):gate.check(p,needs,'b'*40,SHA)
        with self.assertRaises(ValueError):gate.check(p,needs,SHA,'b'*40)
        for value in ('false',None,0):
            q=copy.deepcopy(p);q['jobs']['linux']=value
            with self.assertRaises(ValueError):plan.validate(q)
        full=sample(['Cargo.lock']);full_needs={'plan':{'result':'success'},**{k:{'result':'success'} for k in full['jobs']}}
        for group in ('linux','macos','python','hardware'):
            n=copy.deepcopy(full_needs);n[group]['result']='failure'
            with self.assertRaises(ValueError):gate.check(full,n,SHA,SHA)

    def test_selected_argv_cover_real_feature_consumers(self):
        p=sample(['crates/aleph-cuda/src/lib.rs'])
        lines=[' '.join(c) for c in run.commands(p,'clippy')]
        self.assertTrue(any('-p aleph-cli --features cuda' in c for c in lines))
        self.assertTrue(any('-p aleph-py --features python,cuda' in c for c in lines))
        self.assertTrue(any('--features nccl' in c for c in lines))
        self.assertFalse(any('--workspace' in c for c in lines))
        p=sample(['crates/aleph-metal/src/lib.rs'])
        lines=[' '.join(c) for c in run.commands(p,'macos','beta')]
        self.assertTrue(all(c.startswith('cargo +beta ') for c in lines))
        self.assertTrue(any('-p aleph-cli --features metal' in c for c in lines))
        self.assertTrue(any('-p aleph-py --features python,metal' in c for c in lines))
        self.assertTrue(any('test --locked -p aleph-metal --features metal' in c for c in lines))
        self.assertIn('python,metal',run.commands(p,'python')[0])
        with self.assertRaises(ValueError):run.commands(sample(['README.md']),'linux')


class Documents(unittest.TestCase):
    def test_changed_links_data_and_deleted_assets(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'docs').mkdir();p=root/'docs/report.md'
            p.write_text('[bad](missing.md)')
            with patch.object(docs,'git',return_value=b''):
                with self.assertRaises(ValueError):docs.check(root,{'paths':['docs/report.md'],'base_sha':SHA})
                (root/'docs/missing.md').write_text('exists')
                docs.check(root,{'paths':['docs/report.md'],'base_sha':SHA})
                (root/'docs/missing.md').unlink()
                with self.assertRaises(ValueError):docs.check(root,{'paths':['docs/missing.md'],'base_sha':SHA})
            (root/'docs/report.json').write_text('{broken')
            with self.assertRaises(ValueError):docs.check(root,{'paths':['docs/report.json'],'base_sha':SHA})

    def test_stale_report_checksum_and_symlinks(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'docs/perf/data/case';folder.mkdir(parents=True)
            (folder/'raw.txt').write_text('data')
            (folder/'SHA256SUMS').write_text('0'*64+'  raw.txt\n')
            with self.assertRaises(ValueError):docs.check(root,{'paths':['docs/perf/data/case/raw.txt'],'base_sha':SHA})
            (root/'docs/escape.md').symlink_to('/etc/passwd')
            with self.assertRaises(ValueError):docs.check(root,{'paths':['docs/escape.md'],'base_sha':SHA})


class PlannerIntegration(unittest.TestCase):
    def test_real_docs_entrypoint_uses_no_cargo_and_binds_event_identity(self):
        planner=Path(plan.__file__).resolve()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def g(*args):return subprocess.check_output(['git','-C',d,*args],stderr=subprocess.PIPE).decode().strip()
            g('init','-q');g('config','user.email','test@example.invalid');g('config','user.name','CI Test');g('config','commit.gpgsign','false')
            (root/'README.md').write_text('base')
            g('add','.');g('commit','-qm','base');base=g('rev-parse','HEAD')
            (root/'README.md').write_text('updated prose')
            g('add','.');g('commit','-qm','docs');head=g('rev-parse','HEAD')
            binpath=root/'bin';binpath.mkdir()
            fake=binpath/'cargo';fake.write_text('#!/bin/sh\necho cargo-was-called >&2\nexit 99\n');fake.chmod(0o755)
            event=root/'event.json';output=root/'outputs';summary=root/'summary'
            payload={'pull_request':{'head':{'sha':head,'repo':{'full_name':'fork/repo'}},'base':{'sha':base},'labels':[]}}
            event.write_text(json.dumps(payload))
            env=dict(os.environ,GITHUB_EVENT_NAME='pull_request',GITHUB_EVENT_PATH=str(event),GITHUB_OUTPUT=str(output),GITHUB_STEP_SUMMARY=str(summary),PATH=str(binpath)+os.pathsep+os.environ['PATH'])
            p=subprocess.run([os.sys.executable,str(planner)],cwd=root,env=env,capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            result=json.loads((root/'ci-plan.json').read_text())
            self.assertEqual(result['mode'],'docs')
            self.assertEqual((result['head_sha'],result['base_sha'],result['tested_sha']),(head,base,head))
            self.assertIn('linux=false',output.read_text())
            self.assertIn('not a test pass',summary.read_text())
            env['GITHUB_EVENT_NAME']='push';event.write_text(json.dumps({'before':base,'after':head}))
            p=subprocess.run([os.sys.executable,str(planner)],cwd=root,env=env,capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(json.loads((root/'ci-plan.json').read_text())['mode'],'docs')
            # A failed diff must never become docs success; fallback needs Cargo,
            # which our deliberately unavailable metadata provider makes fatal.
            env['GITHUB_EVENT_NAME']='pull_request';payload['pull_request']['base']['sha']='b'*40;event.write_text(json.dumps(payload))
            p=subprocess.run([os.sys.executable,str(planner)],cwd=root,env=env,capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('cargo-was-called',p.stderr)

    def test_real_docs_gate_never_executes_fenced_shell(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'README.md'
            p.write_text('```sh\ntouch SENTINEL\n[example](missing.md)\n```\n')
            with patch.object(docs,'git',return_value=b''):
                docs.check(root,{'paths':['README.md'],'base_sha':SHA})
            self.assertFalse((root/'SENTINEL').exists())

    def test_reference_baseline_link_does_not_hide_new_broken_link(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'README.md';p.write_text('[old](old.md)\n[new](new.md)')
            with patch.object(docs,'git',return_value=b'[old](old.md)'):
                with self.assertRaisesRegex(ValueError,'new.md'):docs.check(root,{'paths':['README.md'],'base_sha':SHA})
            p.write_text('[old](old.md)')
            with patch.object(docs,'git',return_value=b'[old](old.md)'):
                docs.check(root,{'paths':['README.md'],'base_sha':SHA})

    def test_docs_reject_bad_jsonl_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'docs').mkdir()
            for name,text in [('bad.json','{"value":NaN}'),('bad.jsonl','{}\nnot-json\n')]:
                (root/'docs'/name).write_text(text)
                with self.assertRaises(ValueError):docs.check(root,{'paths':['docs/'+name],'base_sha':SHA})


class WorkflowPolicy(unittest.TestCase):
    def test_plan_cannot_supply_a_runner_label(self):
        workflow=(Path(__file__).resolve().parents[2]/'.github/workflows/ci.yml').read_text()
        lines=[s.strip() for s in workflow.splitlines() if s.strip().startswith('runs-on:')]
        for line in lines:
            if 'needs.plan' in line:
                self.assertEqual(line, "runs-on: ${{ needs.plan.outputs.python_os == 'macos-latest' && 'macos-latest' || 'ubuntu-latest' }}")
            if 'self-hosted' in line:
                self.assertIn('github.event.pull_request.head.repo.full_name == github.repository',line)
                self.assertIn('["ubuntu-latest"]',line)
        self.assertNotIn('pull_request_target:',workflow)
        self.assertNotIn('continue-on-error:',workflow)
        self.assertIn('if: always()',workflow)
        hardware=(Path(__file__).resolve().parents[2]/'.github/workflows/hw.yml').read_text()
        self.assertIn('group: rtl-${{ github.ref }}',hardware)
        self.assertIn('workflow_call:',hardware)


if __name__ == '__main__':
    unittest.main()
