//! Same saved input through full SV, compact SV and sparse Pauli queries, plus MPS.
use aleph_backend::{run, Backend};
use aleph_core::PauliSum;
use aleph_mps::MpsBackend;
use aleph_stab::{SparsePauliError, SparsePauliOptions, SparsePauliQuery};
use aleph_sv::{
    expectation_query::{ExactExpectationQuery, ExactQueryError},
    NaiveSvBackend,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{error::Error, path::Path};
type AnyResult<T> = Result<T, Box<dyn Error>>;
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn close(a: f64, b: f64) -> AnyResult<()> {
    if !a.is_finite() || !b.is_finite() || (a - b).abs() > 1e-10 {
        Err(format!("query disagreement {a} != {b}").into())
    } else {
        Ok(())
    }
}
fn fixture(name: &str, n: u32) -> (String, String, Option<f64>) {
    let mut qasm = format!("OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[{n}] q;\n");
    let (factors, constant, truth) = if name.starts_with("ghz") {
        qasm.push_str("h q[0];\n");
        for q in 1..n {
            qasm.push_str(&format!("cx q[0], q[{q}];\n"));
        }
        qasm.push_str("rz(0.37) q[0];\n");
        ("X".repeat(n as usize), 0.375, Some(0.37f64.cos() + 0.375))
    } else if name == "local128" {
        for layer in 0..3 {
            for q in 0..n {
                qasm.push_str(&format!("ry({}) q[{q}];\n", 0.17 + (q as f64) * 0.002));
            }
            for q in ((layer % 2)..n - 1).step_by(2) {
                qasm.push_str(&format!("cx q[{q}], q[{}];\n", q + 1));
            }
        }
        let mut chars = vec!['I'; n as usize];
        chars[64] = 'Z';
        (chars.into_iter().collect(), 0.25, None)
    } else {
        for q in 0..n {
            qasm.push_str(&format!("ry(0.37) q[{q}];\n"));
        }
        (
            "Z".repeat(n as usize),
            0.,
            Some(0.37f64.cos().powi(n as i32)),
        )
    };
    // GHZ constants are deliberately split into an identity and a nonlocal Z0 Z1 term.
    let mut observable = format!("1.0 {factors}\n");
    if name.starts_with("ghz") {
        observable.push_str(&format!(
            "0.25 {}\n0.125 ZZ{}\n",
            "I".repeat(n as usize),
            "I".repeat(n as usize - 2)
        ));
    } else if constant != 0. {
        observable.push_str(&format!("{constant} {}\n", "I".repeat(n as usize)));
    }
    (qasm, observable, truth)
}
pub fn study(out: &Path, smoke: bool) -> AnyResult<Value> {
    std::fs::create_dir(out)?;
    let mut cases = Vec::new();
    for (name, n) in [
        ("ghz8", 8),
        ("local128", 128),
        ("ghz128", 128),
        (
            if smoke { "product6" } else { "product16" },
            if smoke { 6 } else { 16 },
        ),
    ] {
        let (qasm, observable, analytic) = fixture(name, n);
        std::fs::write(out.join(format!("{name}.qasm")), &qasm)?;
        std::fs::write(out.join(format!("{name}.pauli")), &observable)?;
        // Parse the actual persisted files, not a parallel in-memory circuit construction.
        let c = aleph_parser::parse(&std::fs::read_to_string(out.join(format!("{name}.qasm")))?)?;
        let p = PauliSum::parse(
            &std::fs::read_to_string(out.join(format!("{name}.pauli")))?,
            n,
        )?;
        let dense = ExactExpectationQuery::compile(&c, &p, 64 << 20);
        let reference = if let Some(v) = analytic {
            v
        } else {
            dense
                .as_ref()
                .map_err(|_| "local reference unexpectedly exceeded dense budget")?
                .execute()?
                .value
        };
        let compact = match dense {
            Ok(q) => {
                let r = q.execute()?;
                close(r.value, reference)?;
                json!({"status":"complete","value":r.value,"peak_workspace_bytes":r.peak_dense_workspace_bytes,"active_qubits":q.plan().cones().iter().map(|x|x.circuit().num_qubits()).collect::<Vec<_>>(),"source_indices":q.plan().cones().iter().map(|x|x.retained_source_indices()).collect::<Vec<_>>(),"compact_to_original":q.plan().cones().iter().map(|x|x.compact_to_original()).collect::<Vec<_>>()})
            }
            Err(ExactQueryError::Memory {
                active_qubits,
                required_bytes,
                budget_bytes,
            }) => {
                json!({"status":"resource_limit","active_qubits":active_qubits,"required_bytes":required_bytes,"budget_bytes":budget_bytes})
            }
            Err(e) => return Err(e.into()),
        };
        let mut sparse = Vec::new();
        for cones in [false, true] {
            let r = SparsePauliQuery::compile(
                &c,
                &p,
                SparsePauliOptions {
                    use_cones: cones,
                    ..Default::default()
                },
            )?
            .execute()?;
            close(r.value, reference)?;
            sparse.push(json!({"use_cones":cones,"value":r.value,"max_active_qubits":r.stats.max_active_qubits,"retained_gates":r.stats.retained_gates,"peak_terms":r.stats.peak_terms,"peak_candidates":r.stats.peak_candidates,"peak_workspace_bytes":r.stats.peak_workspace_bytes,"branching_rotations":r.stats.branching_rotations,"splits":r.stats.splits,"merges":r.stats.merges}));
        }
        let full = if n <= 20 {
            let mut b = NaiveSvBackend::with_seed(0);
            let state = run(&mut b, &c)?;
            let mut value = 0.;
            for t in &p.terms {
                value += b.expectation_value(&state, t)?;
            }
            close(value, reference)?;
            json!({"status":"complete","value":value,"state_payload_bytes":16usize*(1usize<<n)})
        } else {
            json!({"status":"not_attempted","reason":"full SV restricted to <=20 qubits before allocation"})
        };
        let mut b = MpsBackend::with_seed(0).with_max_bond(128);
        let state = run(&mut b, &c)?;
        let mut value = 0.;
        for t in &p.terms {
            value += b.expectation_value(&state, t)?;
        }
        close(value, reference)?;
        if !state.truncation_error().is_finite() || state.truncation_error() > 1e-12 {
            return Err("MPS overlap truncated beyond accepted numerical tolerance".into());
        }
        let limit = if name.starts_with("product") {
            match SparsePauliQuery::compile(
                &c,
                &p,
                SparsePauliOptions {
                    max_terms: 8,
                    use_cones: false,
                    ..Default::default()
                },
            )?
            .execute()
            {
                Err(SparsePauliError::Terms { required, limit }) => {
                    json!({"status":"resource_limit","required":required,"limit":limit})
                }
                Err(e) => return Err(e.into()),
                Ok(_) => return Err("term-limited query unexpectedly completed".into()),
            }
        } else {
            Value::Null
        };
        let mut query = Sha256::new();
        query.update(b"p7-query-v1\0");
        query.update((qasm.len() as u64).to_le_bytes());
        query.update(qasm.as_bytes());
        query.update(observable.as_bytes());
        cases.push(json!({"case":name,"qubits":n,"qasm_file":format!("{name}.qasm"),"observable_file":format!("{name}.pauli"),"qasm_sha256":hash(qasm.as_bytes()),"observable_sha256":hash(observable.as_bytes()),"query_sha256":format!("{:x}",query.finalize()),"reference":reference,"reference_kind":if analytic.is_some(){"independent analytic"}else{"P7-03 compact FP64 SV"},"full_sv":full,"compact_sv":compact,"sparse":sparse,"mps":{"value":value,"max_bond":state.max_bond_reached(),"discarded_weight":state.truncation_error()},"term_limit_control":limit}));
    }
    let report = json!({"study_version":1,"smoke":smoke,"initial_state":"zero","precision":"f64","sampling_seed":null,"tolerance":1e-10,"query_hash_format":"SHA256(b'p7-query-v1\\0' || u64_le(qasm_utf8_byte_length) || qasm_utf8 || pauli_utf8)","cases":cases,"scope":"same saved inputs, explicit resource refusal, no automatic routing; this integration run does not establish a clean-host timing speedup"});
    std::fs::write(
        out.join("study.json"),
        serde_json::to_string_pretty(&report)?,
    )?;
    Ok(report)
}
