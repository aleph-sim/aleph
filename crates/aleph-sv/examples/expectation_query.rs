//! Query tool: [circuit.qasm pauli-sum.txt dense-workspace-budget-bytes].
//! No arguments runs a 128-qubit shallow local expectation demo.
use aleph_backend::{run, Backend};
use aleph_sv::expectation_query::ExactExpectationQuery;
use aleph_sv::NaiveSvBackend;
use serde_json::json;
use std::time::Instant;
#[path = "support/cone_workloads.rs"]
mod workloads;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.as_slice() == ["--audit"] {
        return audit();
    }
    let (circuit, observable, budget) = if args.is_empty() {
        (
            workloads::brickwork(128, 3, 0),
            workloads::observable(64),
            1usize << 20,
        )
    } else if args.len() == 3 {
        let source = std::fs::read_to_string(&args[0])?;
        let circuit = aleph_parser::parse(&source)?;
        let observable =
            aleph_core::PauliSum::parse(&std::fs::read_to_string(&args[1])?, circuit.num_qubits())?;
        (circuit, observable, args[2].parse::<usize>()?)
    } else {
        return Err(
            "usage: expectation_query [circuit.qasm pauli-sum.txt dense-workspace-budget-bytes]"
                .into(),
        );
    };
    let query = ExactExpectationQuery::compile(&circuit, &observable, budget)?;
    let result = query.execute()?;
    let maps:Vec<_>=query.plan().cones().iter().zip(&result.cones).map(|(cone,stats)|json!({
        "active_qubits":stats.active_qubits,"retained_operations":stats.retained_operations,
        "source_indices":cone.retained_source_indices(),"compact_to_original":cone.compact_to_original(),
        "original_to_compact":cone.original_to_compact(),"term_indices":cone.observables().iter().map(|(i,_)|*i).collect::<Vec<_>>(),
        "dense_workspace_bytes":stats.dense_workspace_bytes,"execution_ms":stats.execution_time.as_secs_f64()*1000.
    })).collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "original_qubits":circuit.num_qubits(),"original_instructions":circuit.len(),
            "full_state_bytes_formula":format!("16 * 2^{}",circuit.num_qubits()),
            "value":result.value,"term_values":result.term_values,"planning_ms":result.planning_time.as_secs_f64()*1000.,
            "execution_ms":result.execution_time.as_secs_f64()*1000.,"peak_dense_workspace_bytes":result.peak_dense_workspace_bytes,
            "budget_bytes":budget,"precision":"f64","initial_state":"zero","cones":maps
        }))?
    );
    Ok(())
}

// Paired public Rust API calls supplement Criterion's separately calibrated tiers.
fn audit() -> Result<(), Box<dyn std::error::Error>> {
    for (name, circuit, observable) in [
        (
            "local18",
            workloads::brickwork(18, 3, 0),
            workloads::observable(9),
        ),
        ("global16", workloads::global(16), workloads::observable(15)),
    ] {
        let mut reference = NaiveSvBackend::with_seed(0);
        let state = run(&mut reference, &circuit)?;
        let expected = reference.expectation_value(&state, &observable.terms[0])?;
        drop(state);
        let warm = ExactExpectationQuery::compile(&circuit, &observable, 64 << 20)?.execute()?;
        assert!((warm.value - expected).abs() < 1e-10);
        for pair in 0..5 {
            for arm in if pair % 2 == 0 {
                ["full", "query"]
            } else {
                ["query", "full"]
            } {
                let start = Instant::now();
                let (value, planning_ms, execution_ms, width, operations, bytes) = if arm == "full"
                {
                    let mut backend = NaiveSvBackend::with_seed(0);
                    let state = run(&mut backend, &circuit)?;
                    let value = backend.expectation_value(&state, &observable.terms[0])?;
                    drop(state);
                    (
                        value,
                        0.,
                        start.elapsed().as_secs_f64() * 1000.,
                        circuit.num_qubits(),
                        circuit.len(),
                        40 * (1usize << circuit.num_qubits()) + 4096,
                    )
                } else {
                    let query = ExactExpectationQuery::compile(&circuit, &observable, 64 << 20)?;
                    let result = query.execute()?;
                    (
                        result.value,
                        result.planning_time.as_secs_f64() * 1000.,
                        result.execution_time.as_secs_f64() * 1000.,
                        result.cones[0].active_qubits,
                        result.cones[0].retained_operations,
                        result.peak_dense_workspace_bytes,
                    )
                };
                let caller_ms = start.elapsed().as_secs_f64() * 1000.;
                assert!(value.is_finite() && (value - expected).abs() < 1e-10);
                println!(
                    "{}",
                    json!({"case":name,"pair":pair,"arm":arm,"precision":"f64","initial_state":"zero",
                    "original_qubits":circuit.num_qubits(),"active_qubits":width,"operations":operations,
                    "dense_workspace_bound":bytes,"caller_ms":caller_ms,"planning_ms":planning_ms,"execution_ms":execution_ms,"value":value})
                );
            }
        }
    }
    Ok(())
}
