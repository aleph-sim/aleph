//! CPU exact expectation demo and reproducible public-API comparison campaign.
//! No arguments: independently checked 128-qubit GHZ phase expectation.
//! `--one CASE MODE`: one cold query, useful for process RSS observation.
//! `--audit`: ABBA (ABCD/DCBA across four APIs) repeated timings and correctness checks.
use aleph_backend::{run, Backend};
use aleph_core::PauliSum;
use aleph_ir::Circuit;
use aleph_mps::MpsBackend;
use aleph_stab::{SparsePauliOptions, SparsePauliQuery, SparsePauliResult};
use aleph_sv::{expectation_query::ExactExpectationQuery, NaiveSvBackend};
use serde_json::{json, Value};
use std::{error::Error, time::Instant};
#[path = "support/sparse_workloads.rs"]
mod workloads;
fn record(r: SparsePauliResult) -> Value {
    json!({"value":r.value,"planning_ns":r.planning_time.as_nanos() as u64,"execution_ns":r.execution_time.as_nanos() as u64,"peak_terms":r.stats.peak_terms,"peak_candidates":r.stats.peak_candidates,"peak_workspace_bytes":r.stats.peak_workspace_bytes,"branching_rotations":r.stats.branching_rotations,"splits":r.stats.splits,"merges":r.stats.merges,"coefficient_l1":r.stats.final_coefficient_l1,"coefficient_l2":r.stats.final_coefficient_l2,"max_active_qubits":r.stats.max_active_qubits,"retained_gates":r.stats.retained_gates})
}
fn execute(c: &Circuit, p: &PauliSum, mode: &str) -> Result<Value, Box<dyn Error>> {
    let start = Instant::now();
    let mut result = match mode {
        "sparse" | "sparse-cone" => {
            let r = SparsePauliQuery::compile(
                c,
                p,
                SparsePauliOptions {
                    use_cones: mode == "sparse-cone",
                    ..Default::default()
                },
            )?
            .execute()?;
            record(r)
        }
        "cone-sv" => {
            let r = ExactExpectationQuery::compile(c, p, 64 << 20)?.execute()?;
            json!({"value":r.value,"peak_dense_workspace_bytes":r.peak_dense_workspace_bytes})
        }
        "sv" => {
            if c.num_qubits() > 20 {
                return Err("full SV capped at 20 qubits in this campaign".into());
            }
            let mut b = NaiveSvBackend::with_seed(0);
            let s = run(&mut b, c)?;
            let mut v = 0.;
            for t in &p.terms {
                v += b.expectation_value(&s, t)?;
            }
            json!({"value":v,"state_payload_bytes":16usize*(1usize<<c.num_qubits())})
        }
        "mps" => {
            let mut b = MpsBackend::with_seed(0).with_max_bond(128);
            let s = run(&mut b, c)?;
            let mut v = 0.;
            for t in &p.terms {
                v += b.expectation_value(&s, t)?;
            }
            if s.truncation_error() > 1e-12 {
                return Err("MPS truncated; not accepted as exact overlap".into());
            }
            json!({"value":v,"max_bond_reached":s.max_bond_reached(),"truncation_weight":s.truncation_error()})
        }
        _ => return Err("mode: sparse/sparse-cone/cone-sv/sv/mps".into()),
    };
    result["total_ns"] = json!(start.elapsed().as_nanos() as u64);
    result["mode"] = json!(mode);
    Ok(result)
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        let (c, p, analytic) = workloads::fixture("ghz128")?;
        let r = execute(&c, &p, "sparse-cone")?;
        let value = r["value"].as_f64().ok_or("missing result")?;
        if (value - analytic.ok_or("missing analytic reference")?).abs() > 1e-10 {
            return Err("analytic mismatch".into());
        }
        println!(
            "{}",
            serde_json::to_string_pretty(
                &json!({"case":"ghz128","initial_state":"zero","precision":"f64 untruncated","analytic":analytic,"result":r})
            )?
        );
    } else if (args.len() == 4 || (args.len() == 5 && args[4] == "--no-cones"))
        && !args[0].starts_with("--")
    {
        let c = aleph_parser::parse(&std::fs::read_to_string(&args[0])?)?;
        let p = PauliSum::parse(&std::fs::read_to_string(&args[1])?, c.num_qubits())?;
        let options = SparsePauliOptions {
            max_terms: args[2].parse()?,
            max_workspace_bytes: args[3].parse()?,
            use_cones: args.len() == 4,
        };
        let r = SparsePauliQuery::compile(&c, &p, options)?.execute()?;
        println!(
            "{}",
            serde_json::to_string_pretty(
                &json!({"original_qubits":c.num_qubits(),"original_instructions":c.len(),"precision":"f64 untruncated","initial_state":"zero","use_cones":options.use_cones,"max_terms":options.max_terms,"max_workspace_bytes":options.max_workspace_bytes,"result":record(r)})
            )?
        );
    } else if args.len() == 3 && args[0] == "--one" {
        let (c, p, a) = workloads::fixture(&args[1])?;
        let r = execute(&c, &p, &args[2])?;
        println!("{}", json!({"case":args[1],"analytic":a,"result":r}));
    } else if args.as_slice() == ["--audit"] {
        for name in workloads::names() {
            let (c, p, analytic) = workloads::fixture(&name)?;
            let reference = if let Some(a) = analytic {
                a
            } else {
                ExactExpectationQuery::compile(&c, &p, 64 << 20)?
                    .execute()?
                    .value
            };
            let mut modes = vec!["sparse", "sparse-cone", "mps"];
            if c.num_qubits() <= 20 {
                modes.push("sv");
            } else if name == "local128" {
                modes.push("cone-sv");
            }
            for mode in &modes {
                let r = execute(&c, &p, mode)?;
                let value = r["value"].as_f64().ok_or("value")?;
                if !value.is_finite() || (value - reference).abs() > 1e-10 {
                    return Err(
                        format!("{name}/{mode}: oracle mismatch {value} != {reference}").into(),
                    );
                }
            }
            // Predeclared five paired forward/reverse blocks; keep all timings, no fastest-run selection.
            for block in 0..5 {
                for reverse in [false, true] {
                    let order: Vec<_> = if reverse {
                        modes.iter().rev().collect()
                    } else {
                        modes.iter().collect()
                    };
                    for mode in order {
                        let r = execute(&c, &p, mode)?;
                        println!(
                            "{}",
                            json!({"case":name,"block":block,"reverse":reverse,"reference":reference,"original_qubits":c.num_qubits(),"original_gates":c.len(),"result":r})
                        );
                    }
                }
            }
        }
    } else {
        return Err("usage: sparse_expectation [CIRCUIT.qasm PAULI_SUM.txt MAX_TERMS WORKSPACE_BYTES [--no-cones] | --one CASE MODE | --audit]".into());
    }
    Ok(())
}
