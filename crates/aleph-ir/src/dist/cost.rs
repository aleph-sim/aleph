//! P6-05 cost model interface. `aleph-ir` only declares what a cost is; a
//! backend crate's model says what one costs.
//! Objective (spec §6.1): `T = Σ_Local local_segment + Σ_Exchange exchange(k, m)`.

use super::{DistError, DistLayout, DistPlan, DistStep};
use crate::Instruction;

/// Seconds a distributed plan's pieces take on one rank's device.
pub trait CostModel {
    /// One rank's time for one `Local` step (physical indices, pre-specialise).
    fn local_segment(&self, instrs: &[Instruction], layout: DistLayout) -> Result<f64, DistError>;
    /// One `k`-bit exchange of a `2^m` slice.
    fn exchange(&self, k: u32, m: u32) -> f64;
}

/// Predicted per-device time of `plan` under `model`.
///
/// Rejects a step cost or total that is not finite or is negative (a model
/// bug or a missing table entry), so callers never rank plans on NaN/∞.
pub fn plan_cost(plan: &DistPlan, model: &dyn CostModel) -> Result<f64, DistError> {
    const BAD: DistError = DistError::Unsupported {
        kind: "internal: non-finite or negative cost",
    };
    let m = plan.layout.m();
    let mut t = 0.0;
    for step in &plan.steps {
        let c = match step {
            DistStep::Local(instrs) => model.local_segment(instrs, plan.layout)?,
            DistStep::Exchange { global_bits } => model.exchange(global_bits.len() as u32, m),
        };
        // is_finite first: NaN compares false (ADR 0006).
        if !c.is_finite() || c < 0.0 {
            return Err(BAD);
        }
        t += c;
    }
    if !t.is_finite() {
        return Err(BAD);
    }
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::{plan, Router};
    use crate::Circuit;

    /// 1 s per instruction, 10 s per exchanged bit.
    struct Stub;
    impl CostModel for Stub {
        fn local_segment(&self, instrs: &[Instruction], _: DistLayout) -> Result<f64, DistError> {
            Ok(instrs.len() as f64)
        }
        fn exchange(&self, k: u32, _m: u32) -> f64 {
            10.0 * f64::from(k)
        }
    }

    #[test]
    fn plan_cost_sums_locals_and_exchanges() {
        // n=4, g=1: h(3) needs one 1-bit exchange; h(0), h(3) are local work.
        let mut c = Circuit::new(4, 0);
        c.h(0).unwrap();
        c.h(3).unwrap();
        let p = plan(&c, DistLayout::new(4, 1).unwrap(), Router::Naive).unwrap();
        assert_eq!(p.stats.exchanges, 1);
        let locals: usize = p
            .steps
            .iter()
            .map(|s| match s {
                DistStep::Local(v) => v.len(),
                DistStep::Exchange { .. } => 0,
            })
            .sum();
        assert_eq!(plan_cost(&p, &Stub).unwrap(), locals as f64 + 10.0);
    }

    #[test]
    fn plan_cost_propagates_errors() {
        struct Fails;
        impl CostModel for Fails {
            fn local_segment(&self, _: &[Instruction], _: DistLayout) -> Result<f64, DistError> {
                Err(DistError::Unsupported { kind: "test" })
            }
            fn exchange(&self, _: u32, _: u32) -> f64 {
                0.0
            }
        }
        let mut c = Circuit::new(2, 0);
        c.h(0).unwrap();
        let p = plan(&c, DistLayout::new(2, 0).unwrap(), Router::Naive).unwrap();
        assert!(plan_cost(&p, &Fails).is_err());
    }

    #[test]
    fn plan_cost_rejects_non_finite_and_negative() {
        struct Bad(f64);
        impl CostModel for Bad {
            fn local_segment(&self, _: &[Instruction], _: DistLayout) -> Result<f64, DistError> {
                Ok(self.0)
            }
            fn exchange(&self, _: u32, _: u32) -> f64 {
                0.0
            }
        }
        let mut c = Circuit::new(2, 0);
        c.h(0).unwrap();
        let p = plan(&c, DistLayout::new(2, 0).unwrap(), Router::Naive).unwrap();
        let want = Err(DistError::Unsupported {
            kind: "internal: non-finite or negative cost",
        });
        assert_eq!(plan_cost(&p, &Bad(f64::NAN)), want);
        assert_eq!(plan_cost(&p, &Bad(f64::INFINITY)), want);
        assert_eq!(plan_cost(&p, &Bad(-1.0)), want);
        assert_eq!(plan_cost(&p, &Bad(2.0)), Ok(2.0));
    }
}
