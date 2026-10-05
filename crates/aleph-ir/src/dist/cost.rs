//! P6-05 cost model interface. `aleph-ir` only declares what a cost is; a
//! backend crate's model says what one costs.
//! Objective (spec §6.1): `T = Σ step_costs(plan)`; by default `Σ_Local local_segment + Σ_Exchange exchange(k, m)`.

use super::{DistError, DistLayout, DistPlan, DistStep};
use crate::Instruction;

/// Seconds a distributed plan's pieces take on one rank's device.
pub trait CostModel {
    /// One rank's time for one `Local` step (physical indices, pre-specialise).
    fn local_segment(&self, instrs: &[Instruction], layout: DistLayout) -> Result<f64, DistError>;
    /// One `k`-bit exchange of a `2^m` slice.
    fn exchange(&self, k: u32, m: u32) -> f64;
    /// Per-step costs of `plan`, one per step, in order.
    ///
    /// Default: each step priced alone (`local_segment` / `exchange`). A model
    /// may override this to carry state from one step to the next.
    fn step_costs(&self, plan: &DistPlan) -> Result<Vec<f64>, DistError> {
        let m = plan.layout.m();
        plan.steps
            .iter()
            .map(|step| match step {
                DistStep::Local(instrs) => self.local_segment(instrs, plan.layout),
                DistStep::Exchange { global_bits } => {
                    Ok(self.exchange(global_bits.len() as u32, m))
                }
            })
            .collect()
    }
}

/// Predicted per-device time of `plan` under `model`: the sum of
/// `model.step_costs(plan)`.
///
/// Rejects a step-cost list whose length differs from the plan's step count,
/// and a step cost or total that is not finite or is negative (a model bug or
/// a missing table entry), so callers never rank plans on NaN/∞.
pub fn plan_cost(plan: &DistPlan, model: &dyn CostModel) -> Result<f64, DistError> {
    const BAD: DistError = DistError::Unsupported {
        kind: "internal: non-finite or negative cost",
    };
    let costs = model.step_costs(plan)?;
    if costs.len() != plan.steps.len() {
        return Err(DistError::Unsupported {
            kind: "internal: step_costs length mismatch",
        });
    }
    let mut t = 0.0;
    for c in costs {
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

    /// Charges local steps 1 s per instruction, ×2 once any earlier step was
    /// an exchange: a model that carries state across steps.
    struct Stateful;
    impl CostModel for Stateful {
        fn local_segment(&self, instrs: &[Instruction], _: DistLayout) -> Result<f64, DistError> {
            Ok(instrs.len() as f64)
        }
        fn exchange(&self, _: u32, _: u32) -> f64 {
            0.0
        }
        fn step_costs(&self, plan: &DistPlan) -> Result<Vec<f64>, DistError> {
            let mut seen_exchange = false;
            let mut out = Vec::new();
            for s in &plan.steps {
                out.push(match s {
                    DistStep::Local(v) => {
                        let f = if seen_exchange { 2.0 } else { 1.0 };
                        f * self.local_segment(v, plan.layout)?
                    }
                    DistStep::Exchange { .. } => {
                        seen_exchange = true;
                        0.0
                    }
                });
            }
            Ok(out)
        }
    }

    #[test]
    fn plan_cost_uses_step_costs_override() {
        // n=4, g=1: h(0) local, h(3) needs an exchange, then h(3) and h(0) local.
        let mut c = Circuit::new(4, 0);
        c.h(0).unwrap();
        c.h(3).unwrap();
        c.h(0).unwrap();
        let p = plan(&c, DistLayout::new(4, 1).unwrap(), Router::Naive).unwrap();
        let mut want = 0.0;
        let mut seen = false;
        for s in &p.steps {
            match s {
                DistStep::Local(v) => want += if seen { 2.0 } else { 1.0 } * v.len() as f64,
                DistStep::Exchange { .. } => seen = true,
            }
        }
        assert!(seen, "the plan must contain an exchange");
        assert_eq!(plan_cost(&p, &Stateful).unwrap(), want);
        // The default hook prices each step alone: Stub's sum is unchanged.
        assert_eq!(
            Stub.step_costs(&p).unwrap().iter().sum::<f64>(),
            plan_cost(&p, &Stub).unwrap()
        );
    }

    #[test]
    fn plan_cost_rejects_step_count_mismatch() {
        struct Short;
        impl CostModel for Short {
            fn local_segment(&self, _: &[Instruction], _: DistLayout) -> Result<f64, DistError> {
                Ok(1.0)
            }
            fn exchange(&self, _: u32, _: u32) -> f64 {
                1.0
            }
            fn step_costs(&self, _: &DistPlan) -> Result<Vec<f64>, DistError> {
                Ok(vec![1.0])
            }
        }
        let mut c = Circuit::new(4, 0);
        c.h(0).unwrap();
        c.h(3).unwrap();
        let p = plan(&c, DistLayout::new(4, 1).unwrap(), Router::Naive).unwrap();
        assert!(p.steps.len() > 1);
        assert_eq!(
            plan_cost(&p, &Short),
            Err(DistError::Unsupported {
                kind: "internal: step_costs length mismatch"
            })
        );
    }
}
