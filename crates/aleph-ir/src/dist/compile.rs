//! P6-05 compiler (spec §6.4): build every candidate plan — routers
//! {Naive, Lookahead, Reorder{1..=g}} × {identity, `initial_placement`} — price
//! each with a [`CostModel`], return the cheapest. Under the model it is never
//! worse than Naive or Lookahead (exact up to the 1e-9 relative tie band),
//! because both are candidates.

use super::{
    initial_placement, plan_cost, plan_from, CostModel, DistError, DistLayout, DistPlan, DistStep,
    Router,
};
use crate::Circuit;

/// Costs within this relative distance tie; sums of floats are never exactly equal.
const REL_TIE: f64 = 1e-9;

/// One candidate: a router, started from the identity or from `initial_placement`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    /// The routing strategy that produced the plan.
    pub router: Router,
    /// `true`: started from [`initial_placement`]; `false`: identity.
    pub placed: bool,
}

/// The chosen plan plus the evidence for the choice.
#[derive(Debug, Clone)]
pub struct Compiled {
    /// The chosen (cheapest, tie-broken) plan.
    pub plan: DistPlan,
    /// Which candidate produced [`Self::plan`].
    pub choice: Candidate,
    /// `plan_cost(&plan, model)`.
    pub cost: f64,
    /// Every candidate's cost, in enumeration order (identity first; within a
    /// placement Naive, Lookahead, Reorder{1}, …, Reorder{max(g, 1)}).
    pub candidates: Vec<(Candidate, f64)>,
}

/// The cheapest plan under `cost`. See [`compile_detailed`].
pub fn compile(
    circuit: &Circuit,
    layout: DistLayout,
    cost: &dyn CostModel,
) -> Result<DistPlan, DistError> {
    compile_detailed(circuit, layout, cost).map(|c| c.plan)
}

/// Build and price every candidate; return the cheapest. Because Naive and
/// Lookahead are candidates, the result is never costlier than either, exact
/// up to the 1e-9 relative tie band.
///
/// Ties (relative difference ≤ 1e-9) go to fewer exchanges, then to fewer
/// `Local` instructions, then to the earlier candidate. The placed set is
/// skipped when `initial_placement` is the identity. Any candidate's planning
/// or pricing error is returned: a silent skip would hide a planner bug.
pub fn compile_detailed(
    circuit: &Circuit,
    layout: DistLayout,
    cost: &dyn CostModel,
) -> Result<Compiled, DistError> {
    let id: Vec<u32> = (0..layout.n).collect();
    let placed = initial_placement(circuit, layout)?;
    let mut inits: Vec<(bool, &[u32])> = vec![(false, &id[..])];
    if placed != id {
        inits.push((true, &placed[..]));
    }
    let mut routers = vec![Router::Naive, Router::Lookahead];
    routers.extend((1..=layout.g.max(1)).map(|max_k| Router::Reorder { max_k }));

    let mut best: Option<(Candidate, DistPlan, f64)> = None;
    let mut candidates = Vec::with_capacity(inits.len() * routers.len());
    for &(is_placed, init) in &inits {
        for &router in &routers {
            let p = plan_from(circuit, layout, router, init)?;
            let t = plan_cost(&p, cost)?;
            let cand = Candidate {
                router,
                placed: is_placed,
            };
            candidates.push((cand, t));
            let better = match &best {
                None => true,
                Some((_, bp, bt)) => beats(t, &p, *bt, bp),
            };
            if better {
                best = Some((cand, p, t));
            }
        }
    }
    let (choice, plan, cost) = best.ok_or(DistError::Unsupported {
        kind: "internal: compile built no candidates",
    })?;
    Ok(Compiled {
        plan,
        choice,
        cost,
        candidates,
    })
}

/// Whether a plan costing `t` replaces the current best `bt`. Both are finite
/// and ≥ 0 (`plan_cost` guarantees it).
fn beats(t: f64, p: &DistPlan, bt: f64, bp: &DistPlan) -> bool {
    if (t - bt).abs() > REL_TIE * t.max(bt) {
        return t < bt;
    }
    (p.stats.exchanges, local_instrs(p)) < (bp.stats.exchanges, local_instrs(bp))
}

fn local_instrs(p: &DistPlan) -> usize {
    p.steps
        .iter()
        .map(|s| match s {
            DistStep::Local(v) => v.len(),
            DistStep::Exchange { .. } => 0,
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::{plan_cost, plan_from, CostModel};
    use crate::Instruction;

    /// Bytes-like model: exchanges cost `(1 − 2^−k)`, locals cost `w` per instruction.
    struct Model {
        w: f64,
    }
    impl CostModel for Model {
        fn local_segment(&self, instrs: &[Instruction], _: DistLayout) -> Result<f64, DistError> {
            Ok(self.w * instrs.len() as f64)
        }
        fn exchange(&self, k: u32, _m: u32) -> f64 {
            1.0 - 0.5f64.powi(k as i32)
        }
    }

    fn ghz(n: u32) -> Circuit {
        let mut c = Circuit::new(n, 0);
        c.h(0).unwrap();
        for q in 0..n - 1 {
            c.cnot(q, q + 1).unwrap();
        }
        c
    }

    fn qft(n: u32) -> Circuit {
        let mut c = Circuit::new(n, 0);
        for j in (0..n).rev() {
            c.h(j).unwrap();
            for k in (0..j).rev() {
                c.add_gate(aleph_core::GateInstance::controlled(
                    aleph_core::Gate::Phase(aleph_core::Param::Concrete(
                        std::f64::consts::PI / f64::from(1u32 << (j - k)),
                    )),
                    vec![j],
                    vec![k],
                ))
                .unwrap();
            }
        }
        for q in 0..n / 2 {
            c.swap(q, n - 1 - q).unwrap();
        }
        c
    }

    #[test]
    fn picks_the_cheapest_candidate() {
        // QFT-12 g=2: placement is not the identity, so all 8 candidates exist.
        let c = qft(12);
        let l = DistLayout::new(12, 2).unwrap();
        let m = Model { w: 0.0 };
        let out = compile_detailed(&c, l, &m).unwrap();
        assert_eq!(out.candidates.len(), 8);
        let min = out
            .candidates
            .iter()
            .map(|&(_, t)| t)
            .fold(f64::INFINITY, f64::min);
        assert_eq!(out.cost, min);
        assert_eq!(plan_cost(&out.plan, &m).unwrap(), out.cost);
        // Re-planning the reported choice reproduces the returned plan.
        let init = if out.choice.placed {
            initial_placement(&c, l).unwrap()
        } else {
            (0..12).collect()
        };
        let again = plan_from(&c, l, out.choice.router, &init).unwrap();
        assert_eq!(again.final_map, out.plan.final_map);
        assert_eq!(again.stats, out.plan.stats);
        // compile() returns the same plan.
        let p = compile(&c, l, &m).unwrap();
        assert_eq!(p.stats, out.plan.stats);
    }

    #[test]
    fn identity_placement_is_not_duplicated() {
        // GHZ-12 g=2: qubits 10, 11 are needed last → placement is the identity.
        let c = ghz(12);
        let l = DistLayout::new(12, 2).unwrap();
        assert_eq!(
            initial_placement(&c, l).unwrap(),
            (0..12).collect::<Vec<u32>>()
        );
        let out = compile_detailed(&c, l, &Model { w: 0.0 }).unwrap();
        assert_eq!(out.candidates.len(), 4); // Naive, Lookahead, Reorder{1}, Reorder{2}
        assert!(out.candidates.iter().all(|(cand, _)| !cand.placed));
    }

    #[test]
    fn g0_compiles_without_exchanges() {
        let c = qft(5);
        let l = DistLayout::new(5, 0).unwrap();
        let out = compile_detailed(&c, l, &Model { w: 1.0 }).unwrap();
        assert_eq!(out.candidates.len(), 3);
        assert_eq!(out.plan.stats.exchanges, 0);
    }

    #[test]
    fn ties_prefer_fewer_exchanges_then_fewer_instrs() {
        // Zero model: every candidate costs 0, so the tie-break decides.
        struct Zero;
        impl CostModel for Zero {
            fn local_segment(&self, _: &[Instruction], _: DistLayout) -> Result<f64, DistError> {
                Ok(0.0)
            }
            fn exchange(&self, _: u32, _: u32) -> f64 {
                0.0
            }
        }
        let c = qft(12);
        let l = DistLayout::new(12, 2).unwrap();
        let out = compile_detailed(&c, l, &Zero).unwrap();
        let id: Vec<u32> = (0..12).collect();
        let placed = initial_placement(&c, l).unwrap();
        let mut best = (u32::MAX, usize::MAX);
        for &(cand, _) in &out.candidates {
            let init = if cand.placed { &placed } else { &id };
            let p = plan_from(&c, l, cand.router, init).unwrap();
            best = best.min((p.stats.exchanges, local_instrs(&p)));
        }
        assert_eq!((out.plan.stats.exchanges, local_instrs(&out.plan)), best);
        // Deterministic: same answer twice.
        let again = compile_detailed(&c, l, &Zero).unwrap();
        assert_eq!(again.choice, out.choice);
    }

    #[test]
    fn ties_never_depend_on_float_noise() {
        // Convex per-segment cost: fewer, longer segments (fewer exchanges)
        // are *raw-costlier* by ~1e-13 relative, so a bare `<` would pick the
        // wrong plan; only the REL_TIE band lets the tie-break decide.
        struct Noisy;
        impl CostModel for Noisy {
            fn local_segment(&self, i: &[Instruction], _: DistLayout) -> Result<f64, DistError> {
                let n = i.len() as f64;
                Ok(n + 1e-14 * n * n)
            }
            fn exchange(&self, _: u32, _: u32) -> f64 {
                0.0
            }
        }
        let c = qft(12);
        let l = DistLayout::new(12, 2).unwrap();
        let out = compile_detailed(&c, l, &Noisy).unwrap();
        let id: Vec<u32> = (0..12).collect();
        let placed = initial_placement(&c, l).unwrap();
        let key = |cand: Candidate| {
            let p = plan_from(&c, l, cand.router, if cand.placed { &placed } else { &id }).unwrap();
            (p.stats.exchanges, local_instrs(&p))
        };
        // Candidates inside the band of the raw minimum.
        let (raw_cand, raw_min) = out
            .candidates
            .iter()
            .copied()
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        let in_band: Vec<Candidate> = out
            .candidates
            .iter()
            .filter(|&&(_, t)| t - raw_min <= REL_TIE * raw_min)
            .map(|&(cand, _)| cand)
            .collect();
        assert!(in_band.len() > 1);
        assert!(in_band.contains(&out.choice));
        // Fewer exchanges, then fewer local instructions, wins inside the band.
        let best = in_band.iter().map(|&cand| key(cand)).min().unwrap();
        assert_eq!(key(out.choice), best);
        // Non-vacuous: the raw-cheapest candidate is NOT the tie-break winner.
        assert!(key(raw_cand) > best, "noise must oppose the tie-break");
        assert!(out.cost > raw_min);
    }

    #[test]
    fn full_tie_goes_to_the_earliest_candidate() {
        // g=0: no exchanges, all three routers emit the same plan, every cost 0.
        struct Zero;
        impl CostModel for Zero {
            fn local_segment(&self, _: &[Instruction], _: DistLayout) -> Result<f64, DistError> {
                Ok(0.0)
            }
            fn exchange(&self, _: u32, _: u32) -> f64 {
                0.0
            }
        }
        let c = qft(5);
        let l = DistLayout::new(5, 0).unwrap();
        let out = compile_detailed(&c, l, &Zero).unwrap();
        assert_eq!(out.candidates.len(), 3);
        let id: Vec<u32> = (0..5).collect();
        let keys: Vec<_> = out
            .candidates
            .iter()
            .map(|&(cand, _)| {
                let p = plan_from(&c, l, cand.router, &id).unwrap();
                (p.stats.exchanges, local_instrs(&p))
            })
            .collect();
        assert!(keys.iter().all(|k| *k == keys[0]));
        assert_eq!(
            out.choice,
            Candidate {
                router: Router::Naive,
                placed: false
            }
        );
    }

    #[test]
    fn errors_propagate() {
        // Qubit-count mismatch.
        let l = DistLayout::new(4, 1).unwrap();
        assert!(matches!(
            compile(&ghz(5), l, &Model { w: 1.0 }),
            Err(DistError::QubitCountMismatch { .. })
        ));
        // Unsupported instruction.
        let mut c = Circuit::new(4, 1);
        c.h(0).unwrap();
        c.measure(0, 0).unwrap();
        assert!(matches!(
            compile(&c, l, &Model { w: 1.0 }),
            Err(DistError::Unsupported { .. })
        ));
        let mut r = Circuit::new(4, 0);
        r.h(0).unwrap();
        r.reset(0).unwrap();
        assert!(matches!(
            compile(&r, l, &Model { w: 1.0 }),
            Err(DistError::Unsupported { .. })
        ));
        // A failing model.
        struct Fails;
        impl CostModel for Fails {
            fn local_segment(&self, _: &[Instruction], _: DistLayout) -> Result<f64, DistError> {
                Err(DistError::Unsupported { kind: "test" })
            }
            fn exchange(&self, _: u32, _: u32) -> f64 {
                0.0
            }
        }
        assert_eq!(
            compile(&ghz(4), l, &Fails).unwrap_err(),
            DistError::Unsupported { kind: "test" }
        );
    }

    #[test]
    fn never_costlier_than_naive_or_lookahead() {
        for (c, g) in [(qft(12), 2u32), (ghz(12), 3), (qft(10), 1)] {
            let l = DistLayout::new(c.num_qubits(), g).unwrap();
            for w in [0.0, 0.01, 1.0] {
                let m = Model { w };
                let out = compile_detailed(&c, l, &m).unwrap();
                for r in [Router::Naive, Router::Lookahead] {
                    let base = plan_cost(&crate::dist::plan(&c, l, r).unwrap(), &m).unwrap();
                    assert!(
                        out.cost <= base * (1.0 + 1e-9),
                        "g={g} w={w} {r:?}: {} > {base}",
                        out.cost
                    );
                }
            }
        }
    }
}
