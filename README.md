# Algorithm Optimization Playbooks

Per-algorithm guides applying the framework from `OPTIMIZATION_GUIDE.md` to specific quantum algorithms.

## Read order

1. `../OPTIMIZATION_GUIDE.md` — methodology, principles, checklists.
1. `../OPTIMIZATION_CYCLE.md` — step-by-step iteration playbook.
1. **This directory** — algorithm-specific opportunities.

## Playbooks

|Algorithm                      |File                    |Key win                          |When to consult                          |
|-------------------------------|------------------------|---------------------------------|-----------------------------------------|
|Quantum Fourier Transform      |<QFT.md>                |Phase polynomial fusion, AQFT    |Working on diagonal gates, QFT/QPE/Shor  |
|Grover’s algorithm             |<GROVER.md>             |Specialized MCZ, diffusion fusion|Working on multi-controlled gates, search|
|Variational Quantum Eigensolver|<VQE.md>                |Symbolic params, Pauli grouping  |NISQ chemistry; **highest practical ROI**|
|QAOA                           |<QAOA.md>               |Diagonal cost-layer fusion, MPS  |Combinatorial optimization, sparse graphs|
|Random Circuits                |<RANDOM_CIRCUIT.md>     |Generic kernel quality           |Stress testing, supremacy benchmarks     |
|Stabilizer Circuits            |<STABILIZER_CIRCUITS.md>|Bit-packed tableau, batched shots|QEC, surface codes, Clifford-only        |

## Structure of each playbook

Every playbook follows the same template:

1. **Quick Reference** — algorithm at a glance.
1. **Algorithm Overview** — brief technical recap.
1. **Computational Profile** — where the time goes.
1. **Optimization Ladder** — opportunities in ROI order.
1. **Pitfalls** — algorithm-specific gotchas.
1. **Baseline Comparisons** — what to beat (Qiskit Aer / Stim / cuQuantum).
1. **Phase-by-Phase Sub-goals** — what’s expected at each project phase.
1. **Success Metrics** — when an optimization PR is considered successful.
1. **References** — primary literature.

## When to add a new playbook

Add a playbook when:

- The project starts targeting an algorithm not covered (e.g., Shor, Hamiltonian simulation).
- An algorithm reveals optimization opportunities not captured by the global guide.
- Multiple PRs on the same algorithm would benefit from shared context.

Follow the template structure. Open a PR titled `[playbook] Add {AlgorithmName} playbook`.

## When to update an existing playbook

Update a playbook when:

- A new optimization opportunity is discovered.
- A pitfall is encountered in review or in production.
- Baseline numbers change (new external version, new reference hardware).
- A sub-goal is achieved or refined.

Open a PR titled `[playbook] Update {AlgorithmName}: {reason}`.