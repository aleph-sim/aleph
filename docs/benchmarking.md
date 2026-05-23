# Benchmarking

Aleph uses [criterion](https://bheisler.github.io/criterion.rs/book/) for benchmarks and uploads results to [bencher.dev](https://bencher.dev) on every PR + push to `main` for trend tracking.

## Running benchmarks

From the repo root:

```bash
# Run every bench in the workspace.
cargo bench --workspace

# Run a single fixture.
cargo bench --bench bell
cargo bench --bench ghz
cargo bench --bench qft
cargo bench --bench random

# Compile benches without running (CI gate; fast).
cargo bench --workspace --no-run
```

For honest single-machine numbers (everyone's local Mac is noisy):

```bash
RUSTFLAGS="-C target-cpu=native" cargo bench --bench ghz -- --quick
```

For comparing two branches locally:

```bash
git checkout main
cargo bench --bench qft -- --save-baseline main
git checkout <feature>
cargo bench --bench qft -- --baseline main
```

Criterion writes HTML reports to `target/criterion/<bench>/report/index.html`.

## Where benchmarks live

Workspace-level bench crate: [`benches/`](../benches/).

```
benches/
├── Cargo.toml          # package = aleph-benches
├── src/lib.rs          # shared fixture builders (zero_state, …)
└── benches/
    ├── bell.rs         # Bell pair (n=2)
    ├── ghz.rs          # GHZ state preparation (n=10, 15, 20, 25)
    ├── qft.rs          # QFT-style sweep (n=10, 15, 20)
    └── random.rs       # Random-circuit-style workload (n=20, depth=20)
```

Crate-local benches (when a backend wants kernel-level benches: e.g. `crates/aleph-sv/benches/*.rs`) are also discovered by `cargo bench --workspace`.

## Bench design notes

Until [P0-09](../BACKLOG.md) lands the naive `Backend` trait, the four bench fixtures don't execute real circuits — they exercise workloads with similar memory-traffic profiles (allocate `Vec<Complex>` of size `2^n`, sweep with per-amplitude ops). The fixture names, parameter shapes, and `BenchmarkId`s are chosen so the bencher.dev timeline stays continuous when the bench bodies get swapped for real circuit execution.

When you replace a stub:
- **Keep the criterion group name** (`ghz/prepare`, `qft/sweep`, etc.). Bencher's history is keyed by name.
- **Keep the parameter list** (`QUBIT_COUNTS`, `DEPTHS`). Adding new sizes is fine; removing one breaks the timeline.
- **Keep `Throughput::Elements`** as the unit, so plots stay in elements/second rather than ops/second.

## When to add a new bench

Add a bench when:
1. You're about to optimise something — write the bench **before** the change and capture baseline numbers.
2. A new algorithm class becomes interesting (e.g., VQE expectation values, QAOA single-layer apply). Use the existing fixture template; one file per algorithm.
3. A regression slipped past existing benches — write the minimal reproducer that flags it.

## CI integration

[`.github/workflows/bench.yml`](../.github/workflows/bench.yml) runs on PRs (when paths under `crates/**`, `benches/**`, `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml` change) and on every push to `main`.

The workflow:

1. `cargo bench --workspace --no-run` — gates that the bench harness still compiles. This step runs even when no benches exist.
2. `bencher run` — invokes `cargo bench --workspace` and uploads results to the bencher.dev project `aleph`, testbed `self-hosted-linux-x64`. Gated on `hashFiles('benches/**/*.rs', 'crates/*/benches/**/*.rs') != ''` so an empty workspace skips upload cleanly.

The bench runner is the **self-hosted Linux box** (AMD EPYC 8124P, 16C/32T, AVX-512, 123 GiB RAM). Dedicated hardware means bencher.dev's regression detection isn't fighting GitHub-hosted-runner variance.

### Thresholds

Currently **no thresholds** are configured. `bencher run` is invoked without `--err`, so a perf regression flagged by bencher.dev shows up in the PR comment but doesn't block merge. Once benches stabilise (post-P0-09) we'll set per-bench thresholds in the bencher.dev UI and add `--err` back to the workflow with explicit documentation.

### Activating bencher upload locally

The `BENCHER_API_TOKEN` secret + `bencher` CLI are wired in CI; running locally:

```bash
brew install bencher  # macOS
bencher run \
  --project aleph \
  --testbed local-mac \
  --adapter rust_criterion \
  --token "$BENCHER_API_TOKEN" \
  -- cargo bench --workspace
```

Don't upload local numbers to the project default testbed (`self-hosted-linux-x64`) — they'll skew the timeline. Use your own testbed label.

## See also

- [`OPTIMIZATION GUIDE.md`](../OPTIMIZATION%20GUIDE.md) — when/what/how to optimise.
- [`OPTIMIZATION CYCLE.md`](../OPTIMIZATION%20CYCLE.md) — step-by-step playbook for one optimisation iteration.
- Algorithm-specific playbooks at repo root (`QFT.md`, `GROVER.md`, `VQE.md`, `QAOA.md`, `RANDOM CIRCUIT.md`, `STABILIZER CIRCUITS.md`).
