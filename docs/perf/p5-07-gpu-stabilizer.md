# P5-07 — GPU stabilizer backend (CHP tableau)

A device-resident Aaronson–Gottesman stabilizer tableau with word-parallel
Clifford kernels. The goal of the issue: a GPU stabilizer that runs `n ≥ 1000`,
`depth ≥ 1000`, and is **faster than Stim on large inputs**. Both are met — the
GPU beats Stim's `TableauSimulator` by ~3–12× and the CPU `aleph-stab` backend by
~2.5–12× across `n = 1000…65000`.

## Representation

The CPU backend already has the GPU-friendly layout: in its **ColMajor**
orientation a qubit column is a contiguous `Wr = ceil((2n+1)/64)` word-span, and
a Clifford gate updates that span word-parallel (`h_words`/`s_words`/`cnot_words`,
AG 2004 §2). The GPU mirrors it exactly:

```
x[a*Wr + w], z[a*Wr + w]  = word w of qubit a's column (over the 2n+1 row axis)
sign[w]                   = sign bits, packed over the same row axis
```

The bit math is identical to the CPU's, so the GPU tableau is **bit-for-bit
equal** to `aleph-stab`'s after the same gate sequence (the oracle below). Per-gate
parallelism is `Wr`, which grows with `n` — the regime the issue targets.

Two launch paths:

- `stab_gate` — one Clifford gate, `Wr` threads (one per column word), plain XOR
  into `sign`.
- `stab_layer` — a whole layer of gates on **disjoint** qubits in one launch
  (`n_ops·Wr` threads). Disjoint qubits ⇒ no `x`/`z` race; gates share `sign`
  words, so the sign update is an `atomicXor`. This amortises kernel-launch
  overhead over the layer and is what makes the GPU competitive at the smaller
  `n` (where one tableau is still cache-resident).

`stab_init` builds `|0…0⟩` (destabiliser `c = X_c`, stabiliser `c = Z_c`) on the
device — no host-side `O(n²)` buffer or upload.

## Correctness

`tests/stab_oracle.rs` pins the GPU tableau against the CPU `aleph-stab` tableau
(`Tableau::export_generators`), bit-for-bit over the `2n` generator rows, for
`n ∈ {5, 33, 64, 65, 130, 200}` (straddling word boundaries) on random 40-layer
Clifford circuits. **Both** kernels are checked — the batched `apply_layer` and
the per-gate `apply` — plus the freshly-initialised `|0…0⟩` state. Clifford
evolution is deterministic, so bit-equality is an exact oracle; `aleph-stab` is
itself pinned against Stim's canonical stabiliser group, so this transitively
ties the GPU to Stim.

## Performance

RTX 4000 SFF Ada (sm_89), FP64-irrelevant (pure bit ops). Random Clifford
circuit (mixed H/S/CNOT/Pauli, disjoint layers), throughput in **M gates/s**,
best of N runs. GPU = batched `apply_layer` (fresh allocate + all layers + sync);
CPU = `aleph-stab` single-thread; Stim = `TableauSimulator.do_circuit` on the
identical serialised circuit.

### depth = 100

| n     | gates    | GPU  | CPU aleph | Stim | GPU/CPU | GPU/Stim |
|-------|----------|-----:|----------:|-----:|--------:|---------:|
| 1000  | 66 800   | 88.5 | 35.7      | 28.4 | 2.48×   | 3.12×    |
| 4000  | 266 698  | 60.6 | 5.2       | 6.1  | 11.66×  | 9.90×    |
| 16000 | 1 066 786| 15.1 | 1.2       | 1.3  | 12.25×  | 11.81×   |
| 65000 | 4 333 030| 3.6  | 0.3       | 0.4  | 11.01×  | 8.48×    |

### depth = 1000

| n     | gates      | GPU  | CPU aleph | Stim | GPU/CPU | GPU/Stim |
|-------|------------|-----:|----------:|-----:|--------:|---------:|
| 1000  | 667 097    | 84.8 | 40.1      | 28.2 | 2.11×   | 3.01×    |
| 4000  | 2 667 519  | 53.6 | 5.8       | 6.2  | 9.29×   | 8.65×    |
| 16000 | 10 667 131 | 15.2 | 1.5       | 1.3  | 10.19×  | 11.34×   |

The numbers track the `depth = 100` sweep — throughput is depth-independent (more
layers, same per-gate cost), so the acceptance target (`n ≥ 1000`, `depth ≥ 1000`,
faster than Stim) holds at full depth.

The GPU advantage is small at `n = 1000` (the tableau is tiny and the kernels are
launch/occupancy bound) and opens to ~8–12× once `n ≥ 4000`, where the `O(n²)`
tableau gives the GPU enough width to bury single-thread Stim and the CPU. The
absolute throughput falls with `n` for every engine because the per-gate work is
`O(n)` words; the *ratio* is the point.

### Caveats — what this does and does not claim

- This measures **single-state Clifford tableau evolution**, the apples-to-apples
  task for `TableauSimulator`. It is *not* Stim's headline regime: Stim's batched
  **Pauli-frame sampling** amortises one tableau across thousands of shots, and
  for shot-sampling Stim remains the tool to beat. A GPU frame sampler (one
  thread-block per shot) is the natural follow-up that would target that regime.
- The CPU `aleph-stab` arm is single-threaded; Stim is the stronger CPU baseline
  and the one the acceptance criterion names.
- **Measurement/sampling readout is not yet on the GPU.** This PR delivers
  device-resident Clifford *evolution* + bit-exact readout against the CPU
  tableau. GPU measurement (anticommuting-row search + `O(n²)` rowsum, the part
  the issue flags as warp-sync-bound) is the next slice.

## Reproduce

```bash
# correctness
cargo test -p aleph-cuda --features cuda --release --test stab_oracle

# throughput (Stim arm needs a python with `stim`)
ALEPH_STAB_NS="1000,4000,16000,65000" ALEPH_STAB_DEPTH=1000 \
  ALEPH_STIM_PY=/path/to/venv/bin/python \
  cargo test -p aleph-cuda --features cuda --release \
  -- --ignored --nocapture stab_bench
```
