# P4.6-02 — Stabilizer batched-shot (Pauli-frame) sampler

## Key realization
aleph's `Backend::sample(state, shots)` samples the **final** tableau (no gates
between measurements); the surface-cycle benchmark also measures ancillas after
all gates. So the full Stim mid-circuit frame simulator (gate conjugation
between measurements) is **not** needed. Instead:

**All shots share one identical x/z tableau** (same circuit → same collapses →
the same qubits are random; random/deterministic depends only on the x/z
*structure*, never on measured *signs*). Only the **signs** differ per shot (the
random coins). So promote `sign` from 1 bit to a **u64 word** (64 shots), do the
O(n²) x/z rowsum work **once per 64-shot batch**, and let per-shot variation live
entirely in sign-word XORs. This *is* the per-shot CHP algorithm with
`sign → word`, hence provably correct.

## Sign-word update
Scalar: `sign[h] = ((2·sign[h] + 2·sign[i] + phase) mod 4) == 2`, phase from
shared x/z (`rowsum_dispatch`). Since phase is even and the result ∈ {0,2}:
`sign[h] ^= sign[i] ^ phase_bit`, `phase_bit = (phase.rem_euclid(4) == 2)`
broadcast across all 64 shots (`phase_bit ? !0 : 0`).

## Implementation (aleph-stab/src/tableau.rs)
- `rowsum_sign(&mut self, sign_w, h, i)` — `rowsum_dispatch` on shared x/z; XOR
  sign-words with broadcast phase bit.
- `zero_row_sign` / `copy_row_sign` — x/z as today; sign-word reset/copy.
- `sample_qubits_batched(&self, qubits, shots, rng) -> Vec<u64>`:
  per 64-shot batch: clone x/z (RowMajor), broadcast final signs to words, run
  the measure pass over `qubits` (random branch sets `sign_w[p] = rng.gen::<u64>()`;
  deterministic branch reads `sign_w[scratch]`), scatter outcome words → per-shot
  bitstrings (bit i = qubits[i]).
- `Backend::sample` calls it with `qubits = 0..n` (replaces the per-shot loop).

Speedup: x/z rowsum done once per 64 shots → ~64× amortization on the dominant
cost (≫ AC-1's 10×). Future lever: 512-wide sign frames (zmm) — out of scope v1.
Pauli-noise seam (P4.6-04): a sign/X-frame injection hook at measurement — left
visible, not built.

## Tests
- **Oracle (AC-2):** batched distribution vs per-shot `sample()` within
  `assert_distribution_close` (P3-16 helper); GHZ/Bell/random Clifford.
- **Bit-exact unit:** forcing all 64 coins equal (seeded) reproduces a scalar
  per-shot run with that coin.
- **proptest:** random Clifford circuit → batched marginals match per-shot.
- **Determinism (AC-3):** same seed → identical shot table.
- **Stim cross-check (AC-2):** surface-code fixtures.
- **Bench (AC-1, EPYC):** batched vs 1024 sequential single-shot, surface-d11.

## Notes
- Non-Clifford rejection (AC-3) is automatic: the stabilizer backend rejects
  non-Clifford at `apply_gate`, so `sample` only ever sees a Clifford final state.
- n ≤ 64 already guarded in `Backend::sample`.
