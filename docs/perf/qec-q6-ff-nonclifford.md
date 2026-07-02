# Q6-26 — Non-Clifford feed-forward on real silicon: T-gate teleportation chain

Follow-up to Q6-25. That demo teleported *stabilizer* states, where a missed byproduct is a
deterministic Pauli flip, so feed-forward success reduces to composed memory-LER. This is the
genuinely **non-reducible** feed-forward the Q6-25 note flagged as the real frontier: its error
mechanism cannot be captured by any classical bit-flip model.

A logical qubit passes through `gates = 8` **T-gate teleportations**. T is non-Clifford, so the
intermediate states are non-stabilizer and the board simulates the logical qubit with a genuine 1-qubit
**state vector** (numpy), not a tableau. Applying T by gate teleportation (Gottesman–Chuang) needs a
Z-measurement of a magic ancilla and a *conditional S correction*; that measurement is code-protected
(`raw = m ⊕ e`) and is **decoded on the real Arty** streaming decoder (one memory-Z block per magic
measurement, Q6-20 bitstream unchanged). Since T⁸ = I, the correct chain returns the input |+⟩
(verify in X → 0).

**The signature:** a wrong decode applies an extra S. Because the magic outcome is random, that extra
gate is S or S†, and S is *not* a Pauli relative to the verification basis — so a wrong feed-forward
collapses the deterministic result to a **quantum-random** outcome (fidelity 0.5), not a bit flip. A
classical bit-flip (composed-LER) model predicts w≥1 → fidelity 0; the quantum reality is 0.5.

## Result (real Arty Z7-20, d=3 W=9 C=3, uf_arty_dma_win.bit, 50 MHz, p=0.005, gates=8, 4000 trials)

| metric | value |
|--------|-------|
| ON chain fidelity (decoder-corrected S) | **92.14 %** |
| OFF chain fidelity (raw undecoded S)     | 57.99 % |
| **mean fidelity when ≥1 decode wrong (w≥1)** | **0.502** (classical model: 0.0; quantum: 0.5) |
| quantum-random trials (fidelity ≈0.5, a superposition the verify basis can't resolve) | 14.4 % |
| fidelity vs w (from real decodes) | w=0 → 1.00 (3369) · w=1 → 0.50 (575) · w=2 → 0.53 (53) · w=3 → 0.50 (2) |
| decoder (measured on silicon) | mean 1.32 µs/window, worst 1.58 µs vs 3 µs budget (1.9× headroom) |

The **w≥1 → 0.502** measured on the real decoder is the non-reducibility proof: a wrong on-silicon
decode injects quantum randomness, not a classical flip. 14.4 % of trials land in a genuine
superposition w.r.t. the verification basis — a regime no stabilizer/composed-LER description can
produce. The off-board self-check (perfect decoder ê=e) gives ON = 100.00 %, confirming the T-gate
gadget; injecting synthetic wrong decodes reproduces the w≥1 → ~0.5 pattern.

## What is / isn't new

- **New vs Q6-24/Q6-25:** the feed-forward error mechanism is genuinely non-Clifford — its logical
  effect (quantum superposition, fidelity 0.5) is *not* reducible to composed memory-LER, and requires
  a state-vector logical sim to model. This is the first demo in the track where "the decode drives the
  computation" produces a quantum-mechanically distinct outcome, not just a relabeled bit.
- **Still true:** the *rate* of wrong decodes is the memory-LER; what's new is how a wrong decode
  *propagates* through the non-Clifford circuit. The magic states here are prepared directly (not
  distilled), and it is a single logical qubit — a full FT T-factory / multi-qubit non-Clifford circuit
  is the larger follow-up.

## Reproduce

```bash
cargo run --release -p aleph-qec --example qec_q6_ff_nonclifford -- 3 9 3 17 4000 2024 0.005 8 > hw/cosim_ffnc_d3.vec
python3 hw/sw/uf_qubit_ff_nonclifford.py --selfcheck hw/cosim_ffnc_d3.vec      # ON=100%, w≥1→~0.5
scp -i ~/.ssh/arty_pynq hw/sw/uf_qubit_ff_nonclifford.py hw/cosim_ffnc_d3.vec xilinx@10.0.1.182:~/
ssh -i ~/.ssh/arty_pynq xilinx@10.0.1.182 \
  'sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
     uf_qubit_ff_nonclifford.py uf_arty_dma_win.bit cosim_ffnc_d3.vec --trials 4000'
```

## Next levers

1. **Multi-qubit non-Clifford circuit** — a small logical algorithm (e.g. a few T-gates + CNOTs)
   verified by an expectation value, decode-driven end to end.
2. **Reactive stabilizer** — apply recovery to a real stabilizer state so decoder mistakes corrupt
   *future* rounds (the physically-closed loop).
3. **On the KV260 / higher distance** — run the same feed-forward with the d=7 KV260 decoder.
