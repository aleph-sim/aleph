# Q6-27 — Multi-qubit non-Clifford algorithm end-to-end from the decoder: logical Toffoli

Follow-up to Q6-26 (single-qubit T⁸ chain). This runs a real **3-qubit non-Clifford algorithm** — a
logical **Toffoli (CCX)** — end-to-end with the silicon decoder in the loop. Toffoli's standard
decomposition (Nielsen & Chuang §4.3) is **7 T/T† + 6 CNOT + 2 H**; CNOT/H are Clifford (transversal in
an FT machine, no decode), and each of the 7 non-Clifford T's is applied by gate teleportation whose
magic-ancilla Z-measurement is code-protected (`raw = m ⊕ e`) and **decoded on the real Arty** streaming
decoder (one memory-Z block each, Q6-20 bitstream unchanged).

The 3 logical qubits are a genuine 8-amplitude **state vector** (numpy). A wrong decode inserts an extra
S mid-circuit — and because S lands before the decomposition's H gates, the Toffoli output becomes a
superposition, not a bit flip. We verify the **classical truth table** (target flips iff both controls
are 1) with the decoder ON (extra S only where ê ≠ e) vs OFF (extra S wherever the raw measurement
erred, e ≠ 0).

## Result (real Arty Z7-20, d=3 W=9 C=3, uf_arty_dma_win.bit, 50 MHz, p=0.005, 2400 trials, all 8 inputs)

| metric | value |
|--------|-------|
| ON truth-table fidelity (decoder-corrected) | **95.92 %** |
| OFF truth-table fidelity (raw undecoded)     | 69.71 % |
| **gain from the real-time decoder**          | **+26.21 pp** |
| per-input ON (all correct) | \|000⟩→\|000⟩ 96 · \|011⟩→\|011⟩ 95 · **\|110⟩→\|111⟩ 96** · **\|111⟩→\|110⟩ 95** · … |
| fidelity when ≥1 T-decode wrong (ON) | 0.690 (non-Clifford corruption — a partial-overlap superposition, not a clean flip) |
| decoder (measured on silicon) | mean 1.32 µs/window, worst 1.58 µs vs 3 µs budget (1.9× headroom) |
| decode throughput | 21.9k decodes/s (7 decodes/Toffoli) |

ON ≈ 95.9 % matches ~(1−LER)⁷ (all-7-correct ≈ 0.979⁷ ≈ 0.86) plus partial credit from the 0.69 mean
fidelity of wrong-decode trials. The off-board self-check verifies the decomposition reproduces the
Toffoli truth table **exactly** on all 8 inputs (perfect decoder → 100 %), so the only unknown on the
board is the decoder. The result is a real multi-qubit non-Clifford algorithm computed correctly with
the silicon decoder resolving all 7 magic-state injections in real time.

## What is / isn't new

- **New vs Q6-26:** a genuinely *multi-qubit* non-Clifford circuit (Toffoli, 3 logical qubits, 7 T's +
  CNOTs), verified against a classical truth table, with every non-Clifford gate's feed-forward driven
  by the real decoder. Errors propagate through the entangling CNOTs, so a wrong T-decode is a genuine
  quantum corruption (fidelity 0.69, not a clean flip).
- **Still true:** the *rate* of wrong T-decodes is the memory-LER; the magic states are prepared
  directly (not distilled) and CNOT/H are exact. A full FT resource-state factory + lattice-surgery
  CNOTs is the larger follow-up.

## Reproduce

```bash
cargo run --release -p aleph-qec --example qec_q6_toffoli -- 3 9 3 17 2400 2024 0.005 > hw/cosim_toffoli_d3.vec
python3 hw/sw/uf_qubit_toffoli.py --selfcheck hw/cosim_toffoli_d3.vec       # decomposition EXACT, ON=100%
scp -i ~/.ssh/arty_pynq hw/sw/uf_qubit_toffoli.py hw/cosim_toffoli_d3.vec xilinx@10.0.1.182:~/
ssh -i ~/.ssh/arty_pynq xilinx@10.0.1.182 \
  'sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
     uf_qubit_toffoli.py uf_arty_dma_win.bit cosim_toffoli_d3.vec --trials 2400'
```

## Next levers

1. **Small logical algorithm** — e.g. a 3-qubit adder or a Grover iteration (Toffoli + Cliffords),
   verified by its output distribution, decode-driven end to end.
2. **Reactive stabilizer** — recovery applied to a real stabilizer state so decoder mistakes corrupt
   *future* rounds.
3. **KV260 / d=7** — run the same non-Clifford algorithm with the higher-distance decoder.
