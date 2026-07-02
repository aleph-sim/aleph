# Q6-28 — Small logical algorithm end-to-end from the decoder: 3-qubit Grover search

Follow-up to Q6-27 (a single logical Toffoli). This runs a full non-Clifford **algorithm** — 3-qubit
**Grover search** — end-to-end with the silicon decoder in the loop, verified by its output
distribution.

Grover on N=8 with one marked state runs H^⊗3 then 2 iterations of {oracle, diffusion}; the optimal 2
iterations peak the marked-state probability at **94.53 %** (ideal). Both the oracle and the diffusion
contain a CCZ = H·CCX·H, and CCX (Toffoli) = 7 T/T† gates (Q6-27). So the algorithm is 4 CCZ =
**28 T-gate magic-state injections**, each a code-protected logical measurement (`raw = m ⊕ e`)
**decoded on the real Arty** streaming decoder (one memory-Z block each, Q6-20 bitstream unchanged).
X/H/CNOT are Clifford (no decode). A wrong decode inserts an extra S mid-algorithm, corrupting the
amplitude amplification — so the board measures the marked-state probability with the decoder ON
(corrected) vs OFF (raw) vs the uniform baseline 1/8.

## Result (real Arty Z7-20, d=3 W=9 C=3, uf_arty_dma_win.bit, 50 MHz, p=0.003, 512 searches, all 8 marked states)

| P(marked state) | value |
|-----------------|-------|
| **ON (decoder-corrected)** | **86.42 %**  (found-as-argmax **96.1 %**) |
| OFF (raw undecoded) | 26.38 %  (found 43.0 %) |
| uniform (no search) | 12.50 % |
| ideal 2-iteration Grover | 94.53 % |
| per-marked ON | \|000⟩ 82 · \|010⟩ 89 · \|110⟩ 89 · \|111⟩ 87 · … (all 8 consistent) |
| decoder (measured) | mean 1.29 µs/window, worst 1.54 µs vs 3 µs budget (1.9× headroom) |
| decode throughput | 21.9k decodes/s (28 decodes/search) |

ON ≫ OFF ≫ uniform: the algorithm amplifies the marked state end-to-end with the silicon decoder
resolving all 28 magic-state injections in real time. In 96.1 % of searches the marked state is the
most-likely measured outcome. The ON vs ideal gap (86.4 % vs 94.5 %) is the cumulative effect of real
decode errors over 28 T-gates. The off-board self-check verifies the Grover circuit reaches the ideal
peak **exactly** on all 8 marked states (perfect decoder → 94.53 %), so the only unknown on the board
is the decoder.

## What is / isn't new

- **New vs Q6-27:** a full non-Clifford *algorithm* (not one gate) — Grover's oracle + diffusion over
  2 iterations, 28 decoded magic-state injections — verified by an output distribution peaked on the
  marked state. This is the "small logical algorithm end-to-end from the decoder" milestone.
- **Still true:** the *rate* of wrong T-decodes is the memory-LER; magic states are prepared directly
  (not distilled) and the Cliffords are exact. A full FT resource-state factory + lattice-surgery
  entangling gates is the larger follow-up.

## Reproduce

```bash
cargo run --release -p aleph-qec --example qec_q6_grover -- 3 9 3 17 1024 2024 0.003 > hw/cosim_grover_d3.vec
python3 hw/sw/uf_qubit_grover.py --selfcheck hw/cosim_grover_d3.vec        # circuit EXACT, peak 94.53%
scp -i ~/.ssh/arty_pynq hw/sw/uf_qubit_grover.py hw/cosim_grover_d3.vec xilinx@10.0.1.182:~/
ssh -i ~/.ssh/arty_pynq xilinx@10.0.1.182 \
  'sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
     uf_qubit_grover.py uf_arty_dma_win.bit cosim_grover_d3.vec --trials 512'
```

## Next levers

1. **Reactive stabilizer** — apply recovery to a real stabilizer state so decoder mistakes corrupt
   *future* rounds (the physically-closed loop).
2. **KV260 / d=7** — run the same algorithm with the higher-distance decoder (lower LER → closer to the
   ideal Grover peak).
3. **Larger circuit** — more qubits / a resource-state factory with lattice-surgery entangling gates.
