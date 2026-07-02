# Q6-25 — Feed-forward on real silicon: teleportation byproduct driven by the Arty decoder

Follow-up to Q6-24 ("logical qubit in a box", closed-loop memory lifetime). There the decode result is
applied to a *passive* frame. Here it **conditions the next logical operation** — the teleportation
byproduct Pauli — which is the fault-tolerant primitive Q6-24 was the substrate for. The decode no
longer just scores a frame; it *steers a conditional quantum gate*.

```
|ψ⟩ ─┐   Bell-measure → two logical outcomes (m_x, m_z)
     │        each is a CODE-PROTECTED measurement: raw = m ⊕ e (logical meas error e)
     │        the REAL Arty decoder resolves ê from the syndrome block → corrected byproduct bit
     └──► X^{b_x} Z^{b_z} on the teleported qubit   ← conditional gate driven by the on-silicon decode
```

Each trial teleports a single-qubit stabilizer input (|0>,|1>,|+>,|->) through a **genuine
Aaronson–Gottesman CHP stabilizer tableau** running on the board (real prep, Bell pair, Bell
measurement with real outcome randomness, conditional X/Z byproducts, real verification measurement).
Teleportation needs two logical-measurement outcomes to select the byproducts; each is a code-protected
measurement whose raw value `m ⊕ e` must be decoded. The two decodes run on the **real Arty** streaming
decoder (one memory-Z block each, reusing the Q6-20 bitstream unchanged). Per trial, paired on the same
Bell outcomes, we contrast:

- **ON** — byproduct = decoder-corrected outcome (`raw ⊕ ê`) → teleportation succeeds at ~(1−LER);
- **OFF** — byproduct = raw undecoded outcome (`raw`) → teleportation corrupted at the raw error rate.

## Result (real Arty Z7-20, d=3 W=9 C=3, uf_arty_dma_win.bit, 50 MHz PL, p=0.01, 8000 trials)

| metric | value |
|--------|-------|
| ON teleport fidelity (decoder-corrected byproduct) | **91.88 % ± 0.60** |
| OFF teleport fidelity (raw undecoded byproduct)     | 67.31 % |
| **gain from the real-time decoder**                 | **+24.56 pp** |
| per-input ON / OFF | \|0⟩ 92/66 · \|1⟩ 92/69 · \|+⟩ 92/67 · \|−⟩ 93/67 |
| decoder (measured on silicon) | mean 1.37 µs/window, worst 1.66 µs vs 3 µs budget (1.8× headroom) |
| decode throughput | 21.3k decodes/s (2 decodes/trial) |

ON ≈ 92 % matches 1 − LER (the Q6-24 software UF baseline at p=0.01 was 7.88 % → 92.1 %). The
+24.6 pp ON−OFF gap is the on-silicon proof that the real-time decode **steers** the computation — a
thing that cannot exist in open-loop trace-replay. The off-board self-check (perfect decoder ê=e) gives
ON = 100.00 %, confirming the CHP gadget + byproduct logic are correct.

## What is / isn't new

- **New vs Q6-24:** a real conditional quantum operation (teleportation byproduct) executed by a
  genuine stabilizer gadget, with the byproduct measurements resolved by the silicon decoder in the
  loop, and the ON/OFF contrast demonstrating decode-steered control flow.
- **Honest scope:** for Clifford inputs teleportation success reduces to "was the byproduct decode
  right" (= composed memory-LER), so the *rate* is not a new error mechanism. Genuinely non-reducible
  feed-forward (magic-state / adaptive-T, where a missed byproduct injects real superposition
  randomness) needs a non-Clifford state-vector logical sim — a separate track.

## Reproduce

```bash
# 1. generate teleportation trials (two byproduct-measurement blocks per trial)
cargo run --release -p aleph-qec --example qec_q6_feedforward -- 3 9 3 17 8000 2024 0.01 > hw/cosim_ff_d3.vec

# 2. validate the CHP teleportation gadget off-board (perfect decoder -> ON=100%)
python3 hw/sw/uf_qubit_feedforward.py --selfcheck hw/cosim_ff_d3.vec

# 3. run on the real Arty (root + XRT env): real decoder resolves the byproducts in the loop
scp -i ~/.ssh/arty_pynq hw/sw/uf_qubit_feedforward.py hw/cosim_ff_d3.vec xilinx@10.0.1.182:~/
ssh -i ~/.ssh/arty_pynq xilinx@10.0.1.182 \
  'sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
     uf_qubit_feedforward.py uf_arty_dma_win.bit cosim_ff_d3.vec --trials 8000'
```

## Next levers

1. **Non-Clifford feed-forward** — state-vector logical sim so a missed byproduct injects genuine
   superposition randomness (the numerically non-reducible primitive; magic-state / adaptive-T).
2. **Reactive stabilizer** — recovery applied to a real stabilizer state so decoder mistakes corrupt
   *future* rounds.
3. **Two-qubit lattice-surgery feed-forward** — joint-parity measurement conditions a Pauli on a
   second patch (needs a merged-patch decode graph + bitstream).
