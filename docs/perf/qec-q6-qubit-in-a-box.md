# Q6-24 — "Logical qubit in a box": closed-loop memory lifetime on real silicon

Follow-up to Q6-21 (board-free sim↔RTL co-sim, #400) and Q6-20/Q6-22 (on-silicon streaming
decode + finite-experiment LER). Those flows are all **open-loop trace-replay**: a pre-generated
syndrome stream is decoded and the decoder's guess is scored against ground truth offline. The
decoder never touches the state.

This closes the loop. We hold a logical Pauli frame (start `|0⟩_L`). Each cycle is one finite
memory-Z experiment of `slices` rounds streamed through the **real** sliding-window streaming
decoder on the Arty Z7-20; the decoder returns a proposed logical correction (XOR of the committed
windows' obs bits). We apply it to the frame — the corrected frame is identity iff the correction
matches the cycle's true accumulated logical flip. A mismatch is a **logical failure**: the qubit
"died"; we record the survival interval, re-sync the frame to truth, and keep the box running. That
inter-failure statistic is the logical qubit's lifetime, kept alive in real time by the silicon
decoder.

```
[sim: memory-Z cycle] --detector rounds--> [REAL decoder on Arty] --logical correction-->
     ^                                                                                  |
     +------------------ correction applied to the tracked logical frame ---------------+
```

## Result (real Arty Z7-20, d=3 W=9 C=3, uf_arty_dma_win.bit, 50 MHz PL, 20 000 cycles/point)

| p (phys) | on-silicon LER | software UF baseline | verdict | logical lifetime | decoder worst | real-time |
|----------|----------------|----------------------|---------|------------------|---------------|-----------|
| 0.005    | 2.07 % ± 0.20  | 2.14 % ± 0.20        | MATCH   | 48 cyc ≈ 870 µs  | 1.58 µs       | ✅ 1.9× headroom |
| 0.010    | 7.88 % ± 0.37  | 7.81 % ± 0.37        | MATCH   | 13 cyc ≈ 228 µs  | 1.66 µs       | ✅ 1.8× |
| 0.020    | 23.0 % ± 0.58  | 22.6 % ± 0.58        | MATCH   | 4 cyc ≈ 78 µs    | 1.74 µs       | ✅ 1.7× |

Every operating point's on-silicon closed-loop LER matches the boundary-aware software
`SlidingWindowDecoder` within Monte-Carlo CI, and the decoder meets the C-round (3 µs) commit
budget worst-case at every point. Lifetime in µs assumes a 1 µs/round physical syndrome-extraction
cycle (`--round-ns`); that assumption scales only the time axis, not the decoder verdict. Decoder
latency/throughput are measured on the silicon (wall clock + the RTL per-window latency field,
result word bits[15:0], at the 50 MHz PL clock).

## What is / isn't new

- **New:** the maintained logical frame + the live reactive time-to-failure loop (lifetime, not a
  static LER table), and the artifact itself — a logical qubit a real decoder keeps alive in real
  time. It is the substrate for feed-forward, where `pred` would instead condition the *next*
  logical operation.
- **Not new:** for a memory qubit, "apply correction, check identity" equals `pred == truth`, so
  the per-cycle number coincides with the Q6-22 LER co-sim. This is d=3 (demo scale) and pure frame
  tracking — the correction does not yet perturb *future* syndromes (the "reactive stabilizer"
  upgrade), and there is no feed-forward yet.

## Reproduce

```bash
# 1. generate the operating-point stream (reuses the Q6-22 finite-experiment generator)
cargo run --release -p aleph-qec --example qec_q6_stream_ler -- 3 9 3 17 20000 2024 phenom 0.005,0.01,0.02 \
    > hw/cosim_qubit_box_d3.vec

# 2. validate the monitor math off-board (no board, no pynq)
python3 hw/sw/uf_qubit_in_a_box.py --selfcheck hw/cosim_qubit_box_d3.vec --p 0.005

# 3. run the box on the real Arty (root + XRT env)
scp -i ~/.ssh/arty_pynq hw/sw/uf_qubit_in_a_box.py hw/cosim_qubit_box_d3.vec xilinx@10.0.1.182:~/
ssh -i ~/.ssh/arty_pynq xilinx@10.0.1.182 \
  'sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
     uf_qubit_in_a_box.py uf_arty_dma_win.bit cosim_qubit_box_d3.vec --p 0.005 --cycles 20000'
```

## Next levers

1. **Feed-forward** — decode result conditions the next logical operation (true FT primitive; not
   reducible to LER).
2. **Reactive stabilizer** — apply recovery to a real stabilizer state so decoder mistakes corrupt
   *future* rounds (fully closed loop).
3. **Circuit-level noise** — switch to `uf_arty_dma_win_c.bit` + `noise=circuit` (lower threshold,
   honest gate-level physics).
