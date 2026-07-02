#!/usr/bin/env bash
# Q6-29 — algorithm fidelity vs physical error rate, on the real Arty Z7-20.
#
# Runs the logical Grover (28 T-decodes) and Toffoli (7 T-decodes) end-to-end from the silicon decoder
# at several physical error rates p, so the fidelity(p) curve shows how the decoder operating point
# (equivalently, decoder quality) sets the algorithm's output fidelity — the ASIC argument in data:
# a lower effective logical error rate drives the algorithm toward its ideal output.
#
# Reuses the Q6-27/Q6-28 emitters + board drivers unchanged; no new gadget code.
#
# On the Mac (generate the per-p vecs, then scp + run on the board):
#   for p in 0.001 0.002 0.003 0.005; do
#     cargo run --release -p aleph-qec --example qec_q6_grover  -- 3 9 3 17 256 2024 $p > hw/cosim_grover_p$p.vec
#     cargo run --release -p aleph-qec --example qec_q6_toffoli -- 3 9 3 17 800 2024 $p > hw/cosim_toffoli_p$p.vec
#   done
#   scp -i ~/.ssh/arty_pynq hw/sw/uf_qubit_grover.py hw/sw/uf_qubit_toffoli.py hw/cosim_*_p*.vec xilinx@10.0.1.182:~/
#   scp -i ~/.ssh/arty_pynq hw/sw/algo_fidelity_sweep.sh xilinx@10.0.1.182:~/
#   ssh -i ~/.ssh/arty_pynq xilinx@10.0.1.182 'sudo env XILINX_XRT=/usr bash algo_fidelity_sweep.sh'
#
# Prints one CSV line per (algorithm, p):  algo,p,on_fidelity,off_fidelity,found_rate

PY=/usr/local/share/pynq-venv/bin/python3
BIT=uf_arty_dma_win.bit

echo "algo,p,on,off,found"
for p in 0.001 0.002 0.003 0.005; do
  # Grover: parse "ON P(marked) = X%  [found-as-argmax Y%]" and "OFF ... = Z%"
  out=$($PY uf_qubit_grover.py $BIT cosim_grover_p$p.vec --trials 256 2>/dev/null | tr -d '\r')
  on=$(echo "$out"  | grep -oE 'ON  \(decoder-corrected\) = [0-9.]+' | grep -oE '[0-9.]+$')
  off=$(echo "$out" | grep -oE 'OFF \(raw undecoded\)      = [0-9.]+' | grep -oE '[0-9.]+$')
  found=$(echo "$out" | grep -oE 'found-as-argmax [0-9.]+' | grep -oE '[0-9.]+')
  echo "grover,$p,$on,$off,$found"

  # Toffoli: parse "ON  (decoder-corrected) = X%" / "OFF (raw undecoded) = Z%"
  out=$($PY uf_qubit_toffoli.py $BIT cosim_toffoli_p$p.vec --trials 800 2>/dev/null | tr -d '\r')
  on=$(echo "$out"  | grep -oE 'ON  \(decoder-corrected\) = [0-9.]+' | grep -oE '[0-9.]+$')
  off=$(echo "$out" | grep -oE 'OFF \(raw undecoded\)      = [0-9.]+' | grep -oE '[0-9.]+$')
  echo "toffoli,$p,$on,$off,"
done
