#!/usr/bin/env bash
# Q6-31 — 2-D fidelity surface: algorithm fidelity vs (physical error rate p, T-gate count), on the real
# Arty Z7-20. Combines Q6-29 (p axis) x Q6-30 (T-count axis) into one grid by running the C^kX circuit
# (T = 14(k-1)) at each (k, p). Reuses the Q6-30 emitter + board driver unchanged.
#
# On the board (root + XRT env): prints one CSV line per (k, p): k,tcount,p,on,off
#   sudo env XILINX_XRT=/usr bash tcount_p_surface.sh

PY=/usr/local/share/pynq-venv/bin/python3
BIT=uf_arty_dma_win.bit

echo "k,tcount,p,on,off"
for k in 2 3 4 5; do
  t=$((14 * (k - 1)))
  for p in 0.001 0.002 0.003 0.005; do
    line=$($PY uf_qubit_mcx.py $BIT cosim_mcx_k${k}_p${p}.vec --trials 200 2>/dev/null \
           | grep -oE 'RESULT k=[0-9]+ T=[0-9]+ : ON=[0-9.]+ OFF=[0-9.]+')
    on=$(echo "$line"  | grep -oE 'ON=[0-9.]+'  | cut -d= -f2)
    off=$(echo "$line" | grep -oE 'OFF=[0-9.]+' | cut -d= -f2)
    echo "$k,$t,$p,$on,$off"
  done
done
