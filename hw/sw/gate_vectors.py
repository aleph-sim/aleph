#!/usr/bin/env python3
"""Q7-02 Task A4: flatten bp_circ_vectors.txt into the %b-per-line form tb_bp_gate_asap7.sv reads.

The golden file stores bit-strings with bit 0 FIRST; Verilog's %b puts the first character in the MSB,
so every string is reversed (and zero-padded to its declared width) here rather than in the testbench.
Usage: gate_vectors.py bp_circ_vectors.txt > bp_circ_vectors.gate.txt
"""
import sys

lines = [l.rstrip("\n") for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
T, N, C, OBS = map(int, lines[0].split())
print(T, N, C, OBS)
width = {"s": C, "h": N, "o": OBS, "v": 1}
for l in lines[1:]:
    tag, _, bits = l.partition(" ")
    bits = bits.strip()
    w = width[tag]
    bits = (bits + "0" * w)[:w]
    print(bits[::-1])
