## Q6-05 — Xilinx Kria KV260 / Zynq UltraScale+ K26 (xck26-sfvc784-2LV-c).
## Out-of-context Fmax/fit study for the UF decoder PL block. In the real design `clk` is a PS
## PL clock; here we only constrain it for the Fmax report. Board pin constraints belong to Q6-08.
create_clock -name clk -period 3.000 [get_ports clk]
## ^ ~333 MHz target on the faster -2LV part. Reported Fmax = 1000 / (3.000 - WNS).
