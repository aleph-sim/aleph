## Q6-05 — Digilent Zybo Z7-20 / Zynq-7020 (xc7z020clg400-1).
## Out-of-context Fmax/fit study for the UF decoder PL block. In the real design `clk` is the PS
## FCLK_CLK0; here we only constrain it so the tool reports a meaningful Fmax. Board pin LOC /
## IOSTANDARD constraints belong to on-board bring-up (Q6-08), not to this OOC study.
create_clock -name clk -period 5.000 [get_ports clk]
## ^ 200 MHz target. Reported Fmax = 1000 / (5.000 - WNS).
