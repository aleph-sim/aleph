## Q7-02 — Digilent Arty Z7-20 / Zynq-7020 (xc7z020clg400-1).
## Out-of-context Fmax/fit study for the relay-BP decoder PL block. In the real design `clk` is a PS
## PL clock (fclk0); here we only constrain it for the Fmax report. Pin placement is out-of-context.
create_clock -name clk -period 10.000 [get_ports clk]
## ^ 100 MHz target on the -1 speed grade. Reported Fmax = 1000 / (10.000 - WNS).
