// Q7-02 Task A4: behavioural DHLx1 (transparent-high latch) for Verilator, which has no UDP support.
// Same role as ORFS dff.v for DFFHQNx*. Function only; timing is not modelled (zero-delay functional sim).
module DHLx1_ASAP7_75t_R (Q, D, CLK);
    output reg Q;
    input D, CLK;
    always_latch if (CLK) Q = D;
endmodule
