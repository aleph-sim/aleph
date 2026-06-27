// Q6-02 (sim) — repetition-code Union-Find / minimum-weight decoder (the 1-D line case).
//
// The repetition (bit-flip) code: D data qubits, D-1 parity checks `c_i = e_i ^ e_{i+1}`. A valid
// correction `chat` must satisfy `chat_i ^ chat_{i+1} = s_i`, so it is fixed by the single free bit
// `chat_0` — the logical coset. Minimum-weight decoding (which on a 1-D matching graph is exactly
// what Union-Find computes: clusters grow from defects and pair adjacently / to a boundary) picks the
// coset of lower Hamming weight:
//
//   chat(chat_0=0) = prefix-XOR of the syndrome;   chat(chat_0=1) = its complement.
//
// We compute the prefix-XOR, its popcount `w0`, and choose the complement iff `2*w0 > D` (so the
// chosen weight <= D/2). The logical observable is X on data qubit 0, so the predicted logical flip
// is the chosen `chat_0`. This is a real datapath (prefix network + popcount + compare), not a ROM —
// the stepping stone to the 2-D surface-code cluster-growth Union-Find datapath (Q6-02 proper).
//
// Interface: 1-cycle valid/valid handshake. On `in_valid`, `syndrome` is latched and `correction`
// (D bits) + `obs_flip` appear with `out_valid` one clock later.

`timescale 1ns / 1ps

module uf_rep_decoder #(
    parameter int D = 7  // data qubits; D-1 = checks
) (
    input  logic               clk,
    input  logic               rst_n,
    input  logic               in_valid,
    input  logic [D-2:0]       syndrome,    // D-1 checks, LSB = check 0
    output logic               out_valid,
    output logic [D-1:0]       correction,  // data qubits to flip
    output logic               obs_flip     // predicted flip of logical observable 0 (= chosen chat_0)
);
  // Population count over a D-bit vector.
  function automatic int unsigned popcount(input logic [D-1:0] v);
    popcount = 0;
    for (int i = 0; i < D; i++) popcount += v[i];
  endfunction

  // Prefix-XOR of the syndrome with chat_0 = 0: prefix[j] = s[0] ^ ... ^ s[j-1]. A local
  // accumulator avoids reading the result vector mid-loop (Verilator ALWCOMBORDER-clean).
  function automatic logic [D-1:0] prefix_xor(input logic [D-2:0] s);
    logic acc;
    prefix_xor    = '0;
    acc           = 1'b0;
    for (int j = 1; j < D; j++) begin
      acc            = acc ^ s[j-1];
      prefix_xor[j]  = acc;
    end
  endfunction

  // Combinational decode of the current request.
  logic [D-1:0] prefix;          // chat for chat_0 = 0
  logic         use_complement;
  logic [D-1:0] cand;            // chosen (lower-weight) correction
  always_comb begin
    prefix         = prefix_xor(syndrome);
    use_complement = (2 * popcount(prefix)) > D;  // tie (even D) keeps chat_0 = 0
    cand           = use_complement ? ~prefix : prefix;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_valid  <= 1'b0;
      correction <= '0;
      obs_flip   <= 1'b0;
    end else begin
      out_valid  <= in_valid;
      correction <= in_valid ? cand : '0;
      obs_flip   <= in_valid ? use_complement : 1'b0;
    end
  end
endmodule
