OPENQASM 3.0;
include "stdgates.inc";
qubit[6] q;
bit[2] c;
// Z-parity stabilizer on data q0..q3 via ancilla q4.
cx q[0], q[4];
cx q[1], q[4];
cx q[2], q[4];
cx q[3], q[4];
measure q[4] -> c[0];
// X-parity stabilizer on data q0..q3 via ancilla q5.
h q[5];
cx q[5], q[0];
cx q[5], q[1];
cx q[5], q[2];
cx q[5], q[3];
h q[5];
measure q[5] -> c[1];
