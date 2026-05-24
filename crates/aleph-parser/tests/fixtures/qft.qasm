// 3-qubit QFT (without final swap), expressed with cz + rz to exercise pi/2^k expressions.
OPENQASM 3.0;
include "stdgates.inc";

qubit[3] q;

h q[2];
cz q[1], q[2];
rz(pi/2) q[2];
h q[1];
cz q[0], q[1];
rz(pi/4) q[1];
cz q[0], q[2];
rz(pi/8) q[2];
h q[0];
