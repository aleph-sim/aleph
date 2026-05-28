OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
x q[1];
x q[2];
ccx q[1], q[2], q[0];
