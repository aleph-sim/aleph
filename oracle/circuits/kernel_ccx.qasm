OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
x q[0];
x q[1];
ccx q[0], q[1], q[2];
