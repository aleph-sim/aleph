OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
rx(pi/7) q[0];
ry(pi/3) q[0];
rz(pi/5) q[0];
