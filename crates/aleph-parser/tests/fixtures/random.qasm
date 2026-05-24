// 4-qubit "random" circuit exercising every gate class used by P0-08.
OPENQASM 3.0;
include "stdgates.inc";

qubit[4] q;
bit[4] c;

h q[0];
x q[1];
y q[2];
z q[3];
s q[0];
sdg q[1];
t q[2];
tdg q[3];
rx(0.7) q[0];
ry(-1.2) q[1];
rz(pi/3) q[2];
p(pi/4) q[3];
u3(0.1, 0.2, 0.3) q[0];
cx q[0], q[1];
cz q[1], q[2];
swap q[2], q[3];
ccx q[0], q[1], q[2];
barrier q;
measure q -> c;
