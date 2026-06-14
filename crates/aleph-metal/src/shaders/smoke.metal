#include <metal_stdlib>
using namespace metal;

// Doubles each element in place. Trivially verifiable host-side and exercises a
// read-modify-write over unified memory — enough to prove the full dispatch
// path (device -> queue -> runtime-compiled pipeline -> buffer -> readback).
// Contract: the caller must dispatch exactly one thread per element (grid size
// == buffer length); the kernel performs no bounds check.
kernel void smoke_double(device float* buf [[buffer(0)]],
                         uint i [[thread_position_in_grid]]) {
    buf[i] = buf[i] * 2.0;
}
