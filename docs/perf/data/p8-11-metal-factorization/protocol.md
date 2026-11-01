# Frozen P8-11 experiment protocol

Freeze before hardware pilot (2026-11-01). This is research, with production
routing, reconstruction guard and f64 fallback unchanged.

- Compare production single-group Jacobi with one-group-per-independent-pair,
  circle-scheduled Jacobi. Same FP32 rotation formula/tolerance (1e-6), 60-sweep
  cap, per-sweep host convergence wait; each round is a distinct encoder for
  inter-round device-buffer dependencies. Normalization parallel over columns.
- chi 32/64/128/256/512; square (2chi,2chi), tall (2chi,chi), wide (chi,2chi).
  Deterministic complex butterfly-densified well/clustered/ill/rank-deficient
  matrices, unit Frobenius norm; bond cap min(shape)/2, renormalization enabled.
- Stage timings include host encode/submit/wait, exclude compilation and fixture
  creation. Pack, factor, current finalize+guard, diagnostic sort/install with
  guard removed, faer f64 factor/full split, GPU Householder QR, faer CPU QR.
  Guard-free result is attribution-only; actual guard always runs for acceptance.
  Stage timings are intrusive and not summed into a claimed fused execution time.
- One untimed pack+factor warmup per mode per case. Per-case two repeats, orders reference/candidate then candidate/reference.
  No hidden retuning after looking at timing. Criterion pack+factor control on
  chi32/64, 10 samples, 1s warmup, 2s requested measurement; known expensive
  reference cells may extend it. Explicit actual-device acquisition, no skips.
- Numerical gates: finite data, full relative reconstruction <=1e-3 (current
  guard), retained-column max Gram error <=1e-4; discarded-weight, normalized
  Frobenius weight and flattened bipartite Z observable errors <=1e-5 vs faer.
  Report rejection/fallback frequencies, including failures; no guard disabling.
- GO requires an eligible chi>=128 regime, >=20% benefit versus both reference
  and full CPU split including the mandatory guard and fallback budget, and
  passing numerical gates for all frozen fixtures in that regime. Stage-only
  evidence is a necessary gate; any GO requires integration/end-to-end MPS and
  real bond distributions before changing production routing. Otherwise publish
  NO-GO for this prototype, keeping the existing backend untouched.
- Preserve current MPS oracle/canonical/truncation tests. Record host contention;
  no user processes stopped. Dedicated/thermal isolation is not assumed.
