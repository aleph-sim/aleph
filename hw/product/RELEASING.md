# Cutting an appliance release

The appliance ships on its own tag series, **`appliance-vN`**, separate from the simulator's `vN.N.N`
tags. They version different things and must not share a release: `.github/workflows/release.yml` is
tag-gated on `v*` and asserts the tag matches `Cargo.toml`, which has nothing to do with a bitstream.

There is no CI job for this, and that is deliberate: **GitHub runners have no Vivado**, so the bitstream
cannot be rebuilt by a workflow. It is a binary artefact produced on a machine with the toolchain, and
the release is therefore a manual, documented act rather than an automated one. What CI *can* still
guarantee — that the RTL the bitstream was built from is bit-exact against the golden — it already does
on every push (`.github/workflows/hw.yml`).

-----

## What goes in a release

| file | what it is |
|---|---|
| `bp_kv260_stream_banked_p003.bit` | the PL image: banked 16/48 core, batched AXI-DMA front end, noise prior p = 0.003 |
| `bp_circ_vectors.txt` | the 40-shot circuit-level golden — the same one the co-simulation gates on |
| `bp_stream_banked_kv260.py` | the self-test / batch driver |
| `SHA256SUMS` | checksums for all of the above; `deploy.sh` refuses to run anything that fails it |

`deploy.sh` is *not* a release artefact — users get it from the repository, because it is the thing that
fetches and verifies the release.

### The noise prior belongs in the filename

The bitstream bakes both the Tanner graph and the noise prior λ(p) it was built for. Feeding it traffic
generated at a different p, compared against a golden built at that other p, produces mismatches that
look exactly like an RTL defect. That is not hypothetical — it consumed a whole debugging campaign
(issue #478) before the divergence was traced to the prior rather than the core, which was bit-exact
throughout.

So the prior is in the filename, in the release notes, and asserted by `deploy.sh`. Do not publish a
bitstream whose name does not say what p it was built at.

-----

## Procedure

1. **Identify the exact commit the bitstream was built from.** Not "roughly main" — the commit. If you
   cannot name it, rebuild rather than guess; a published binary nobody can trace to a source tree is
   the one thing this project cannot ship.

2. **Re-run the correctness gate on that commit**, so the release notes state a fact rather than a
   memory:

   ```bash
   make -C hw bpbanked          # ~6 min, needs Verilator >= 5.050 and a Rust toolchain
   ```

3. **Collect the artefacts and checksum them:**

   ```bash
   mkdir -p /tmp/appliance-v1 && cd /tmp/appliance-v1
   cp /path/to/bp_kv260_stream_banked_p003.bit .
   cp ~/GitHub/aleph/hw/bp_circ_vectors.txt .
   cp ~/GitHub/aleph/hw/sw/bp_stream_banked_kv260.py .
   sha256sum bp_kv260_stream_banked_p003.bit bp_circ_vectors.txt bp_stream_banked_kv260.py > SHA256SUMS
   ```

4. **Create the release as a draft**, so nothing is public until a deployment has been walked through:

   ```bash
   gh release create appliance-v1 \
     --draft \
     --target <the-commit-sha> \
     --title "Decoder appliance v1 — KV260, banked 16/48, p=0.003" \
     --notes-file notes.md \
     bp_kv260_stream_banked_p003.bit bp_circ_vectors.txt bp_stream_banked_kv260.py SHA256SUMS
   ```

5. **Deploy from the draft onto a board that has been wiped**, using only `deploy.sh` and the README.
   Anything you have to fix by hand is a bug in the script, not a step to remember. Fix it and repeat.

6. **Publish**, and record in `hw/product/README.md` what was verified and by whom.

-----

## The release notes must say

Copy this shape; each line exists because omitting it has misled someone.

- **Configuration and geometry:** banked 16/48, gross bivariate-bicycle `[[144,12,12]]`, circuit-level
  noise model, **noise prior p = 0.003**.
- **Measured latency:** 15.64 µs worst case, 0.85 µs median with early exit, at 133.332 MHz on a KV260.
  Measured on silicon, not projected.
- **Correctness:** bit-exact against the software golden in co-simulation, and 0 mismatches in 10⁶ × 3
  shots on silicon (matched-prior campaign, p = 0.003 / 0.005 / 0.007).
- **What it is not:** not a surface-code MWPM decoder; not runtime-reconfigurable to another code; not a
  syndrome-extraction system; not sub-microsecond. Point at `interface-spec.md` §7.
- **`valid_flag` is load-bearing.** A converged decode is almost never wrong, so the decoder's logical
  error rate is essentially its non-convergence rate. Tell integrators not to discard it.
- **The source commit**, and the CI run that gated it.

-----

## What "verified" is allowed to mean

`hw/product/README.md` claims deployability. Be exact about which of these was actually done, because
the count of external deployments is the gate on everything downstream in the silicon programme, and
inflating it corrupts the only signal that matters:

| claim | what it takes |
|---|---|
| *the procedure is complete* | wipe the development board, deploy using only `deploy.sh`, self-test passes |
| *a stranger can deploy it* | someone who has never seen this repository does the above, unaided, and does not ask a question |

The first is worth doing and is much better than nothing — it catches every step that exists only in
somebody's memory. **It is not the second**, and only the second is what Task P1 Step 4 asks for. If
only the first has happened, say so in those words.
