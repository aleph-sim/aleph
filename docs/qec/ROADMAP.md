# QEC Decoders — Track Roadmap (North Star)

> Long-term track: from the aleph quantum simulator toward a **QEC decoder** and
> **decoder hardware**.
> This is a living document — strategy and end goal. Detailed issues live in
> `docs/qec/BACKLOG.md`. Update it as clarity emerges.

-----

## 0. North Star — the end goal (so we don't lose it)

**A physical decoder chip (ASIC) that decodes surface-code / qLDPC syndromes in real time
(< 1 µs/round, < 100 mW), co-designed with the aleph simulator.**

This is the endgame. It sits at the end **on purpose**: the ASIC is the finish line, not the
start. The path to it runs through a software decoder → GPU decoder → FPGA → and only then
ASIC. Every step is a self-contained deliverable with its own value, even if we never reach
the ASIC.

> **Honest disclaimer.** An ASIC tape-out is $1M+ and 18+ months. It only makes sense *after*
> the FPGA results are competitive and there is a real customer (a QPU company). Do not spend
> money on an ASIC until Q6 (FPGA) is done and the business gate Q7-03 is cleared. But we keep
> the goal in mind from day one — it drives every architectural decision below.

-----

## 1. Where we actually are (2026-06): reality vs the original plan

The original document was written assuming the simulator was at Phase 3. **That is no longer
true.** aleph has shipped Phases 0–5.11. So what the old plan called "Phase A: build the
stabilizer" is **mostly done**.

What aleph **already has** that serves this track:

| Component | Where | Status |
|-----------|-------|--------|
| Stabilizer (CHP tableau, Aaronson-Gottesman) | `crates/aleph-stab` | ✅ beats Stim 2-3× on CPU |
| GPU stabilizer | `crates/aleph-cuda/src/stab` | ✅ 3-12× CPU at n=1K-65K |
| Surface code (rotated) + syndrome extraction | `benches/src/lib.rs` | ✅ + Stim oracle, logical-error tests |
| Pauli-frame batched sampling (64-shot) | `crates/aleph-stab` | ✅ Monte-Carlo infrastructure |
| Deep CUDA expertise | Phases 5.x | ✅ this is our moat |

What is **missing** and defines the start of the track:

1. ❌ **No decoder at all** (neither MWPM nor Union-Find).
2. ❌ **The stabilizer cannot inject noise** → no noisy syndromes → no closed loop
   `noise → syndrome → decode → correct → logical error rate`. This loop is the **core
   instrument** of all decoder research, and it is our first blocker (Phase Q0).

-----

## 2. Where the truth is (honest strategic analysis)

1. **An ASIC right now is a mistake.** The real hardware milestone is **FPGA**, not ASIC.
   The ASIC is years out. (See North Star above.)

2. **MWPM on the surface code is effectively "solved."** Sparse Blossom / PyMatching does
   ~a million errors per core-second; Riverlane has Collision Clustering on FPGA/ASIC. We
   build MWPM **as a baseline and to learn**, not as a product.

3. **The genuinely open frontier in 2025-2026:**
   - **qLDPC decoders** (BP+OSD, relay-BP) for bivariate-bicycle / gross codes. The surface
     code is qubit-expensive; the field is moving to qLDPC, where the decoder is *not* solved.
   - **Real-time / streaming** (sliding/parallel window, the "backlog problem"): not accuracy
     but **latency < 1 µs in a stream**.

4. **Our moat is not "another decoder" — it's the GPU.** Nobody has a tightly co-designed
   *fast GPU stabilizer + GPU decoder*. That gives us (a) massive Monte-Carlo for threshold
   studies, (b) a GPU decoder as a genuine technical contribution. CUDA depth is rare in the
   QEC community (mostly Python+C++ and FPGA people).

5. **The economic truth.** This is **not** "the next Splynx with an ASIC exit in 5 years." It
   is a path to technical respect → a role at Riverlane/AWS/Google, or a co-founded decoder
   company. The simulator is the best possible portfolio + research instrument. The decoder is
   the product.

-----

## 3. Engineering Track — phases Q0…Q7

The engineering track (distinct from the career phases A-E below). Each phase is a set of
issues in `docs/qec/BACKLOG.md`. Issue IDs: `Q{phase}-{nn}` (e.g. `Q0-01`).

| Phase | Name | Exit metric | Depends on |
|-------|------|-------------|------------|
| **Q0** | Experiment Loop Foundation | Reproduce surface-code threshold ~1%, cross-checked with Stim | — |
| **Q1** | MWPM Decoder | Logical error rate equals perfect matching; bench vs PyMatching | Q0 |
| **Q2** | Union-Find Decoder | UF faster than MWPM on ≥1 regime, accuracy documented | Q1 |
| **Q3** | GPU Decoder (differentiator) | GPU decoder beats CPU MWPM/UF on throughput; GPU Monte-Carlo | Q1, Q2 |
| **Q4** | Real-Time / Streaming | Sliding-window decode without backlog blow-up; per-round latency budget | Q1/Q2 |
| **Q5** | qLDPC Frontier | BP+OSD on a gross code, threshold within literature range | Q1 |
| **Q6** | FPGA | UF on Arty A7 with measured latency; GPU-vs-FPGA report | Q2 (Q3) |
| **Q7** | **ASIC (North Star)** | Architecture spec + RTL core + tape-out feasibility / customer gate | Q6 |

**Q6 status (pre-silicon, on KV260 + Zybo via Vivado post-route):** the synthesizable UF surface
decoder is **real-time at d=5 on both boards** (KV260 294 ns / Zybo 655 ns) and **at d=7 on KV260**
(562 ns, ~1.8× under the ~1 µs round budget); Zybo d=7 (1.185 µs) is the one cell still over. Bit-exact
vs the CPU reference at d=3/5/7. **Q6-03 (GPU-vs-FPGA report) is done** and returns a **conditional GO
for Q7**: FPGA beats the GPU UF decoder ~10–50× on single-decode latency and 150–600× on energy/decode;
its latency is 76–82 % routing, the tax an ASIC removes. Remaining board ACs (Q6-01/02/08) need
physical hardware (KV260 in hand) and would replace the post-route estimates with measured silicon.
**Q7 trigger is commercial, not technical** (funding + a committed QPU-company customer).

**Recommended start: Q0.** It unblocks everything downstream, builds entirely on what already
exists, and delivers the first publishable result (a threshold plot) for little work.

-----

## 4. Personal / Career Roadmap

The engineering track above is *what to build*. This section is *why, and where it leads
career-wise*.

### Phase A: Foundation — ✅ mostly done via the simulator

Stabilizer + surface code + GPU stabilizer are ready. The remainder of A is reading Fowler et
al. "Surface codes" (ArXiv:1208.0928) and Aaronson-Gottesman in full, doing the Stim
tutorials, and reproducing the threshold (= Phase Q0). Decision point: is the topic
captivating, not merely interesting?

### Phase B: Decoder Implementation — = Phases Q1-Q3

MWPM + Union-Find + GPU decoder from scratch in Rust/CUDA, integrated with the simulator,
benchmarked vs PyMatching/FusionBlossom. Deliverable: an open-source decoder library + a
technical writeup (blog series or ArXiv preprint). Decision point: are the benchmarks
competitive → tier-1?

### Phase C: Real-time / Hardware — = Phases Q4, Q6

The shift from algorithm space to systems/hardware space. FPGA (Arty A7 ~$200 to start, not
ZCU106), Verilog/SystemVerilog or Chisel/Amaranth. Decision point: is hardware enjoyable or
painful?

### Phase D: Community engagement (in parallel, indefinitely)

ArXiv quant-ph daily; follow Gidney/Higgott/Riverlane; Quantum Computing Stack Exchange; QOSF
Slack. Conferences (1/year): QIP, APS March Meeting, IEEE Quantum Week, Q2B. Contribute PRs to
Stim/PyMatching/Qiskit. **The QEC community is small (~500 people active); ~50 key people are
directly reachable.**

### Phase E: Career move (months 18-36)

1. **Join an existing QEC company** (most likely): Riverlane (most direct fit), AWS CQC, Google
   Quantum AI, Microsoft Azure Quantum, IBM, PsiQuantum. Senior SWE/FPGA roles ~$150-250K
   US/UK. Portfolio (simulator + decoder) is ideal.
2. **Found a decoder company.** Precedent: Riverlane was founded by a non-physicist (Steve
   Brierley). Needs a hardware/ASIC co-founder + a customer (a QPU company) + pre-seed
   ($1-5M: Playground Global, Quantonation, In-Q-Tel, Cambridge Innovation Capital).
   Realistic: 2028-2029, after a Splynx exit. **This is the path to the North Star (ASIC).**
3. **PhD** (not recommended given age/experience): ETH, TU Delft (QuTech), Caltech (Preskill),
   Sydney, Edinburgh.
4. **Consulting + advisory** ($150-300/hour, ~$200K/year): lower commitment, lower ceiling.

-----

## 5. Realistic Milestones (time vs check)

| Stage | Detail | Sanity check |
|-------|--------|--------------|
| Now | Q0 — close the loop, threshold ~1% | Is the surface code clearer than "magic" now? |
| +3 mo | Q1 — MWPM, bench vs PyMatching | Is the implementation competitive with open source? |
| +6 mo | Q2-Q3 — UF + GPU decoder | Does the GPU give a real edge? |
| +9 mo | Q4 — real-time, first FPGA experiment | Is digital hardware interesting or painful? |
| +12 mo | Public visibility — blog/preprint | Is the community reacting? |
| +18-36 | Career move / company / (far) ASIC | Concrete opportunities in hand? |

**First real decision point: ~6 months.** If by then you've concluded it isn't for you — fine,
you saved years. Re-check yourself every 6 months; don't invest 3 years blind.

-----

## 6. Red flags

1. After 3 months, stabilizer formalism / syndromes are still confusing → not a native area.
2. FPGA brings frustration without compensating excitement → hardware isn't for you.
3. You publish work and the community doesn't react (no stars/engagement) → work isn't
   competitive enough (fixable) or the community won't accept it (fundamental).
4. After 12 months the portfolio isn't stronger than a typical PhD student after 2-3 years.
5. The Splynx exit happened, there's runway, but you're not excited to continue → the meaning
   was in Splynx.

-----

## 7. Key reading (priority order)

**Must-read:**
1. Fowler, Mariantoni, Martinis, Cleland, "Surface codes: Towards practical large-scale
   quantum computation" (2012). ArXiv:1208.0928. The canonical paper.
2. Aaronson, Gottesman, "Improved Simulation of Stabilizer Circuits" (2004). quant-ph/0406196.
3. Gidney, "Stim: a fast stabilizer circuit simulator" (2021). ArXiv:2103.02202.
4. Higgott, "PyMatching" (2021). ArXiv:2105.13082.

**Deeper into decoders:**
5. Delfosse, Nickerson, "Almost-linear time decoding algorithm for topological codes" (2017). ArXiv:1709.06218.
6. Higgott, Gidney, "Sparse Blossom: correcting a million errors per core second" (2023). ArXiv:2303.15933.
7. Bravyi et al., "High-threshold and low-overhead fault-tolerant quantum memory" (IBM gross codes, 2024). ArXiv:2308.07915.

**Hardware / real-time:**
8. Battistel et al., "Real-time decoding for fault-tolerant quantum computing" (2023). ArXiv:2303.00054.
9. Liyanage, Wu, Tannu, Holmes, "Scalable QEC for Surface Codes using FPGA" (2023). ArXiv:2301.08419.

**Context / business:**
10. Riverlane technical blog — https://www.riverlane.com/insights
11. Quantum Computing Report — https://quantumcomputingreport.com/

-----

## 8. Tools

**Software (free):** Stim, PyMatching, Tesseract, FusionBlossom, Qiskit, stim-surf.

**Hardware (Phase Q6):** Arty A7 (~$200, start), Xilinx ZCU106 (~$3000, serious work), Xilinx
Vivado (WebPack edition is free).

**Courses:** edX MIT 6.111 (digital systems), Coursera HDL for FPGA, MIT 8.371 (QIS III,
Preskill), Stanford CS269Q.

**Books:** Nielsen & Chuang; Lidar, Brun (eds.) *Quantum Error Correction* (2013); Patterson &
Hennessy *Computer Organization and Design*.

-----

## 9. The honest final look

QEC decoders are a niche where an honest 3-4 years of focused work can yield: (a) technical
respect in the community, (b) a career move into a quantum company, (c) possibly your own
company, if the edge is unique. This is **not** "change the world" in any romantic sense. It is
interesting technical work in a developing field with real industrial relevance.

The biggest risk: spending 2-3 years and not realizing you're in the wrong genre → which is why
Q0 (6 months) has an honest decision point. The biggest reward: doing work that matters in a
community that sees it. In 5 years — a name that serious players in QEC recognize.

Update this document as you progress. It is your personal map, not universal truth.
