# Q7 Track O, Task F5 — arXiv preprint + Zenodo data DOI (design)

Date: 2026-08-29. Author line: **Ruslan Malymon** (no affiliation, no ORCID). Approved in chat.

## Purpose
Make the Q7 decoder work discoverable and citable: the reason no stranger has deployed appliance v1 is
that nobody has seen the repository. One preprint (arXiv quant-ph, cross-list cs.AR) and one Zenodo
deposit (campaign CSVs + appliance-v1 bitstream) whose DOI the paper and README cite.

## Scope — no new measurements. Every number comes from an existing report in `docs/`.
| § | Content | Source reports |
|---|---|---|
| 1 | Intro, contributions | this spec |
| 2 | Decoder: relay-BP, Q5.3 fixed point, circuit-level DEM, 6×10 schedule, no OSD | qec-q7-fixed-bp.md M0, M5, M5-followups; qec-q5-*.md |
| 3 | Architecture ladder M2→M4→M7 banked→M8; Beneš/AS-Waksman gather; latch RF (Q7-08) | qec-q7-fixed-bp.md M2–M8, M9c; regfile-plan.md; asic-architecture.md |
| 4 | Silicon: KV260 15.64 µs worst / 0.85 µs median; Q7-06 10⁶×3 bit-exact; Q7-07 heralding | qec-q7-fixed-bp.md M8; q7-06-ac1-batched-dma.md; q7-07-nonconvergence-policy.md |
| 5 | Scaling: VU47P 64/192, 144/864 (B2); ASAP7 543 cyc @ 614.6 MHz (B3); real-activity power 0.31 µJ (A4) | q7-02-fullparallel-fpga.md, q7-02-b3-asap7-fullparallel.md, q7-02-asap7-timing.md |
| 6 | Limitations: hold on latch RF, 7 nm predictive vs 28 nm, streaming core no-fit, no OSD, runt frames | q7-02-asap7-timing.md §4b/§7, open-silicon-program.md risk register |
| 7 | Reproducibility: deploy.sh, CI gates, DOI | hw/product/*, .github/workflows/hw.yml |

## Artefacts
- `paper/main.tex` (revtex-free: plain `article`, 10pt, two-column via `multicol`-free `\documentclass[twocolumn]`), `paper/sections/*.tex`, `paper/refs.bib`, `paper/figs/` (3 plots from `docs/perf/data/*.csv` via `paper/figs/make_figs.py`, matplotlib from `scripts/qiskit-baseline/.venv`), `paper/Makefile` (tectonic).
- Zenodo deposit: `docs/perf/data/qec-q7-*.csv`, `qec-q5-*.csv`, appliance-v1 bitstream tarball; DOI into `paper/main.tex`, `README.md`, `hw/product/README.md`. Needs the user's Zenodo login — the deposit step is handed to the user with a prepared `paper/zenodo/` bundle + `.zenodo.json`.

## Style rules
British/US-neutral English, numbers exactly as in the source report with the report named in a
footnote-free way (a "Provenance" appendix table maps every figure to its `docs/` file and commit).
Nothing predictive is called "the chip"; ASAP7 is always "7 nm predictive". Negative results stay in.

## Delivery
PR 1: skeleton + §2–§5 drafted + figures + builds with tectonic. PR 2: §1, §6, §7, abstract, provenance
appendix, Zenodo bundle, DOI wiring.
