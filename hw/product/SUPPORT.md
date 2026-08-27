# Support policy — decoder appliance v1

One page. It says what is supported, what "stable" means, what is answered and how fast, and what to
send when something fails. It is deliberately narrow: the project is one maintainer, and a promise it
cannot keep is worse than none.

-----

## 1. What is supported

Exactly the configuration that has been deployed and verified, and nothing else:

| | supported |
|---|---|
| board | Kria **KV260** (Starter Kit) |
| OS | Ubuntu **22.04** Certified for Xilinx Devices (`iot-limerick-kria-classic-desktop-2204-*`) |
| PYNQ | **3.0.1** from Kria-PYNQ tag **v3.0**, installed per `BRINGUP.md` |
| image | release [`appliance-v1`](https://github.com/aleph-sim/aleph/releases/tag/appliance-v1) — banked 16/48, `[[144,12,12]]`, **p = 0.003** |
| deployment | `deploy.sh`, ending in `CORRECTNESS: PASS (40/40 ...)` |
| host interface | AXI4-Lite and batched AXI-DMA, as in `interface-spec.md` §2–§3 |

**Not supported, and reports about them will be answered with "not supported":** other Kria boards
(KR260, KD240), Ubuntu 24.04 or PetaLinux, any other PYNQ version, other noise priors or codes (a
different p or a different Tanner graph is a *different bitstream*, not a configuration option), the
serial transport of `interface-spec.md` §4 (not implemented), and any bitstream you rebuilt yourself.
The last one is welcome — the RTL is Apache-2.0 — but it is your build.

## 2. What "stable" means

Within the `appliance-v1` tag:

- The **self-test is the contract.** A board where `deploy.sh` reports 40/40 bit-exact against the
  golden is a working decoder. A board where it does not is not, whatever else appears to work.
- The **register map, data formats, bit order and `valid_flag` semantics** are frozen, per
  `interface-spec.md` §6. The same section says what is *not* promised — read it, especially about
  latency.
- **Release assets are immutable.** `SHA256SUMS` on the release page is the reference; `deploy.sh`
  refuses anything that does not match. A fix ships as a new tag (`appliance-v1.1`, `appliance-v2`),
  never as a silently replaced asset.
- **Measured numbers are not a promise.** 15.64 µs worst / 0.85 µs median is what one board did at
  133.332 MHz; it is stated so you can check yours, not guaranteed for yours.

## 3. What is answered, and how fast

Everything goes through [GitHub Issues](https://github.com/aleph-sim/aleph/issues). There is no
private channel, no email support and no phone.

| kind of report | response |
|---|---|
| `deploy.sh` fails on a supported configuration | **answered**; treated as a bug in this repository until shown otherwise. Target: first reply within 5 working days. |
| self-test passes, decoder gives wrong answers on your traffic | **answered**; this is the interesting case. First question will be whether your syndromes were generated at p = 0.003 (issue #478). |
| `BRINGUP.md` step does not work as written | **answered**; the document is part of the product. |
| unsupported configuration (§1) | closed as not supported, with a pointer if one exists |
| feature requests (other codes, other boards, serial transport) | read; left open as a demand signal; no commitment |

"Answered" means a human reads it and replies. It does not mean fixed by a date. This is a
best-effort, single-maintainer project; there is no SLA and no paid tier. If you need one, say so in an
issue — that demand is precisely what decides whether a v2/v3 gets built (`README.md`).

## 4. What to send

A report without these is answered with a request for them, which costs everyone a round-trip:

```bash
# 1. the self-test log
cat /opt/aleph-decoder/selftest.log
# 2. the versions
cd / && /usr/local/share/pynq-venv/bin/python3 -c \
  "import pynq, numpy, importlib.metadata as m; print(m.version('pynq'), numpy.__version__)"
lsb_release -ds; uname -r
tr -d '\0' < /sys/firmware/devicetree/base/chosen/*version* ; echo
# 3. the checksums actually on the board
sha256sum /opt/aleph-decoder/*
```

plus the full output of `sudo bash deploy.sh`, and for a wrong-answer report: the p your syndromes were
generated at, and one syndrome/correction pair that disagrees.

Successful deployments are wanted too — they are counted. Use the **Deployment report** issue
template; it takes two minutes and is the number that gates the silicon programme.

## 5. Security

The decoder has no network surface of its own; `deploy.sh` runs as root on your board and fetches
from GitHub over HTTPS, verifying SHA-256 against a list fetched the same way — it does **not** verify
a signature, so it trusts GitHub. If you need more than that, download the assets on a machine you
trust, check them against `SHA256SUMS` yourself and place them in `/opt/aleph-decoder` before running;
`deploy.sh` uses local copies when present.
