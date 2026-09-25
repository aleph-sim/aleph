#!/usr/bin/env bash
# Decoder appliance — one command from a board running PYNQ to a verified decoder.
#   v1: Kria KV260 + Kria-PYNQ  -> release appliance-v1, banked 16/48, 2085 cycles worst case
#   v2: ZCU104 + PYNQ           -> release appliance-v2, banked 36/144, 1036 cycles = 8.29 us at 125 MHz
# The board is detected from the device tree; RELEASE / BITSTREAM override the choice.
# (v1's 15.64 us is the AXI-Lite overlay at 133.332 MHz; the v1 DMA image this script deploys runs its
# 2085 cycles at ~90.9 MHz. The driver prints the latency this board really delivers.)
#
# Run it ON THE BOARD, as root:
#     sudo ./deploy.sh
#
# What it does, in order, stopping at the first failure:
#   1. checks the board (KV260 or ZCU104), PYNQ and XRT
#   2. fetches the published bitstream, self-test vectors and driver (or uses local copies)
#   3. verifies every artefact against the published SHA-256 list
#   4. programs the PL, checks the PL clock is the one timing closed at, and decodes the 40-shot
#      golden, requiring 40/40 bit-exact
#   5. reports the measured decode latency
#
# It installs nothing outside $INSTALL_DIR and changes no system state except programming the PL.
# Re-running it is safe.
#
# Everything here is Apache-2.0 (hw/LICENSE). Issues: https://github.com/aleph-sim/aleph/issues

set -euo pipefail

REPO="${REPO:-aleph-sim/aleph}"
INSTALL_DIR="${INSTALL_DIR:-/opt/aleph-decoder}"
PYNQ_PY="${PYNQ_PY:-/usr/local/share/pynq-venv/bin/python3}"
BASE_ADDR="${BASE_ADDR:-0xA0000000}"

PRIOR="0.003"

VECTORS="bp_circ_vectors.txt"
DRIVER="bp_stream_banked_kv260.py"
SUMS="SHA256SUMS"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32mok\033[0m  %s\n' "$*"; }
die()  { printf '\n\033[31mFAILED:\033[0m %s\n\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. preflight

say "Checking the board"

[ "$(id -u)" -eq 0 ] || die "run as root: sudo $0"

case "$(uname -m)" in
  aarch64) ok "aarch64" ;;
  *) die "this is a $(uname -m) machine. The appliance is a Kria KV260 (aarch64) image.
       You are probably running this on your laptop instead of on the board." ;;
esac

MODEL=""
[ -r /proc/device-tree/model ] && MODEL=$(tr -d '\0' < /proc/device-tree/model)
# Each board gets the build made for its part AND its PS clocks. PL_MHZ is the clock the bitstream
# closed timing at; the driver reads back what PYNQ really programmed and refuses to decode if it is
# faster, because PYNQ applies the .hwh's PL0 divisors to the *board's* IOPLL (a build made for another
# board's clocks runs at another frequency with no error anywhere). WORST is in cycles because that
# is what the bitstream fixes; the driver prints the microseconds at the clock this board really runs.
case "$MODEL" in
  *ZCU104*)
    BOARD="ZCU104"; DEF_RELEASE="appliance-v2"; DEF_BIT="bp_zcu104_stream_banked_p003.bit"
    GEOMETRY="36/144 banked, gross bivariate-bicycle [[144,12,12]]"
    WORST="1036 cycles = 8.29 us at 125 MHz"; PL_MHZ="125" ;;
  *KV260*|*Kria*)
    BOARD="KV260"; DEF_RELEASE="appliance-v1"; DEF_BIT="bp_kv260_stream_banked_p003.bit"
    GEOMETRY="16/48 banked, gross bivariate-bicycle [[144,12,12]]"
    # v1 closed timing at 96.969 MHz and deploys have measured ~90.9 MHz, but the clock guard has not
    # yet been run on a KV260, so it only reports here; set PL_MHZ=96.969 once it has.
    WORST="2085 cycles (this image clocks ~90.9 MHz, ~22.9 us)"; PL_MHZ="" ;;
  *)
    [ -n "${BITSTREAM:-}" ] || die "board reports \"${MODEL:-unknown}\"; the appliance ships builds for a
       Kria KV260 (appliance-v1) and a ZCU104 (appliance-v2) only. A bitstream is tied to its FPGA part
       and to the board's PS clocks, so there is no generic fallback. If you know what you are doing,
       set RELEASE=... and BITSTREAM=... explicitly."
    BOARD="${MODEL:-unknown}"; DEF_RELEASE="appliance-v1"; DEF_BIT="$BITSTREAM"
    GEOMETRY="(set by you)"; WORST="(unknown for this board)"; PL_MHZ="" ;;
esac
ok "board: ${MODEL:-unknown} -> $BOARD"

RELEASE="${RELEASE:-$DEF_RELEASE}"
# The bitstream bakes the Tanner graph AND the noise prior it was built for. Decoding traffic from a
# different p against a golden built at another p looks exactly like an RTL bug -- it cost this project
# a full debugging campaign (issue #478) before the mismatch was traced to the prior, not the core.
# So the prior travels in the filename and is asserted here rather than left implicit.
BITSTREAM="${BITSTREAM:-$DEF_BIT}"
# PYNQ builds its IP map from the hardware-handoff file, and finds it by basename next to the .bit.
# Without it the overlay loads but the DMA is invisible, the driver falls back to a hardcoded base,
# and every DMA status register reads 0x00000000 -- which looks like dead hardware, not a missing file.
HWH="${BITSTREAM%.bit}.hwh"
ok "release: $RELEASE, bitstream: $BITSTREAM"

[ -x "$PYNQ_PY" ] || die "no PYNQ python at $PYNQ_PY.
       This board is not brought up yet. See hw/product/BRINGUP.md -- six steps, most of them traps
       the upstream instructions do not warn about (Ubuntu 22.04 not 24.04; a multi-hour first-boot
       upgrade that holds the dpkg lock; no C compiler and no Boost headers in the stock image;
       pin Kria-PYNQ to v3.0).
       If your PYNQ lives elsewhere, set PYNQ_PY=/path/to/python3."
ok "pynq python: $PYNQ_PY"

"$PYNQ_PY" - <<'PY' >/dev/null 2>&1 || die "the PYNQ python cannot import pynq/numpy. Reinstall Kria-PYNQ."
import pynq, numpy  # noqa: F401
PY
ok "pynq and numpy import"

if [ "$BOARD" = "KV260" ]; then
  [ -d /usr/lib/firmware/xilinx ] || printf '    \033[33mwarn\033[0m /usr/lib/firmware/xilinx missing; PL programming may fail.\n'
fi
ok "XRT expected at XILINX_XRT=/usr"

for t in curl sha256sum; do
  command -v "$t" >/dev/null || die "missing required tool: $t"
done

# ---------------------------------------------------------------- 2. artefacts

say "Fetching artefacts into $INSTALL_DIR"

mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

fetch() {
  local f="$1"
  if [ -f "$f" ]; then ok "$f (already present)"; return; fi
  local url="https://github.com/$REPO/releases/download/$RELEASE/$f"
  curl -fsSL --retry 3 -o "$f.part" "$url" \
    || die "could not download $f from $url
       If this board has no internet, copy the four release files into $INSTALL_DIR by hand
       and re-run: $BITSTREAM $HWH $VECTORS $DRIVER $SUMS"
  mv "$f.part" "$f"
  ok "$f (downloaded)"
}

fetch "$SUMS"
fetch "$BITSTREAM"
fetch "$HWH"
fetch "$VECTORS"
fetch "$DRIVER"

# ---------------------------------------------------------------- 3. integrity

say "Verifying checksums"

# Check only the files we actually fetched: the release may carry more than the self-test needs.
for f in "$BITSTREAM" "$HWH" "$VECTORS" "$DRIVER"; do
  grep -F " $f" "$SUMS" > ".sum.$f" 2>/dev/null || die "$f is not listed in $SUMS.
       Either the release is malformed or \$BITSTREAM does not match this release."
  sha256sum -c ".sum.$f" >/dev/null 2>&1 \
    || die "$f fails its published SHA-256.
       Do not run it. Delete $INSTALL_DIR/$f and re-run to fetch a clean copy."
  rm -f ".sum.$f"
  ok "$f"
done

# ---------------------------------------------------------------- 4. self-test

say "Programming the PL and decoding the golden"

printf '    geometry: %s\n    noise prior baked into this bitstream: p = %s\n\n' "$GEOMETRY" "$PRIOR"

LOG="$INSTALL_DIR/selftest.log"
set +e
# -u: stdout is a pipe into tee, so Python would block-buffer and the self-test would sit silent
# for several minutes before printing anything at once. Silence here is indistinguishable from a hang.
EXPECT=()
[ -n "$PL_MHZ" ] && EXPECT=(--max-mhz "$PL_MHZ")
env XILINX_XRT=/usr "$PYNQ_PY" -u "$DRIVER" "$BITSTREAM" "$VECTORS" --base "$BASE_ADDR" ${EXPECT[@]+"${EXPECT[@]}"} 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

[ "$rc" -eq 0 ] || die "the self-test driver exited $rc. Full output in $LOG.
       If it failed while programming the PL, the usual causes are a bitstream for another board or
       a PYNQ version other than 3.0.1. If it failed on the PL clock check, the board's PS clocks
       differ from the ones this bitstream was built for -- report it with $LOG attached."

grep -qE '40/40|PASS' "$LOG" \
  || die "the decoder ran but did not report 40/40 bit-exact against the golden.
       This is the correctness gate and it is not negotiable -- do not use this deployment.
       Attach $LOG to an issue at https://github.com/$REPO/issues"

ok "40/40 bit-exact against the software golden"

# ---------------------------------------------------------------- 5. report

say "Deployed"

grep -iE 'latency|us/|experiments/sec|throughput' "$LOG" | sed 's/^/    /' || true

cat <<EOF

    The decoder is programmed and verified on this board.

    What you have:      $GEOMETRY on a $BOARD, worst case $WORST
    Host interface:     AXI4-Lite and batched AXI-DMA -- register map and bit order in
                        hw/product/interface-spec.md. READ THE BIT-ORDER SECTION.
    valid_flag:         a converged decode is almost never wrong. If your architecture can
                        act on a "decode failed" herald, that is where the value is.
                        See docs/qec/q7-07-nonconvergence-policy.md.

    This bitstream decodes ONE code at ONE noise prior (p = $PRIOR). A different code or a
    materially different p needs a different build -- the Tanner graph is baked in.

    Self-test log: $LOG

EOF
