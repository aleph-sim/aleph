# Board bring-up — before `deploy.sh`

`deploy.sh` starts from a KV260 that already runs Kria-PYNQ. Getting to *that* is two steps it cannot
do for you, and both have a trap that the upstream instructions do not mention. Everything here was
found by wiping a card and following the official docs, in that order, and hitting each wall in turn.

**Reference configuration** — what the working development board runs, and what the driver has actually
been validated against:

| | |
|---|---|
| OS | **Ubuntu 22.04.4 LTS (jammy)**, `iot-limerick-kria-classic-desktop-2204-*` |
| kernel | `5.15.0-1027-xilinx-zynqmp` |
| PYNQ | **3.0.1**, from Kria-PYNQ tag **v3.0**, venv at `/usr/local/share/pynq-venv` |
| boot firmware | 2023.01 (in the board's QSPI, not on the card) |

-----

## 1. Flash Ubuntu **22.04**, not 24.04

Get *Certified Ubuntu for Xilinx Devices* for the KV260 and write it to the card with Raspberry Pi
Imager, balenaEtcher or `dd`. Default login is `ubuntu` / `ubuntu`.

> **The trap.** Canonical's download page serves **24.04** by default now, and 24.04 boots fine on the
> KV260 — so the mistake is invisible until much later. But **Kria-PYNQ has exactly two tags, v1.0 and
> v3.0, both from 2022, and both target 22.04.** On 24.04 the system Python is 3.12, so `install.sh`
> fails at `apt install python3.10-venv` with *"Unable to locate package"*, after it has already added
> a jammy PPA to your noble system. There is no supported Kria-PYNQ on 24.04. Use 22.04.

## 2. Check the boot firmware — probably do not update it

The KV260 needs boot firmware **2022.1 or newer** to boot Ubuntu 22.04, and most guides tell you to
update it. Updating QSPI firmware is the one step in this whole procedure that can brick the board, so
check before you touch it:

```bash
tr -d '\0' < /sys/firmware/devicetree/base/chosen/*version* ; echo
```

Anything ≥ 2022.1 means you are done — skip the update. **Firmware lives in the board's QSPI flash, not
on the SD card**, so it survives reflashing and card swaps: if the board has ever booted 22.04, its
firmware is already new enough.

## 3. Install the build toolchain **before** Kria-PYNQ

```bash
sudo apt update
sudo apt install -y build-essential python3-dev portaudio19-dev libcairo2-dev pkg-config
```

> **The trap.** The stock Kria Ubuntu image ships **no C compiler**, and `install.sh` does not install
> one. Most of its Python dependencies have prebuilt aarch64 wheels, but **PyAudio and pycairo do not**
> and must compile from source — roughly twenty minutes into the run, after several hundred megabytes
> of downloads, it dies with:
>
> ```
> error: command 'aarch64-linux-gnu-gcc' failed: No such file or directory
> ERROR: Could not build wheels for PyAudio, pycairo
> ```
>
> `build-essential` supplies the compiler; `portaudio19-dev` and `libcairo2-dev` + `pkg-config` supply
> the headers those two need. Neither package is used by the decoder — they are dependencies of the
> Jupyter demo stack that Kria-PYNQ installs wholesale — but the install is all-or-nothing.

## 4. Install Kria-PYNQ, pinned to v3.0

```bash
git clone https://github.com/Xilinx/Kria-PYNQ.git
cd Kria-PYNQ
git checkout v3.0
sudo bash install.sh -b KV260      # ~25 minutes
```

**Pin the tag.** `main` moves and would give a different PYNQ version. The decoder driver carries a
workaround written specifically against PYNQ 3.0.1 — `pynq.Overlay()` fails on designs with no PL DRAM
banks, so the PL is programmed with `pynq.Bitstream().download()` and the DMA engine driven directly
over MMIO. That has not been tested on any other PYNQ version.

Verify before continuing:

```bash
/usr/local/share/pynq-venv/bin/python3 -c "import pynq; print(pynq.__version__)"   # expect 3.0.1
```

## 5. Now run `deploy.sh`

```bash
sudo ./deploy.sh
```

-----

## If you are reproducing on a second card

Keep the working card. Do not wipe it — flash a *new* card and swap. The old card is a complete,
known-good system and the cheapest possible rollback; the boot firmware it depends on is in the board,
not on it, so nothing is lost by setting it aside.
