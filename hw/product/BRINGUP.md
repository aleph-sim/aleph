# Board bring-up — before `deploy.sh`

`deploy.sh` starts from a KV260 that already runs Kria-PYNQ. Getting to *that* is several steps it cannot
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

Get *Certified Ubuntu for Xilinx Devices* for the KV260 from <https://ubuntu.com/download/amd-xilinx> —
pick the **22.04 LTS** entry for the *Kria KV260*, not the 24.04 one at the top; the file is named
`iot-limerick-kria-classic-desktop-2204-*.img.xz` (as of 2026-08-27 the direct link is
<https://people.canonical.com/~platform/images/xilinx/kria-ubuntu-22.04/iot-limerick-kria-classic-desktop-2204-20240304-165.img.xz>). Write it to the card with Raspberry Pi Imager,
balenaEtcher or `dd`. Default login is `ubuntu` / `ubuntu`.

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

## 3. Wait out the first-boot upgrade

A freshly flashed image starts `unattended-upgrades`, which on 22.04 pulls roughly **200 packages** —
`dpkg`, `systemd`, `perl` and `libc-bin` among them — and on an SD card that takes **hours**. It holds
the dpkg lock the entire time, so every `apt` command you try meanwhile simply refuses to run, and the
refusal reads like a broken installer rather than a busy machine.

```bash
# finished when this prints nothing
pgrep -af /usr/bin/unattended-upgrade
```

**Do not kill it.** It is upgrading dpkg and systemd themselves; interrupting it mid-transaction is how
you get a system that cannot install anything again. If you must script around it, let apt do the
waiting — `apt-get -o DPkg::Lock::Timeout=3600 install ...` — rather than polling for the lock
yourself.

## 4. Install the build toolchain **before** Kria-PYNQ

```bash
sudo apt update
sudo apt install -y build-essential python3-dev portaudio19-dev libcairo2-dev pkg-config
sudo apt install -y libboost-dev
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

> **The second trap, further in.** With the compiler present the build gets as far as **PYNQ itself**
> and then dies on a missing Boost header:
>
> ```
> displayport.cpp:19:10: fatal error: boost/scope_exit.hpp: No such file or directory
> error: command '/usr/bin/make' failed with exit code 2
> ```
>
> `libboost-dev` fixes it, and it is a **separate `apt install` line above on purpose**: apt is
> transactional, so one unsatisfiable package fails the whole command and takes the packages you
> actually needed down with it. We lost a cycle to exactly that — bundling a speculative `libdrm-dev`
> alongside `libboost-dev` produced `pkgProblemResolver::Resolve generated breaks`, and *nothing*
> installed, including the Boost header PYNQ was waiting for. Install the minimum; add anything else
> one line at a time.

**Sanity check before continuing** — all four must be present:

```bash
command -v gcc && ls /usr/include/portaudio.h /usr/include/boost/scope_exit.hpp && pkg-config --modversion cairo
```

## 5. Install Kria-PYNQ, pinned to v3.0 — under a pip constraint file

```bash
sudo tee /etc/pip-constraints.txt <<'EOF'
numpy<2
wheel<0.45
EOF

git clone https://github.com/Xilinx/Kria-PYNQ.git
cd Kria-PYNQ
git checkout v3.0
sudo env PIP_CONSTRAINT=/etc/pip-constraints.txt bash install.sh -b KV260      # ~25 minutes
```

**`sudo env`, not plain `sudo`** — `sudo` drops environment variables, and without `PIP_CONSTRAINT`
reaching every `pip` call inside `install.sh` neither pin below takes effect.

**Pin the tag.** `main` moves and would give a different PYNQ version. The decoder driver carries a
workaround written specifically against PYNQ 3.0.1 — `pynq.Overlay()` fails on designs with no PL DRAM
banks, so the PL is programmed with `pynq.Bitstream().download()` and the DMA engine driven directly
over MMIO. That has not been tested on any other PYNQ version.

**Why a constraint file, and why these two pins.** Kria-PYNQ's 2022 requirements pin almost nothing,
so `install.sh` resolves against *today's* PyPI, and two of today's packages break it. Both were hit
on the 2026-08-25 reflash, and both were first fixed the wrong way — patched *after* the failure —
before it became clear that only something governing every `pip` invocation of the run can hold:

> **`wheel<0.45`.** PyAudio has no aarch64 wheel and builds from source in a pip *isolated build
> environment*. pip populates that environment with the newest `wheel`, and `wheel` ≥ 0.45 (Nov 2024)
> requires `packaging>=24`; the isolated environment cannot see the venv but *can* see the system
> `/usr/lib/python3/dist-packages`, where jammy ships `packaging 21.3`. Result, before a single line
> compiles:
>
> ```
> pkg_resources.VersionConflict: (packaging 21.3 (/usr/lib/python3/dist-packages), Requirement.parse('packaging>=24.0'))
> error: metadata-generation-failed
> ```
>
> Installing a newer `packaging` into the venv does **nothing** — the build environment never sees the
> venv. (Installing it into the *system* Python works, but reaches across `apt`'s territory.)
> `PIP_CONSTRAINT` is the one pip setting honoured inside isolated build environments, so the pin
> goes there.

> **`numpy<2`.** PYNQ 3.0.1 predates numpy 2, and its `PynqBuffer` sets an attribute on an ndarray
> subclass in a way numpy 2 forbids. An earlier revision of this page said to downgrade numpy *after*
> `install.sh`; that is too late — `install.sh` itself imports numpy partway through and dies with
> numpy's own *"downgrade to 'numpy<2' or try to upgrade the affected module"* banner. And even when
> the installer survives, the first DMA buffer allocation fails:
>
> ```
> AttributeError: attribute 'device' of 'numpy.ndarray' objects is not writable
> ```
>
> With the constraint in force every step resolves numpy 1.26.4, and nothing later in the run can
> upgrade it back. Verified 2026-08-26: a second wiped card, this procedure, `3.0.1 1.26.4` at the
> check below. (Budget a slow link: `opencv-python` is 50 MB and the board's PyPI download timed out
> once mid-file; re-running the same command resumes from pip's cache. `PIP_DEFAULT_TIMEOUT=300`
> alongside `PIP_CONSTRAINT` helps.)

**If you already ran `install.sh` once without the constraint** (a venv with numpy 2 exists), repair it
before re-running — the installer does not recreate the venv:

```bash
sudo /usr/local/share/pynq-venv/bin/python3 -m pip install "numpy<2" "packaging>=24"
```

Verify before continuing — **from `/`, not from the clone**, because `~/Kria-PYNQ` contains a directory
named `pynq` that shadows the real module and makes a failed install look half-working:

```bash
cd / && /usr/local/share/pynq-venv/bin/python3 -c \
  "import pynq, numpy, importlib.metadata as m; print(m.version('pynq'), numpy.__version__)"
# expect: 3.0.1 1.26.4
```

### `install.sh` exits non-zero on `pynq_helloworld` — that is expected

On every run so far the installer's **last** step fails like this, *after* PYNQ itself is installed
correctly:

```
KV260 notebooks
Collecting pynq_helloworld
  Using cached pynq_helloworld-3.0.0.tar.gz (4.1 MB)
  Preparing metadata (pyproject.toml) ... error
      ...
      error: invalid command 'bdist_wheel'
error: metadata-generation-failed
```

`pynq_helloworld` is a demo-notebook package with nothing to do with the decoder. If the version check
above prints `3.0.1 1.26.4`, ignore the exit code and go to step 6. (The 2026-08-27 stranger-mode run
stalled here for want of this paragraph: the error text was not quoted, so it did not look like "the
known one".) If the version check fails, read the tail of the log — something *earlier* broke.

The installer is also **interactive** — `jupyter_core` asks whether to overwrite the default notebook
config. Run it in a terminal and answer. If you must run it detached, feed it answers (`yes | sudo env
PIP_CONSTRAINT=/etc/pip-constraints.txt bash install.sh -b KV260`); with no stdin at all it dies with
`EOFError`.

## 6. Now run `deploy.sh`

`deploy.sh` is not on the board yet — it lives in this repository at `hw/product/deploy.sh`. Fetch it
and run it with `bash` (`curl` does not set the execute bit, so `./deploy.sh` says *command not found*):

```bash
curl -fsSLO https://raw.githubusercontent.com/aleph-sim/aleph/main/hw/product/deploy.sh
sudo bash deploy.sh
```

It fetches the five release artefacts into `/opt/aleph-decoder`, verifies their SHA-256, programs the
PL and must end with `CORRECTNESS: PASS (40/40 ...)`.

-----

## If you are reproducing on a second card

Keep the working card. Do not wipe it — flash a *new* card and swap. The old card is a complete,
known-good system and the cheapest possible rollback; the boot firmware it depends on is in the board,
not on it, so nothing is lost by setting it aside.
