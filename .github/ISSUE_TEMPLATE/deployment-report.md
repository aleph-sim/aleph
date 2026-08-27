---
name: Deployment report
about: You ran deploy.sh on a KV260 — tell us how it went, pass or fail. Every report is counted.
title: "[deploy] appliance-v1 on <your board> — PASS / FAIL"
labels: appliance, deployment-report
---

<!-- Two minutes. Successful runs matter as much as failures: the count of external deployments is
     what decides whether v2/v3 gets built (hw/product/README.md). -->

**Result:** PASS / FAIL (last line of `deploy.sh`)

**Had you seen this repository before?** yes / no  <!-- "no" is the answer we are looking for -->

**Board / image**
```
<paste: lsb_release -ds; uname -r>
<paste: cd / && /usr/local/share/pynq-venv/bin/python3 -c "import pynq, numpy, importlib.metadata as m; print(m.version('pynq'), numpy.__version__)">
<paste: tr -d '\0' < /sys/firmware/devicetree/base/chosen/*version*>
```

**Throughput line from `deploy.sh`** (if PASS)
```
best  ... s  -> ... exp/s
```

**Where you had to guess, ask or search** — anything not answered by `BRINGUP.md` / `README.md`.
Each of these is a documentation bug; list them even if you got through:

1.
2.

**If FAIL** — attach `/opt/aleph-decoder/selftest.log` and the full `deploy.sh` output.
