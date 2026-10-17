#!/usr/bin/env python3
"""Independent Stim forced-mechanism reconstruction and PyMatching fixture replay."""
import itertools
import json
from pathlib import Path
import numpy as np
import stim
import pymatching


def forced_model(text, faults):
    selected = set(faults)
    lines = []
    index = 0
    for line in text.splitlines():
        if line.startswith("error("):
            line = f"error({int(index in selected)})" + line.split(")", 1)[1]
            index += 1
        lines.append(line)
    assert selected.issubset(range(index))
    return stim.DetectorErrorModel("\n".join(lines))


def reconstruct(text, faults):
    model = forced_model(text, faults)
    syndrome, truth, _ = model.compile_sampler(seed=543).sample(shots=1)
    return np.flatnonzero(syndrome[0]).tolist(), truth[0].tolist()


def main():
    directory = Path(__file__).resolve().parents[2] / "crates/aleph-qec/tests/data/witness"
    report = {"stim": stim.__version__, "pymatching": pymatching.__version__, "cases": []}
    for path in sorted(directory.glob("*.json")):
        witness = json.loads(path.read_text())
        model = stim.DetectorErrorModel(witness["dem"])
        matching = (pymatching.Matching.from_detector_error_model(model)
                    if witness["decoder"]["kind"] == "Mwpm" else None)
        for fault_key, outcome_key in [("original_faults", "original_outcome"), ("faults", "outcome")]:
            outcome = witness[outcome_key]
            syndrome, truth = reconstruct(witness["dem"], witness[fault_key])
            assert syndrome == outcome["syndrome"], (path, outcome_key, "syndrome")
            assert truth == outcome["truth"], (path, outcome_key, "truth")
            if matching is not None:
                bits = np.zeros(model.num_detectors, dtype=np.uint8)
                bits[syndrome] = 1
                prediction = matching.decode(bits).astype(bool).tolist()
                assert prediction == outcome["prediction"], (path, outcome_key, "prediction")
                assert (prediction != truth) == outcome["logical_failure"]
            report["cases"].append({"fixture": path.name, "fault_set": fault_key,
                                     "stim_reconstruction": "PASS", "pymatching": "PASS" if matching is not None else "incompatible hypergraph; not run"})
    # Independent XOR semantics for every subset of a correlated, multi-observable tiny DEM.
    text = "error(0.5) D0 L0\nerror(0.5) D0 L1\nerror(0.5) L0\nerror(0.5) D0 L0 ^ D1 L1"
    dets = [[0], [0], [], [0, 1]]
    obs = [[0], [1], [0], [0, 1]]
    for mask in itertools.product([False, True], repeat=4):
        faults = [i for i, fired in enumerate(mask) if fired]
        ds = [False] * 2
        os = [False] * 2
        for i in faults:
            for d in dets[i]:
                ds[d] ^= True
            for o in obs[i]:
                os[o] ^= True
        assert reconstruct(text, faults) == ([i for i, b in enumerate(ds) if b], os)
    report["tiny_subsets"] = 16
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
