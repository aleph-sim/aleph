"""Generate the committed VQE Hamiltonian data (single source of truth shared by
aleph and Qiskit).

- n=4: the real H2/STO-3G/Jordan-Wigner Hamiltonian at 0.7414 A, from
  OpenFermion's bundled molecular data (authoritative; no PySCF). Also writes the
  exact FCI energy (min eigenvalue) to vqe_n4.fci, and asserts it is ~ -1.137 Ha
  so a broken Hamiltonian is caught at generation time.
- n=6,8: a deterministic molecular-style scalable Pauli sum for the energy-eval
  performance points (convergence is NOT asserted on these).

Run from repo root with the qiskit venv that has openfermion installed:
    scripts/qiskit-baseline/.venv/bin/python scripts/vqe/gen_hamiltonians.py
"""
import math
from pathlib import Path

OUT = Path(__file__).parent / "hamiltonians"


def pauli_str(term, n):
    chars = ["I"] * n
    for q, p in term:
        chars[q] = p
    return "".join(chars)


def h2_4q_terms():
    """Real H2/STO-3G/JW Hamiltonian (4 qubits) from OpenFermion bundled data.

    API adaptation notes (openfermion 1.7.1, 2026-06):
    - MolecularData.load() resolves `self.filename` + '.hdf5'.  The default
      filename is built from the molecule name WITHOUT the bond-length suffix,
      so MolecularData(geometry, ...) -> filename='.../H2_sto-3g_singlet' which
      does not exist.  The bundled file is actually named
      H2_sto-3g_singlet_0.7414.hdf5 under openfermion/testing/data/.
    - Fix: pass filename= explicitly pointing at the bundled 0.7414 file.
      Determined by inspecting openfermion.__file__ so the path works in any
      venv, not just this one.
    - Fallback: if the bundled file is still not found (different install
      layout), try openfermionpyscf; raise a clear error if that is also absent.
    - qop.terms is a dict: { tuple_of_(int,str)_pairs : complex_coeff }.
      The identity term has key () (empty tuple).
    """
    import os
    try:
        # Preferred: top-level re-exports (openfermion >= 1.0)
        from openfermion import MolecularData, jordan_wigner, get_fermion_operator
        import openfermion as _of
    except ImportError:
        from openfermion.chem import MolecularData
        from openfermion.transforms import jordan_wigner, get_fermion_operator
        import openfermion as _of

    geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]

    # Locate the bundled HDF5 data directory relative to the openfermion package.
    of_dir = os.path.dirname(_of.__file__)
    bundled = os.path.join(of_dir, "testing", "data", "H2_sto-3g_singlet_0.7414")
    if os.path.isfile(bundled + ".hdf5"):
        # Happy path: point MolecularData directly at the bundled file.
        mol = MolecularData(geometry, "sto-3g", multiplicity=1, charge=0,
                            filename=bundled)
    else:
        # Bundled file absent (unusual install layout) — construct without filename
        # and let load() try its default path, then fall back to PySCF.
        mol = MolecularData(geometry, "sto-3g", multiplicity=1, charge=0)

    try:
        mol.load()
    except Exception as exc:
        # Bundled data not found — fall back to PySCF
        print(f"[gen] mol.load() failed ({exc}); trying openfermionpyscf ...")
        try:
            from openfermionpyscf import run_pyscf
            mol = run_pyscf(mol, run_scf=True, run_fci=False)
        except ImportError:
            raise RuntimeError(
                "OpenFermion bundled H2 data not found and openfermionpyscf is not "
                "installed. Install it with: pip install openfermionpyscf"
            ) from exc

    qop = jordan_wigner(get_fermion_operator(mol.get_molecular_hamiltonian()))
    out = []
    for ops, coeff in qop.terms.items():
        c = coeff.real
        if abs(c) < 1e-12:
            continue
        # ops is a tuple of (qubit_index, pauli_char) pairs; identity is ()
        out.append((c, list(ops)))
    # Canonical, version-independent order: by (weight, qubit/pauli pattern) so
    # re-running with any OpenFermion version produces a byte-identical file.
    out.sort(key=lambda t: (len(t[1]), [(q, p) for q, p in t[1]]))
    return out, 4


def model_terms(n):
    """Deterministic molecular-style scalable Pauli sum (perf points only):
    H = sum_i h_i Z_i + sum_{i<j} J_ij Z_iZ_j + sum_{i<j} K_ij (X_iX_j + Y_iY_j).
    Bounded deterministic coefficients; O(n^2) terms."""
    out = []
    for i in range(n):
        out.append((0.5 * math.cos(i + 1), [(i, "Z")]))
    for i in range(n):
        for j in range(i + 1, n):
            out.append((0.25 * math.cos(i + j + 1), [(i, "Z"), (j, "Z")]))
            k = 0.1 * math.sin(i * j + 1)
            out.append((k, [(i, "X"), (j, "X")]))
            out.append((k, [(i, "Y"), (j, "Y")]))
    return out, n


def fci_energy(terms, n):
    """Exact ground-state energy = min eigenvalue of the dense 2^n x 2^n matrix."""
    import numpy as np
    P = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    H = np.zeros((2**n, 2**n), dtype=complex)
    for coeff, term in terms:
        ops = ["I"] * n
        for q, p in term:
            ops[q] = p
        m = np.array([[1.0 + 0j]])
        for ch in ops:
            m = np.kron(m, P[ch])
        H += coeff * m
    return float(np.linalg.eigvalsh(H)[0])


def write_ham(path, terms, n):
    lines = [f"# {path.name}: {len(terms)} terms, {n} qubits"]
    for coeff, term in terms:
        lines.append(f"{coeff:.12f} {pauli_str(term, n)}")
    path.write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h2, n4 = h2_4q_terms()
    write_ham(OUT / "vqe_n4.txt", h2, n4)
    fci = fci_energy(h2, n4)
    assert -1.20 < fci < -1.05, f"H2 FCI {fci} not ~ -1.137; check Hamiltonian"
    (OUT / "vqe_n4.fci").write_text(f"{fci:.12f}\n")
    print(f"[gen] vqe_n4: {len(h2)} terms, FCI={fci:.6f} Ha")
    for n in (6, 8):
        terms, _ = model_terms(n)
        write_ham(OUT / f"vqe_n{n}.txt", terms, n)
        print(f"[gen] vqe_n{n}: {len(terms)} terms")


if __name__ == "__main__":
    main()
