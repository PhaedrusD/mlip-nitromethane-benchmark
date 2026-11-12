#!/usr/bin/env python
"""
add_methyl_h.py
===============

Add the three methyl hydrogens to each nitromethane molecule in a heavy-atom-only
crystal structure (C, N, O, O per molecule). Hydrogens are placed GEOMETRICALLY
from the C->N bond direction (tetrahedral carbon, C-H = 1.09 A), so no bond-order
guessing is involved and the nitro oxygens are never protonated.

Any pre-existing H in the input is stripped first, so this also cleans up a file
that obabel mis-hydrogenated (e.g. spurious O-H).

Output is an extxyz with atoms ordered molecule-by-molecule as C,N,O,O,H,H,H
(clean 7-atom blocks), with the cell and pbc preserved, ready for
build_nm_crystal.py / compress_box.py.

Example:
    python add_methyl_h.py --in NM_crystal.cif --out nm_unit_H.extxyz
"""

import argparse
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.neighborlist import natural_cutoffs, NeighborList

CH_BOND = 1.09                       # Angstrom
TET_ANGLE = np.deg2rad(109.47)       # H-C-N angle


def connected_molecules(atoms):
    """Group atom indices into molecules via a covalent-radius neighborlist."""
    cut = natural_cutoffs(atoms, mult=1.2)
    nl = NeighborList(cut, self_interaction=False, bothways=True)
    nl.update(atoms)
    n = len(atoms)
    seen = np.zeros(n, bool)
    mols = []
    for s in range(n):
        if seen[s]:
            continue
        stack, grp = [s], []
        while stack:
            a = stack.pop()
            if seen[a]:
                continue
            seen[a] = True
            grp.append(a)
            nbrs, _ = nl.get_neighbors(a)
            stack.extend(nbrs)
        mols.append(sorted(grp))
    return nl, mols


def methyl_hydrogens(c_pos, n_pos):
    """Three H positions around carbon, pointing away from N."""
    u = (n_pos - c_pos)
    u /= np.linalg.norm(u)
    # arbitrary perpendicular basis
    t = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    v = np.cross(u, t); v /= np.linalg.norm(v)
    w = np.cross(u, v)
    hs = []
    for k in range(3):
        phi = np.deg2rad(120.0 * k)          # arbitrary methyl phase; eq relaxes it
        d = (np.cos(TET_ANGLE) * u
             + np.sin(TET_ANGLE) * (np.cos(phi) * v + np.sin(phi) * w))
        hs.append(c_pos + CH_BOND * d)
    return hs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", default="nm_unit_H.extxyz")
    args = ap.parse_args()

    atoms = read(args.infile)
    atoms.set_pbc([True, True, True])

    # strip any existing hydrogens (cleans up obabel's wrong O-H)
    keep = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s != "H"]
    if len(keep) != len(atoms):
        print(f"Stripped {len(atoms) - len(keep)} pre-existing H atom(s).")
    atoms = atoms[keep]

    cell = atoms.get_cell()
    pos = atoms.get_positions()
    sym = atoms.get_chemical_symbols()

    nl, mols = connected_molecules(atoms)
    print(f"Heavy-atom molecules found: {len(mols)}")

    out_sym, out_pos = [], []
    for grp in mols:
        elems = [sym[i] for i in grp]
        c_local = [i for i in grp if sym[i] == "C"]
        n_local = [i for i in grp if sym[i] == "N"]
        o_local = [i for i in grp if sym[i] == "O"]
        if not (len(c_local) == 1 and len(n_local) == 1 and len(o_local) == 2):
            raise SystemExit(
                f"Molecule {grp} has composition {sorted(elems)} (expected "
                "C1 N1 O2). Check the input structure / bonding.")
        c = c_local[0]
        n = n_local[0]

        # minimum-image C->N vector via neighborlist offsets
        idxs, offs = nl.get_neighbors(c)
        nvec = None
        for j, off in zip(idxs, offs):
            if j == n:
                nvec = pos[n] + off @ cell - pos[c]
                break
        if nvec is None:                       # fallback: direct mic
            d = pos[n] - pos[c]
            nvec = d  # heavy structure is intramolecular-contiguous in practice

        # emit heavy atoms in C, N, O, O order, then the 3 methyl H
        for i in [c, n] + o_local:
            out_sym.append(sym[i]); out_pos.append(pos[i])
        for h in methyl_hydrogens(pos[c], pos[c] + nvec):
            out_sym.append("H"); out_pos.append(h)

    result = Atoms(symbols=out_sym, positions=np.array(out_pos),
                   cell=cell, pbc=[True, True, True])

    from collections import Counter
    comp = Counter(result.get_chemical_symbols())
    print(f"Output composition: {dict(comp)}  ({len(result)} atoms, "
          f"{len(result)//7} molecules)")
    assert comp["H"] == 3 * comp["C"] == 3 * comp["N"] == comp["O"] * 3 // 2, \
        "composition check failed"

    write(args.out, result)
    print(f"Wrote {args.out} (atoms ordered in clean 7-atom blocks)")


if __name__ == "__main__":
    main()
