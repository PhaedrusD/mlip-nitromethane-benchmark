#!/usr/bin/env python
"""
check_box.py -- is this starting structure actually 32 (or 72) nitromethanes?
==============================================================================

>>> WHY THIS EXISTS
    fingerprint_runs.py answers "which box did this run start from?". It does
    not answer "was that box any good?", and the answer turned out to be no for
    two of the four candidates:

        nm_crystal_1p97.xyz            36 of 72 molecules carry an N-O bond of
                                       0.954 A and a C-N of 1.188 A. Real values
                                       are 1.22 and 1.45-1.49. Used by
                                       off23_run_crystal, osaka_run_crystal and
                                       uma_run_crystal.

        nm_box_1p97_clean_crystal.xyz  bond lengths are fine, but one hydrogen
                                       sits 2.38 A from its own carbon and 1.95 A
                                       from a neighbouring oxygen. The box is 71
                                       nitromethanes, one CH2NO2, and one free H.
                                       Used by ani_run_crystal.

    Neither failure is visible in a fragment count at a loose threshold, in the
    density, or in anything a submission script records. Both put a large amount
    of stored strain, or an already-broken molecule, into frame 0 of a 3000 K
    run -- and the resulting chemistry is then attributed to the model.

>>> WHAT IS CHECKED
    For every molecule in the box, against gas-phase nitromethane:

        C-N   1.489 A        N-O   1.224 A        C-H   1.088 A

    A bond more than --tol away from reference is reported. So is any molecule
    whose atom count is not 7, any atom bonded to nothing, and the closest
    intermolecular contact, which is the separate question fingerprint_runs.py
    already answers but that belongs in the same report.

    Molecules are found at a STRICT 1.15 threshold. A generous one was tried
    first, on the reasoning that the search should reach across a broken bond
    rather than split a damaged molecule into two innocent-looking fragments.
    It backfired: at 1.35 the 1.290 A intermolecular contacts in
    nm_box_1p97.xyz bridge 22 separate molecules into one 154-atom blob, and
    every cross-molecule pair inside it is then audited as if it were a bond.
    The tool cried wolf on a box whose molecules are individually perfect.

    Strict finding plus two explicit checks does the job without that: a
    fragment whose atom count is not 7 catches the damaged molecule, and the
    lone-atom test below catches the piece that came off. Neither depends on
    the fragment search reaching across anything.

USAGE
    python check_box.py nm_box_1p97_clean.xyz
    python check_box.py nm_*.xyz --tol 0.08
"""

import argparse
import sys

import numpy as np

# Gas-phase nitromethane, for reference. A crystal geometry will differ from
# these by a few hundredths; --tol 0.10 passes that and fails the two boxes
# described above by a wide margin.
REF = {("C", "N"): 1.489, ("N", "O"): 1.224, ("C", "H"): 1.088}
MOL_ATOMS = 7
M_NITROMETHANE = 61.04
N_AVOGADRO = 6.02214076e23

# >>> 1.15, STRICT. See the note above: a generous threshold merges molecules
# across close intermolecular contacts and turns a clashy box into one blob.
FIND_SCALE = 1.15


def pair(a, b):
    return REF.get((a, b)) or REF.get((b, a))


def molecules(atoms, d):
    from ase.data import covalent_radii
    rc = np.array([covalent_radii[z] for z in atoms.get_atomic_numbers()])
    A = d < FIND_SCALE * (rc[:, None] + rc[None, :])
    n = len(atoms)
    seen = np.zeros(n, bool)
    out = []
    for i in range(n):
        if seen[i]:
            continue
        stack, comp = [i], []
        seen[i] = True
        while stack:
            k = stack.pop()
            comp.append(k)
            for j in np.where(A[k] & ~seen)[0]:
                seen[j] = True
                stack.append(j)
        out.append(sorted(comp))
    return out


def check(path, tol, quiet):
    from ase.io import read

    atoms = read(path)
    sym = np.array(atoms.get_chemical_symbols())
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)

    mols = molecules(atoms, d)
    n_nm = len(atoms) // MOL_ATOMS
    rho = n_nm * M_NITROMETHANE / N_AVOGADRO / (atoms.get_volume() * 1e-24)

    print(f"\n  {path}")
    print(f"    {len(atoms)} atoms, cell {atoms.cell.lengths()[0]:.3f} A, "
          f"{rho:.3f} g/cc, {len(mols)} fragments "
          f"(expected {n_nm} molecules of {MOL_ATOMS})")

    problems = []

    # >>> A FRAGMENT THAT IS A MULTIPLE OF 7 IS A DIFFERENT FAULT.
    # Two intact molecules fused by one short contact report as a single
    # 14-atom fragment. Saying "fragment of 14 atoms" invites the reader to
    # hunt for a damaged molecule that is not there, so name the contact that
    # joined them instead. In nm_box_1p97.xyz it is an O-O pair at 1.469 A,
    # which is a peroxide bond length between two separate nitromethanes.
    wrong_size = [m for m in mols if len(m) != MOL_ATOMS]
    for m in wrong_size:
        counts = "".join(f"{s}{list(sym[m]).count(s)}"
                         for s in ("C", "H", "N", "O") if list(sym[m]).count(s))
        if len(m) % MOL_ATOMS == 0 and len(m) > MOL_ATOMS:
            from ase.data import covalent_radii
            rc = np.array([covalent_radii[z]
                           for z in atoms.get_atomic_numbers()])
            worst_link = None
            for x, i in enumerate(m):
                for j in m[x + 1:]:
                    if pair(sym[i], sym[j]) is not None:
                        continue      # a real bond type; not the bridge
                    if d[i, j] < 1.15 * (rc[i] + rc[j]):
                        if worst_link is None or d[i, j] < worst_link[0]:
                            worst_link = (d[i, j], i, j)
            link = (f"; joined by {sym[worst_link[1]]}{worst_link[1]}-"
                    f"{sym[worst_link[2]]}{worst_link[2]} at "
                    f"{worst_link[0]:.3f} A" if worst_link else "")
            problems.append(
                f"{len(m) // MOL_ATOMS} molecules fused into one fragment "
                f"({counts}){link}")
        else:
            problems.append(
                f"fragment of {len(m)} atoms ({counts}), atoms "
                f"{[int(x) for x in m[:8]]}")

    # bond-length audit, within molecules only
    worst = {}
    n_bad = 0
    bad_mols = set()
    for mi, m in enumerate(mols):
        for x, i in enumerate(m):
            for j in m[x + 1:]:
                r0 = pair(sym[i], sym[j])
                if r0 is None or d[i, j] > 1.9:
                    continue
                dev = d[i, j] - r0
                key = f"{sym[i]}-{sym[j]}" if sym[i] < sym[j] \
                    else f"{sym[j]}-{sym[i]}"
                if key not in worst or abs(dev) > abs(worst[key][0]):
                    worst[key] = (dev, d[i, j], r0, i, j)
                if abs(dev) > tol:
                    n_bad += 1
                    bad_mols.add(mi)

    for key in sorted(worst):
        dev, r, r0, i, j = worst[key]
        flag = "  <== OUT OF RANGE" if abs(dev) > tol else ""
        print(f"    {key:<5} worst {r:.3f} A (ref {r0:.3f}, "
              f"{dev:+.3f}) atoms {i},{j}{flag}")

    if n_bad:
        problems.append(f"{n_bad} bond(s) more than {tol} A from reference, "
                        f"in {len(bad_mols)} of {len(mols)} fragments")

    # detached atoms: nothing within a real bond length
    lone = [i for i in range(len(atoms))
            if not any(pair(sym[i], sym[j]) and d[i, j] < 1.35
                       for j in range(len(atoms)))]
    for i in lone:
        j = int(np.argmin(d[i]))
        problems.append(f"atom {i} ({sym[i]}) has no bond within 1.35 A; "
                        f"nearest is {sym[j]} at {d[i, j]:.3f} A")

    # closest intermolecular contact
    inter = d.copy()
    for m in mols:
        inter[np.ix_(m, m)] = np.inf
    mi_val = float(inter.min())
    ii, jj = np.unravel_index(np.argmin(inter), inter.shape)
    print(f"    closest intermolecular contact {mi_val:.3f} A "
          f"({sym[ii]}{ii}-{sym[jj]}{jj})")
    if mi_val < 1.5:
        problems.append(f"closest intermolecular contact is {mi_val:.3f} A; "
                        f"models that fail by pairwise collapse will be "
                        f"confounded by this box")

    if problems:
        print(f"    {'!' * 60}")
        for p in problems:
            print(f"    !! {p}")
        print(f"    {'!' * 60}")
    else:
        print("    OK")
    return not problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("boxes", nargs="+")
    ap.add_argument("--tol", type=float, default=0.10,
                    help="allowed deviation from the reference bond length, A")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    ok = [check(b, a.tol, a.quiet) for b in a.boxes]
    print(f"\n  {sum(ok)} of {len(ok)} box(es) clean\n")
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
