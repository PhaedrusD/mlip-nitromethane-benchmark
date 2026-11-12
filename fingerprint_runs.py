#!/usr/bin/env python
"""
fingerprint_runs.py -- which starting box did each production run ACTUALLY use?
===============================================================================

>>> WHY A SUBMISSION SCRIPT IS NOT EVIDENCE
    nm_ani__crystal_v100.sh on the node says --box nm_box_1p97_clean_crystal.xyz.
    The ani_crystal_eq.traj in the synced folder fingerprints as
    nm_crystal_1p97.xyz. Both statements can be true: a script records what the
    NEXT submission will do, a trajectory records what one submission DID. When
    they disagree, only the trajectory is evidence.

    Getting this wrong is not cosmetic. "Every model starts from an identical
    configuration" is a Methods claim, and the whole liquid comparison between
    ANI-1xnr and MACE-OMOL rests on the two runs sharing a starting structure.

>>> THE FINGERPRINT
    Frame 0 of an equilibration trajectory IS the starting box, so three numbers
    identify it unambiguously:

        n_atoms          224 liquid / 504 crystal
        n_fragments      at a STRICT 1.1 covalent threshold, so a close contact
                         is not mistaken for a bond
        min INTERmolecular distance   the discriminator; the four candidate
                         boxes sit at 1.290 / 1.759 / 1.652 / 1.687 A, which are
                         separated by far more than thermal noise in frame 0

    Frame 0 is exact, not approximate: no dynamics has happened yet, so the
    match should be to the third decimal. Anything looser means the trajectory
    did not start from any candidate box and something else is going on.

>>> RUN THIS ON THE CLUSTER
    It is pure ASE file reading, no model, no GPU, seconds per trajectory.

        python fingerprint_runs.py --root /mfs/io/groups/nielsen/Cary/MLIP

USAGE
    python fingerprint_runs.py --root . --boxes nm_box_1p97.xyz nm_box_1p97_clean.xyz
"""

import argparse
import glob
import os

import numpy as np

# >>> nm_box.xyz IS IN THIS LIST BECAUSE IT WAS ACTUALLY USED.
# It was left out on the reasoning that the study is a 1.97 g/cc study, so the
# ambient box could not be a candidate. off23_run started from it, and the
# omission turned a clean identification into "NO MATCH (closest
# nm_box_1p97_clean.xyz)" -- a result that reads like a corrupt trajectory
# rather than like a run at the wrong density. A candidate list must hold every
# box that EXISTS, not every box that should have been used.
DEFAULT_BOXES = ["nm_box.xyz", "nm_box_1p97.xyz", "nm_box_1p97_clean.xyz",
                 "nm_crystal_1p97.xyz", "nm_box_1p97_clean_crystal.xyz"]

# >>> 1.1, NOT 1.2. At 1.2 the close contacts in nm_box_1p97.xyz are read as
# bonds and 32 molecules become 27 fragments, which would make two identical
# boxes fingerprint differently depending on how clashy they are. 1.1 reads all
# four candidate boxes as intact molecules, so the fragment count carries
# information about the STRUCTURE rather than about the threshold.
BOND_SCALE = 1.1


def fingerprint(atoms):
    from ase.data import covalent_radii
    rc = np.array([covalent_radii[z] for z in atoms.get_atomic_numbers()])
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    A = d < BOND_SCALE * (rc[:, None] + rc[None, :])

    n = len(atoms)
    seen, comps = np.zeros(n, bool), []
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
        comps.append(np.array(comp))

    inter = d.copy()
    for c in comps:
        inter[np.ix_(c, c)] = np.inf
    return dict(n=n, frags=len(comps), min_inter=float(inter.min()))


def match(fp, refs, tol=0.002):
    """Frame 0 has seen no dynamics, so the match must be near exact."""
    best = None
    for name, r in refs.items():
        if r["n"] != fp["n"]:
            continue
        dev = abs(r["min_inter"] - fp["min_inter"])
        if best is None or dev < best[1]:
            best = (name, dev)
    if best is None:
        return "NO BOX OF MATCHING SIZE", np.nan
    return (best[0] if best[1] <= tol else f"NO MATCH (closest {best[0]})"), best[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--boxes", nargs="*", default=DEFAULT_BOXES)
    ap.add_argument("--pattern", default="*eq*.traj",
                    help="equilibration trajectories; frame 0 is the start box")
    a = ap.parse_args()

    from ase.io import read
    from ase.io.trajectory import Trajectory

    refs = {}
    for b in a.boxes:
        p = os.path.join(a.root, b)
        if not os.path.exists(p):
            p = b
        if not os.path.exists(p):
            print(f"  [missing box] {b}")
            continue
        refs[b] = fingerprint(read(p))

    print("\n  CANDIDATE BOXES")
    print(f"  {'box':<34}{'atoms':>7}{'frags':>7}{'minINTER':>10}")
    for k, v in refs.items():
        print(f"  {k:<34}{v['n']:>7}{v['frags']:>7}{v['min_inter']:>10.3f}")

    # equilibration trajectories, wherever they live
    hits = sorted(set(
        glob.glob(os.path.join(a.root, "**", a.pattern), recursive=True)
        + glob.glob(os.path.join(a.root, "**", "equilibration.traj"), recursive=True)
    ))
    if not hits:
        print(f"\n  no trajectories matching {a.pattern} under {a.root}")
        return

    print(f"\n  RUNS  ({len(hits)} trajectories)")
    print(f"  {'trajectory':<52}{'atoms':>7}{'frags':>7}{'minINTER':>10}"
          f"   started from")
    rows = []
    for t in hits:
        try:
            frame0 = Trajectory(t)[0]
        except Exception as e:
            print(f"  {os.path.relpath(t, a.root):<52}  unreadable: {e}")
            continue
        fp = fingerprint(frame0)
        name, dev = match(fp, refs)
        rel = os.path.relpath(t, a.root)
        print(f"  {rel:<52}{fp['n']:>7}{fp['frags']:>7}{fp['min_inter']:>10.3f}"
              f"   {name}")
        rows.append((rel, name, os.path.getmtime(t)))

    # >>> THE ANSWER THE METHODS SECTION NEEDS, STATED PLAINLY.
    # Group by box size so liquid and crystal are judged separately: they can
    # never have started from the same file and it is not a finding that they
    # did not.
    print("\n  VERDICT")
    for size, label in ((224, "liquid"), (504, "crystal")):
        used = {}
        for t in hits:
            try:
                f0 = Trajectory(t)[0]
            except Exception:
                continue
            if len(f0) != size:
                continue
            fp = fingerprint(f0)
            nm, _ = match(fp, refs)
            used.setdefault(nm, []).append(os.path.relpath(t, a.root))
        if not used:
            continue
        if len(used) == 1:
            box = next(iter(used))
            print(f"    {label}: ALL {sum(len(v) for v in used.values())} run(s) "
                  f"started from {box}. The common-starting-point claim holds.")
        else:
            print(f"    {label}: RUNS DISAGREE. The common-starting-point claim "
                  f"is FALSE as written:")
            for k, v in used.items():
                print(f"       {k}")
                for x in v:
                    print(f"          {x}")

    if rows:
        print("\n  (modification times, to spot a rerun that superseded an older run)")
        for rel, name, mt in sorted(rows, key=lambda r: r[2]):
            import datetime
            ts = datetime.datetime.fromtimestamp(mt).strftime("%Y-%m-%d %H:%M")
            print(f"    {ts}   {rel:<48} {name}")


if __name__ == "__main__":
    main()
