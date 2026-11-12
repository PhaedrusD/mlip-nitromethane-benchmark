#!/usr/bin/env python
"""
short_range_floor.py -- is each model's surface bounded below at short range?
==============================================================================

Takes a CLEAN liquid frame, picks one O-H bond, compresses it in place from
its equilibrium length down to 0.05 A, and evaluates the bare model at each
point. Nothing else in the box moves. The answer is a curve per model, in
kcal/mol relative to the unperturbed frame, next to bare Coulomb repulsion.

>>> WHY THIS REPLACES THE 3.7e7 eV NUMBER
    That figure was the potential energy MACE-Osaka24 reported at the collapse
    frame of a run that started from the defective crystal. The model's energy
    at a geometry does not depend on how the geometry was produced, so the
    number was sound -- but it owed its existence to a run the paper no longer
    cites, and it was a single point. This produces the same kind of number
    from a clean-box frame, for all five models, along a controlled coordinate.
    It turns "Osaka24's surface is unbounded below" into a row of five curves
    and tells us which of the others have a floor at all.

>>> WHY ONE O-H IN A FULL BOX, NOT AN ISOLATED DIMER
    The collapses happened in the condensed phase with every neighbour present
    inside the cutoff. A gas-phase O-H scan would test the model in a different
    environment from the one it failed in. The frame is the real box; only two
    atoms move.

>>> ONE MODEL PER PROCESS, ALWAYS
    mace-dox and uma cannot share a shell (LD_LIBRARY_PATH stacking kills
    libtorch). Run this once per --model in its own srun block, then
    --merge the CSVs. Do NOT set KMP_DUPLICATE_LIB_OK.

USAGE (cluster, one block per model)
    python short_range_floor.py --model osaka24 --osaka-path .../mace-osaka24-medium-headfix.model
    python short_range_floor.py --model off23
    python short_range_floor.py --model omol
    python short_range_floor.py --model uma   --uma-path uma-s-1p1
    python short_range_floor.py --model ani   --ani-info .../ani-1xnr.info
    python short_range_floor.py --merge        # prints the table, writes the figure
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

KCAL = 23.0605
COULOMB_EV_A = 14.3996          # e^2 / (4 pi eps0), in eV*Angstrom
LABEL = {"off23": "MACE-OFF23", "osaka24": "MACE-Osaka24", "omol": "MACE-OMOL-0",
         "uma": "UMA-S-1P1", "ani": "ANI-1xnr"}


def pick_oh(atoms):
    """The first intact nitromethane C-H... no: an O-H CONTACT is what
    collapsed, and nitromethane has no O-H bond. So this picks the closest
    O...H intermolecular pair in the frame and compresses THAT, which is the
    coordinate the collapses actually followed."""
    sym = np.array(atoms.get_chemical_symbols())
    D = atoms.get_all_distances(mic=True)
    O = np.where(sym == "O")[0]
    H = np.where(sym == "H")[0]
    sub = D[np.ix_(O, H)]
    i, j = np.unravel_index(np.argmin(sub), sub.shape)
    return int(O[i]), int(H[j]), float(sub[i, j])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(LABEL))
    ap.add_argument("--frame", default="osaka_run_2/production.traj",
                    help="clean-box trajectory to take the frame from")
    ap.add_argument("--index", type=int, default=2000,
                    help="frame index (2000 = 50 ps, well into steady state)")
    ap.add_argument("--rmin", type=float, default=0.05)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--osaka-path"); ap.add_argument("--ani-info")
    ap.add_argument("--uma-path"); ap.add_argument("--omol-path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="MLIP_paper/short_range_floor")
    ap.add_argument("--merge", action="store_true",
                    help="merge per-model CSVs in --outdir into one table + figure")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    if a.merge:
        return merge(a.outdir)
    if not a.model:
        sys.exit("--model is required unless --merge")

    from ase.io.trajectory import Trajectory
    from run_reactive_md import setup_calculator
    atoms = Trajectory(a.frame)[a.index].copy()
    iO, iH, r0 = pick_oh(atoms)
    print(f"  frame {a.index} of {a.frame}: compressing O{iO}-H{iH}, "
          f"starting at {r0:.3f} A")

    calc = setup_calculator(a.model, osaka_path=a.osaka_path, ani_info=a.ani_info,
                            uma_path=a.uma_path, omol_path=a.omol_path,
                            device=a.device)
    atoms.calc = calc
    E0 = atoms.get_potential_energy()

    # move the H along the O->H vector; everything else fixed
    pos0 = atoms.get_positions().copy()
    v = atoms.get_distance(iO, iH, mic=True, vector=True)
    u = v / np.linalg.norm(v)
    rs = np.linspace(r0, a.rmin, a.n)
    rows = []
    for r in rs:
        p = pos0.copy()
        p[iH] = p[iO] + u * r
        atoms.set_positions(p)
        try:
            E = atoms.get_potential_energy()
            dE = (E - E0) * KCAL
            ok = np.isfinite(E)
        except Exception as exc:          # a model that NaNs here is a result too
            dE, ok = float("nan"), False
        coul = COULOMB_EV_A * 8 * 1 / r * KCAL
        rows.append((r, dE, coul, ok))
        print(f"    r = {r:5.3f} A   dE = {dE:+14.1f} kcal/mol   "
              f"bare Coulomb {coul:+12.1f}" + ("" if ok else "   <-- not finite"))

    out = os.path.join(a.outdir, f"floor_{a.model}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["r_A", "dE_kcal", "coulomb_kcal", "finite", "model", "frame",
                    "index", "iO", "iH", "r0"])
        for r, dE, c, ok in rows:
            w.writerow([f"{r:.4f}", f"{dE:.3f}", f"{c:.3f}", int(ok), a.model,
                        a.frame, a.index, iO, iH, f"{r0:.4f}"])
    print(f"  wrote {out}")


def merge(outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import poster_style; poster_style.apply()
    except ModuleNotFoundError:
        pass
    files = sorted(glob.glob(os.path.join(outdir, "floor_*.csv")))
    if not files:
        sys.exit(f"  no floor_*.csv in {outdir}")
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    fig.subplots_adjust(left=0.17, bottom=0.14, right=0.98, top=0.97)
    print(f"\n  {'model':<14}{'min dE (kcal/mol)':>20}{'at r (A)':>10}"
          f"{'turns attractive below':>24}")
    coul_drawn = False
    for f in files:
        rows = list(csv.DictReader(open(f)))
        m = rows[0]["model"]
        r = np.array([float(x["r_A"]) for x in rows])
        dE = np.array([float(x["dE_kcal"]) for x in rows])
        c = np.array([float(x["coulomb_kcal"]) for x in rows])
        if not coul_drawn:
            ax.plot(r, c, "k--", lw=1.4, label="bare Coulomb, O–H")
            coul_drawn = True
        ax.plot(r, dE, lw=2.2, label=LABEL.get(m, m))
        # first r (going inward) where the slope flips to attractive
        ddE = np.diff(dE)                      # r decreasing, so dE should rise
        bad = np.where(ddE < 0)[0]
        turn = r[bad[0] + 1] if len(bad) else float("nan")
        k = int(np.nanargmin(dE))
        print(f"  {LABEL.get(m, m):<14}{np.nanmin(dE):>20.1f}{r[k]:>10.3f}"
              f"{turn:>24.3f}")
    ax.set_xlabel("O–H separation (Å)")
    ax.set_ylabel("ΔE relative to unperturbed frame (kcal mol$^{-1}$)")
    ax.set_yscale("symlog", linthresh=100)
    ax.invert_xaxis()
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    out = os.path.join(outdir, "fig_short_range_floor.png")
    fig.savefig(out, dpi=300)
    print(f"\n  wrote {out}")
    print("  'turns attractive below' is the first separation, compressing "
          "inward, at which\n  the energy stops rising. A surface with a floor "
          "never does; nan means never.")


if __name__ == "__main__":
    main()
