#!/usr/bin/env python
"""
ablate_subset.py
================

Does the ablation result GENERALISE beyond nitromethane C-N?

Finding 19 removed 91-97% of the seam by scaling one global-label module, on ONE
bond in ONE molecule. That is a single measurement. This script repeats it across
the 7-bond rigid subset, under matched conditions, with the sham carried through.

    seam collapses on all 7          -> the mechanism is general. Much stronger.
    seam collapses on some           -> tells us WHICH chemistry is vulnerable,
                                        which is a finding in its own right.
    lambda does not transfer         -> also worth knowing, and better found here
                                        than by a referee.

⚠️⚠️ READ THIS BEFORE QUOTING ANY NUMBER FROM THIS SCRIPT ⚠️⚠️
--------------------------------------------------------------
**ABLATION IS A DIAGNOSTIC, NOT A REMEDIATION. IT DOES NOT FIX THE MODEL.**

Scaling the label embedding down does two things at once:

    1. it removes the seam                        <- what we want to measure
    2. it shifts the whole asymptote              <- collateral, unavoidable

In the nitromethane mean-row run the plateau moved by -1.208 eV. The seam went
away and the BDE got WORSE. That is expected: we deliberately destroyed the
model's ability to know which electronic state it was asked about.

So this script reports BOTH, side by side, and refuses to let the second one hide:

    seam_kcal   -- internal consistency. Does the curve still have a step?
    bde_kcal    -- accuracy against experiment. Is the answer right?

    seam -> 0 and bde still wrong   =  EXPECTED. Diagnosis confirmed, no fix.
    seam -> 0 and bde -> experiment =  would be remarkable. Do not assume it.

If anyone reads a table from this script as "we fixed UMA", the fix is
`--fragment-route`, not this. Say so out loud in the write-up.

METHOD NOTE -- the geometry is optimised ONCE, with the INTACT model, and reused
for every condition. Re-optimising under ablation would conflate a geometry
change with an energy change, and the intact geometry is the physically correct
one anyway.

Usage
-----
    conda activate uma  && python ablate_subset.py --model uma
    conda activate mace && python ablate_subset.py --model omol
    python ablate_subset.py --model uma --only Nitromethane
    python ablate_subset.py --model uma --conditions "composition:0.9,sham:0.9"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.optimize import BFGS

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "MLIP-Benchmark-BDE-Heatmap"))

from ablate_embedding import (load_uma, load_omol, ablate,      # noqa: E402
                              sham_candidates)
from rigid_subset import SUBSET                                  # noqa: E402

EV_TO_KCAL = 23.0605

# the hyper-extended grid: the seam sits near 5.8 A, so the scan MUST run past it
R_VALUES = [1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75,
            1.8, 1.85, 1.9, 1.95, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8,
            2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1,
            4.2, 4.3, 4.4, 4.5, 4.7, 4.9, 5.1, 5.3, 5.5, 5.7, 5.9, 6.1, 6.3,
            6.5, 6.7, 7.0]

# module regexes, per model. UMA has three global channels; MACE-OMOL has one.
TARGETS = {
    "uma": {
        "spin":        r"backbone\.spin_embedding$",
        "composition": r"backbone\.composition_embedding$",
        "routing":     r"backbone\.routing_mlp$",
        "chargespin":  r"backbone\.(spin|charge)_embedding$",
    },
    "omol": {
        "spin":        r"joint_embedding\.embedders\.total_spin$",
        "chargespin":  r"joint_embedding\.embedders\.total_(spin|charge)$",
    },
}

# best rung from the nitromethane C-N ladder (Finding 19). Whether these transfer
# to other bonds is precisely what this script tests -- do not assume they do.
DEFAULT_CONDITIONS = {
    "uma":  "composition:0.9,spin:0.25,routing:0.75,sham:0.9",
    "omol": "spin:0.9,chargespin:0.75,sham:0.9",
}


def build_from_smiles(smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=42)      # same seed as bde_scan.py
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    return Atoms(symbols=[a.GetSymbol() for a in mol.GetAtoms()],
                 positions=np.array([[conf.GetAtomPosition(i).x,
                                      conf.GetAtomPosition(i).y,
                                      conf.GetAtomPosition(i).z]
                                     for i in range(mol.GetNumAtoms())]))


def rigid_scan(calc, atoms0, ia, ib, charge=0, spin=1):
    """Rigid scan from a FIXED starting geometry. No optimisation, no constraints."""
    from bde_scan import stretch_any_bond
    E = np.full(len(R_VALUES), np.nan)
    for i, r in enumerate(R_VALUES):
        a = atoms0.copy()
        stretch_any_bond(a, ia, ib, r)
        a.info = {"charge": int(charge), "spin": int(spin)}
        a.calc = calc
        try:
            E[i] = a.get_potential_energy()
        except Exception as exc:
            print(f"      r={r:.2f} failed: {type(exc).__name__}")
    return E


def metrics(E, E_intact=None):
    """Step size, its location, and collateral damage. All kcal/mol.

    >>> FIXED 2026-08-05, AND THE FIRST VERSION WAS BADLY WRONG.
    It computed  seam = E(plateau) - E(5.0 A),  hard-coding the assumption that
    the curve has already flattened by 5.0 A. That is true for nitromethane C-N.
    It is FALSE for ethyl hydroperoxide O-O, whose rigid curve peaks at 4.5 A and
    then DESCENDS -- so 5.0 A sits on a downward slope and the formula returned
    -30.4 kcal/mol, a number that looks like a huge seam and is not a seam at all.

    The seam is a STEP: energy that appears after the curve should already have
    gone flat. So measure it as a step, and locate it:

        rise      E(plateau) - min(E) over r >= 4.0 A. Zero if the curve never
                  climbs after flattening -- which is the correct answer for a
                  curve that only descends.
        drop      max(E) - E(plateau) over r >= 4.0 A. A downward discontinuity
                  is the SAME cutoff artifact with the opposite sign, and it was
                  invisible to the old metric. Reported separately, not folded in.
        step_r    where the largest single-interval jump beyond 4.0 A occurs.
                  If this coincides with the cutoff radius, that is the evidence.
        step_size that jump, signed.
    """
    r = np.array(R_VALUES)
    rel = (E - np.nanmin(E)) * EV_TO_KCAL
    far = r >= 4.0
    rise = float(rel[-1] - np.nanmin(rel[far]))
    drop = float(np.nanmax(rel[far]) - rel[-1])
    jumps = np.diff(rel)
    sel = np.where(r[:-1] >= 4.0)[0]
    k = sel[int(np.nanargmax(np.abs(jumps[sel])))]
    step_r, step_size = float(r[k + 1]), float(jumps[k])

    bde = float(rel[-1])          # rigid; NOT comparable to experiment (see below)
    bonded = np.nan
    if E_intact is not None:
        a = (E_intact - np.nanmin(E_intact)) * EV_TO_KCAL
        m = (r >= 1.2) & (r <= 3.0)
        bonded = float(np.nanmax(np.abs(a[m] - rel[m])))
    return dict(rise=rise, drop=drop, step_r=step_r, step_size=step_size,
                bde=bde, bonded=bonded)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["uma", "omol"], required=True)
    ap.add_argument("--uma-ckpt", default="uma-s-1p1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--only", default=None,
                    help="substring filter on molecule name or bond")
    ap.add_argument("--conditions", default=None,
                    help="comma list of target:lambda, e.g. 'composition:0.9,sham:0.9'")
    ap.add_argument("--sham-seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    conds = args.conditions or DEFAULT_CONDITIONS[args.model]
    conds = [(c.split(":")[0], float(c.split(":")[1])) for c in conds.split(",")]
    out = args.out or f"MLIP_paper/ablate_subset_{args.model}.csv"

    todo = SUBSET
    if args.only:
        q = args.only.lower()
        todo = [s for s in SUBSET if q in s[0].lower() or q in s[3].lower()]
    if not todo:
        sys.exit(f"nothing matches --only {args.only!r}")

    print(f"loading {args.model} ...")
    if args.model == "uma":
        calc, model = load_uma(args.device, args.dtype, args.uma_ckpt)
    else:
        calc, model = load_omol(args.device, args.dtype)

    # resolve the sham once, so every bond gets the SAME sham module -- otherwise
    # the control varies across rows and the comparison is not matched.
    sham_rx = None
    if any(t == "sham" for t, _ in conds):
        picks = sham_candidates(model, k=1, seed=args.sham_seed)
        if picks:
            import re as _re
            sham_rx = f"^{_re.escape(picks[0][0])}$"
            print(f"  sham module: {picks[0][0]}  ({picks[0][1]} params)")
        else:
            print("  !! no sham candidate found -- sham conditions will be skipped")

    rows, curves = [], []
    for name, exp_bde, smiles, bond, _n, idx, _g, _e, _h in todo:
        ia, ib = idx
        print(f"\n{'='*78}\n{name}  {bond}   (exp {exp_bde:.1f} kcal/mol)\n{'='*78}")

        # --- geometry: optimised ONCE with the intact model, reused throughout
        atoms0 = build_from_smiles(smiles)
        atoms0.info = {"charge": 0, "spin": 1}
        atoms0.calc = calc
        try:
            BFGS(atoms0, logfile=None).run(fmax=0.01, steps=400)
        except Exception as exc:
            print(f"  optimisation FAILED ({type(exc).__name__}) -- skipping")
            continue
        atoms0.calc = None

        E0 = rigid_scan(calc, atoms0, ia, ib)
        m0 = metrics(E0)
        hdr = (f"\n{'condition':<16}{'lam':>5}{'RISE':>9}{'vs intact':>11}"
               f"{'drop':>8}{'step@':>8}{'size':>8}{'bonded':>9}{'ratio':>8}")
        print(hdr)
        print("-" * 82)
        print(f"{'intact':<16}{1.0:>5.2f}{m0['rise']:>9.1f}{0.0:>11.1f}"
              f"{m0['drop']:>8.1f}{m0['step_r']:>8.1f}{m0['step_size']:>8.1f}"
              f"{0.0:>9.2f}{'—':>8}")
        if m0["rise"] < 2.0:
            print("   ⚠️  no upward step on this bond -- there is NO SEAM here to "
                  "remove.\n       Ablation having no effect is the correct answer, "
                  "not a failure.")
        rows.append(f"{name},{bond},{exp_bde},intact,1.0,{m0['rise']:.3f},"
                    f"{m0['drop']:.3f},{m0['step_r']:.2f},{m0['step_size']:.3f},"
                    f"{m0['bde']:.3f},0.0,")
        curves += [f"{name},{bond},intact,1.0,{R_VALUES[i]:.2f},{E0[i]:.6f}"
                   for i in range(len(R_VALUES))]

        for target, lam in conds:
            rx = sham_rx if target == "sham" else TARGETS[args.model].get(target)
            if rx is None:
                continue
            try:
                handles, _ = ablate(model, rx, scale=lam)
            except SystemExit as exc:
                print(f"  {target}: {exc}")
                continue
            E1 = rigid_scan(calc, atoms0, ia, ib)
            for h in handles:
                h.remove()
            m1 = metrics(E1, E_intact=E0)
            d = m1["rise"] - m0["rise"]
            ratio = abs(d) / m1["bonded"] if m1["bonded"] > 1e-9 else np.inf
            flag = ("  BROKEN" if (not np.isfinite(m1["bonded"])
                                   or m1["bonded"] > 23.0) else "")
            print(f"{target:<16}{lam:>5.2f}{m1['rise']:>9.1f}{d:>+11.1f}"
                  f"{m1['drop']:>8.1f}{m1['step_r']:>8.1f}{m1['step_size']:>8.1f}"
                  f"{m1['bonded']:>9.2f}{ratio:>8.1f}{flag}")
            rows.append(f"{name},{bond},{exp_bde},{target},{lam:g},{m1['rise']:.3f},"
                        f"{m1['drop']:.3f},{m1['step_r']:.2f},{m1['step_size']:.3f},"
                        f"{m1['bde']:.3f},{m1['bonded']:.3f},{ratio:.2f}")
            curves += [f"{name},{bond},{target},{lam:g},{R_VALUES[i]:.2f},{E1[i]:.6f}"
                       for i in range(len(R_VALUES))]

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("molecule,bond,exp_bde_kcal,condition,lambda,rise_kcal,drop_kcal,"
                "step_r_A,step_size_kcal,bde_rigid_kcal,bonded_max_kcal,ratio\n")
        f.write("\n".join(rows) + "\n")
    with open(out.replace(".csv", "_curves.csv"), "w") as f:
        f.write("molecule,bond,condition,lambda,r_A,E_eV\n")
        f.write("\n".join(curves) + "\n")
    print(f"\nwrote {out}  ({len(rows)} rows)")
    print(f"wrote {out.replace('.csv', '_curves.csv')}  ({len(curves)} rows)")

    # ---- the summary that matters ------------------------------------------
    # >>> FIXED 2026-08-05. The first version averaged the SIGNED seam across
    # bonds. With one bond at +36.9 and another at -30.4 that produced +3.2 --
    # a number that means nothing and reads like "almost no seam anywhere".
    # Bonds with no seam are now EXCLUDED from the average rather than diluting
    # it, and the count of included bonds is printed so nothing hides.
    import collections
    P = [r.split(",") for r in rows]
    seam_bonds = {(p[0], p[1]) for p in P
                  if p[3] == "intact" and float(p[5]) >= 2.0}
    noseam = {(p[0], p[1]) for p in P if p[3] == "intact"} - seam_bonds

    print(f"\n{'='*78}\nACROSS BONDS THAT ACTUALLY HAVE A SEAM  "
          f"({len(seam_bonds)} of {len(seam_bonds) + len(noseam)})\n{'='*78}")
    if noseam:
        print("  excluded (no upward step in the intact model -- nothing to remove):")
        for n, b in sorted(noseam):
            print(f"     {n}  {b}")
        print()
    if seam_bonds:
        by = collections.defaultdict(list)
        for p in P:
            if (p[0], p[1]) in seam_bonds:
                by[p[3]].append((float(p[5]), float(p[10]), p[11]))
        base = {k: v for k, v in by.items()}
        intact_rise = np.mean([x[0] for x in base["intact"]])
        print(f"{'condition':<20}{'mean rise':>12}{'% removed':>12}"
              f"{'max bonded':>13}{'mean ratio':>13}")
        print("-" * 78)
        for k, v in by.items():
            mr = np.mean([x[0] for x in v])
            bd = max(x[1] for x in v)
            rr = [float(x[2]) for x in v if x[2]]
            pct = 100 * (1 - mr / intact_rise) if intact_rise else 0.0
            print(f"{k:<20}{mr:>12.1f}{pct:>11.0f}%{bd:>13.2f}"
                  f"{(np.mean(rr) if rr else np.nan):>13.1f}")

    print("\n⚠️  TWO THINGS THIS TABLE DOES NOT SAY:")
    print("  1. bde_rigid_kcal is a RIGID dissociation energy -- the fragments never")
    print("     relax, so it is systematically far above experiment on every row.")
    print("     Do NOT read bde_rigid vs exp_bde as model error. It is protocol.")
    print("  2. rise -> 0 with the rigid BDE unchanged is the EXPECTED result and")
    print("     confirms the diagnosis. It is NOT a fix.")


if __name__ == "__main__":
    main()
