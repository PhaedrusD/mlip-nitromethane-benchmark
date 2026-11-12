#!/usr/bin/env python
"""
orca_matrix.py -- DLPNO-CCSD(T) reference for the 17-cell bond x environment
matrix (Figure 2), on the same footing as the 26-bond set
==============================================================================

WHY THIS EXISTS
    bde_matrix_data.py scores the five models against hand-typed experimental
    D(298 K) values of unrecorded origin (EXP_BDE_KCAL in the heatmap repo),
    not against GPAW as draft 3 says. One of them (nitro N-O, 59.0) is wrong
    by ~35 kcal/mol, all of them are enthalpies where the models return De,
    and the N-O split was run with the wrong multiplicities. This puts the
    matrix on the reference the rest of the paper uses:
        DLPNO-CCSD(T)=FULL/aug-cc-pVQZ // wB97X-D3/def2-TZVP
    using the templates and parser from orca_molecules.py unchanged.

WHAT IS ALREADY DONE
    11 of the 25 species exist in orca_dlpno/ (C2H6, C2H5, CH3, H, CH3CH2NH2,
    NH2, NO2, H2CO, HCO, OH, O). Four cells need nothing new. This script
    writes inputs only for species with no finished single point.

MULTIPLICITIES
    Odd electrons -> doublet, even -> singlet, O atom -> triplet (from
    orca_molecules.ATOM_MULT). The nitro N-O split is CH3CH2NO2 -> CH3CH2NO
    (closed-shell singlet) + O (3P). The heatmap repo ran it as [1,2,2].

BOND INDICES
    Match thermodynamic_bde_calculation.py exactly (RDKit SMILES order with
    hydrogens appended parent by parent), so each cell dissociates the same
    bond the models were asked about: every C-H cell is a METHYL C-H.

USAGE
    python orca_matrix.py                      # write inputs for missing species
    python orca_matrix.py --parse              # energies -> matrix_refs.csv + error matrix
    python orca_matrix.py --status             # what exists, what is missing
"""
import argparse, csv, glob, json, os, re, sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import orca_molecules as OM

DLPNO_DIR = "orca_dlpno"            # existing production set
OUT_DIR = "orca_matrix"             # new species go here
HEATMAP = "MLIP-Benchmark-BDE-Heatmap"

# name -> SMILES with explicit radical sites. Names follow orca_dlpno's style.
SPECIES = {
    # parents
    "C2H6":       "CC",
    "CH3CH2OH":   "CCO",
    "CH3CH2NH2":  "CCN",
    "CH3CH2NO2":  "CC[N+]([O-])=O",
    "CH3CH2OOH":  "CCOO",
    "H2CO":       "C=O",
    # fragments
    "H":          "[H]",
    "O":          "[O]",
    "CH3":        "[CH3]",
    "C2H5":       "[CH2]C",
    "OH":         "[OH]",
    "NH2":        "[NH2]",
    "NO2":        "[N+](=O)[O-]",
    "HCO":        "[CH]=O",
    "CH2CH2OH":   "[CH2]CO",
    "CH2OH":      "[CH2]O",
    "CH2CH2NH2":  "[CH2]CN",
    "CH2NH2":     "[CH2]N",
    "CH3CH2NH":   "CC[NH]",
    "CH2CH2NO2":  "[CH2]C[N+]([O-])=O",
    "CH2NO2":     "[CH2][N+]([O-])=O",
    "CH3CH2NO":   "CCN=O",
    "CH2CH2OOH":  "[CH2]COO",
    "CH2OOH":     "[CH2]OO",
    "CH3CH2O":    "CC[O]",
}

# (molecule, bond, environment, fragment A, fragment B), same 17 cells and
# the same bond choice as the heatmap repo.
CELLS = [
    ("C2H6",      "C-H", "Alkanes",         "C2H5",      "H"),
    ("C2H6",      "C-C", "Alkanes",         "CH3",       "CH3"),
    ("CH3CH2OH",  "C-H", "Alcohols",        "CH2CH2OH",  "H"),
    ("CH3CH2OH",  "C-C", "Alcohols",        "CH3",       "CH2OH"),
    ("CH3CH2OH",  "C-O", "Alcohols",        "C2H5",      "OH"),
    ("CH3CH2NH2", "C-H", "Amines",          "CH2CH2NH2", "H"),
    ("CH3CH2NH2", "C-C", "Amines",          "CH3",       "CH2NH2"),
    ("CH3CH2NH2", "N-H", "Amines",          "CH3CH2NH",  "H"),
    ("CH3CH2NH2", "C-N", "Amines",          "C2H5",      "NH2"),
    ("CH3CH2NO2", "C-H", "Nitro compounds", "CH2CH2NO2", "H"),
    ("CH3CH2NO2", "C-C", "Nitro compounds", "CH3",       "CH2NO2"),
    ("CH3CH2NO2", "C-N", "Nitro compounds", "C2H5",      "NO2"),
    ("CH3CH2NO2", "N-O", "Nitro compounds", "CH3CH2NO",  "O"),
    ("CH3CH2OOH", "C-H", "Peroxides",       "CH2CH2OOH", "H"),
    ("CH3CH2OOH", "C-C", "Peroxides",       "CH3",       "CH2OOH"),
    ("CH3CH2OOH", "O-O", "Peroxides",       "CH3CH2O",   "OH"),
    ("H2CO",      "C-H", "Carbonyls",       "HCO",       "H"),
]

CONFORMERS = {"CH3CH2OH": "anti vs gauche OH", "CH3CH2OOH": "OOH torsion",
              "CH3CH2NO": "syn vs anti C-N=O", "CH2OH": "non-planar radical",
              "CH2CH2OH": "OH torsion", "CH2CH2OOH": "OOH torsion",
              "CH2OOH": "OOH torsion"}

# heatmap-repo names (results.json) -> ours
HEAT_MOL = {"Ethane": "C2H6", "Ethanol": "CH3CH2OH", "Ethylamine": "CH3CH2NH2",
            "Nitroethane": "CH3CH2NO2", "Ethyl-Hydroperoxide": "CH3CH2OOH",
            "Methanal": "H2CO"}
MODELS = ["MACE-OFF23", "MACE-Osaka24", "ANI-1xnr", "MACE-OMOL", "UMA-S-1P1"]


def build(name):
    """RDKit geometry from the SMILES above; ORCA optimises it afterwards, so
    only connectivity and isomer matter."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from ase import Atoms
    smi = SPECIES[name]
    if name in ("H", "O"):
        return Atoms(name, positions=[[0, 0, 0]])
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    sym = [a.GetSymbol() for a in mol.GetAtoms()]
    pos = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    return Atoms(sym, positions=pos)


def finished(name):
    """Directory holding a normally-terminated single point for name, or None."""
    for d in (DLPNO_DIR, OUT_DIR):
        out = os.path.join(_HERE, d, f"{name}_sp.out")
        if os.path.exists(out) and "ORCA TERMINATED NORMALLY" in open(out, errors="ignore").read():
            return d
    return None


def needed():
    return sorted({s for c in CELLS for s in (c[0], c[3], c[4])})


def status():
    print(f"  {'species':<12}{'atoms':>6}{'mult':>6}   where")
    missing = []
    for nm in needed():
        a = build(nm)
        d = finished(nm)
        print(f"  {nm:<12}{len(a):>6}{OM.multiplicity(a, nm):>6}   "
              f"{d if d else 'MISSING'}")
        if not d:
            missing.append(nm)
    print(f"\n  {len(needed()) - len(missing)} done, {len(missing)} missing: "
          + " ".join(missing))
    done_cells = [c for c in CELLS if all(finished(s) for s in (c[0], c[3], c[4]))]
    print(f"  cells computable now: {len(done_cells)} of {len(CELLS)}")
    return missing


def write_inputs(maxcore):
    outdir = os.path.join(_HERE, OUT_DIR)
    os.makedirs(outdir, exist_ok=True)
    made = []
    for nm in needed():
        if finished(nm):
            continue
        a = build(nm)
        mult = OM.multiplicity(a, nm)
        extra = f"   <-- has conformers ({CONFORMERS[nm]}); verify the optimised isomer" \
            if nm in CONFORMERS else ""
        geom = "".join(f"{s:<3}{p[0]:>14.8f}{p[1]:>14.8f}{p[2]:>14.8f}\n"
                       for s, p in zip(a.get_chemical_symbols(), a.get_positions()))
        if len(a) == 1:
            with open(os.path.join(outdir, f"{nm}_sp.inp"), "w") as fh:
                fh.write(OM.ATOM_TEMPLATE.format(
                    name=nm, sym=a.get_chemical_symbols()[0], mult=mult,
                    state=OM.ATOM_STATE.get(nm, "?"), maxcore=maxcore,
                    method=OM.method_line(nm, a)))
        else:
            with open(os.path.join(outdir, f"{nm}_opt.inp"), "w") as fh:
                fh.write(OM.OPT_TEMPLATE.format(name=nm, natoms=len(a), mult=mult,
                                                extra=extra, geom=geom, maxcore=maxcore))
            with open(os.path.join(outdir, f"{nm}_sp.inp"), "w") as fh:
                fh.write(OM.SP_TEMPLATE.format(name=nm, mult=mult, maxcore=maxcore,
                                               level=OM.LEVEL, method=OM.method_line(nm, a)))
        made.append((nm, len(a), mult))

    # same runner as orca_molecules.write_inputs, copied verbatim
    sh = os.path.join(outdir, "run_orca_matrix.sh")
    with open(sh, "w") as fh:
        fh.write(
            '#!/bin/bash\n'
            '# Resumable ORCA driver, same as run_all.sh: logs each job to <job>.out,\n'
            '# skips finished ones. Set ORCA if orca is not on PATH:\n'
            '#     ORCA=/opt/orca/orca bash run_orca_matrix.sh\n'
            '# On the cluster the install is not on PATH until orca_env.sh is sourced.\n'
            '[ -f "$HOME/orca_env.sh" ] && source "$HOME/orca_env.sh"\n'
            'ORCA=${ORCA:-${ORCA_BIN:-orca}}\n'
            'command -v "$ORCA" >/dev/null || { echo "orca not found: source ~/orca_env.sh or set ORCA"; exit 1; }\n'
            'cd "$(dirname "$0")"\n'
            'ok () { grep -q "ORCA TERMINATED NORMALLY" "$1" 2>/dev/null; }\n'
            'run () {\n'
            '    ok "$1.out" && { echo "skip $1"; return 0; }\n'
            '    echo "=== $1  $(date +%H:%M:%S)"\n'
            '    "$ORCA" "$1.inp" > "$1.out" 2>&1\n'
            '    ok "$1.out" || { echo "    FAILED: see $1.out"; return 1; }\n'
            '}\n\n'
            'for f in $(ls -S -r *_opt.inp 2>/dev/null); do run "${f%.inp}"; done\n\n'
            'for f in $(ls -S -r *_sp.inp 2>/dev/null); do\n'
            '    b=${f%_sp.inp}\n'
            '    if [ -f "${b}_opt.inp" ] && [ ! -s "${b}_opt.xyz" ]; then\n'
            '        echo "SKIP ${b}_sp: no ${b}_opt.xyz (optimisation failed?)"\n'
            '        continue\n'
            '    fi\n'
            '    run "${b}_sp"\n'
            'done\n\n'
            'echo; echo "done. Check:  grep -L \\"TERMINATED NORMALLY\\" *.out"\n')
    os.chmod(sh, 0o755)
    print(f"  {len(made)} species written to {OUT_DIR}/")
    print(f"  {'species':<12}{'atoms':>6}{'mult':>6}")
    for nm, n, m in sorted(made, key=lambda x: (x[1], x[0])):
        print(f"  {nm:<12}{n:>6}{m:>6}" + ("   conformers" if nm in CONFORMERS else ""))
    print(f"\n  run:  bash {OUT_DIR}/run_orca_matrix.sh   (orca must be on PATH, as for run_all.sh)")


def energies():
    """{name: (E_hartree, ZPE_kcal)} from whichever directory holds it."""
    got = {}
    for nm in needed():
        d = finished(nm)
        if not d:
            continue
        txt = open(os.path.join(_HERE, d, f"{nm}_sp.out"), errors="ignore").read()
        e = None
        for p in OM._PATS:
            m = p.findall(txt)
            if m:
                e = float(m[-1]); break
        zpe = 0.0 if len(build(nm)) == 1 else None
        opt = os.path.join(_HERE, d, f"{nm}_opt.out")
        if os.path.exists(opt):
            otxt = open(opt, errors="ignore").read()
            z = OM._ZPE.findall(otxt)
            if z:
                zpe = float(z[-1]) * OM.HARTREE_KCAL
            if "***imaginary mode***" in otxt.lower():
                print(f"  !! {nm}: imaginary frequency, not a minimum")
        got[nm] = (e, zpe)
    return got


def parse():
    E = energies()
    K = OM.HARTREE_KCAL
    rows = []
    print(f"\n  {'molecule':<12}{'bond':<6}{'environment':<17}{'De':>8}{'D0':>8}")
    for mol, bond, env, fa, fb in CELLS:
        if not all(s in E for s in (mol, fa, fb)):
            print(f"  {mol:<12}{bond:<6}{env:<17}   missing: "
                  + " ".join(s for s in (mol, fa, fb) if s not in E))
            continue
        de = (E[fa][0] + E[fb][0] - E[mol][0]) * K
        zs = [E[s][1] for s in (mol, fa, fb)]
        d0 = de + (zs[1] + zs[2] - zs[0]) if None not in zs else float("nan")
        rows.append(dict(molecule=mol, bond=bond, environment=env, fragA=fa, fragB=fb,
                         De_kcal=f"{de:.2f}", D0_kcal=f"{d0:.2f}", level=OM.LEVEL))
        print(f"  {mol:<12}{bond:<6}{env:<17}{de:>8.2f}{d0:>8.2f}")
    out = os.path.join(_HERE, "matrix_refs.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n  wrote {out}  ({len(rows)} of {len(CELLS)} cells)")

    # model errors against De, from the heatmap repo's results.json
    rj = os.path.join(_HERE, HEATMAP, "thermodynamic_bde_results", "results.json")
    if not os.path.exists(rj):
        return
    R = json.load(open(rj))
    inv = {v: k for k, v in HEAT_MOL.items()}
    print(f"\n  signed error, model minus DLPNO De (kcal/mol)")
    print(f"  {'cell':<22}" + "".join(f"{m:>14}" for m in MODELS))
    err_rows = []
    for r in rows:
        hm = inv[r["molecule"]]
        line = f"  {r['molecule']+' '+r['bond']:<22}"
        er = dict(molecule=r["molecule"], bond=r["bond"], environment=r["environment"])
        for m in MODELS:
            v = R.get(m, {}).get(hm, {}).get(r["bond"])
            e = v - float(r["De_kcal"]) if v is not None else float("nan")
            er[m] = f"{e:.2f}"
            line += f"{e:>14.2f}"
        if r["bond"] == "N-O":
            line += "   <- OMOL/UMA ran with wrong multiplicities; rerun"
        print(line)
        err_rows.append(er)
    out2 = os.path.join(_HERE, "matrix_errors_dlpno.csv")
    with open(out2, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(err_rows[0].keys()))
        w.writeheader(); w.writerows(err_rows)
    print(f"\n  wrote {out2}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--parse", action="store_true")
    ap.add_argument("--maxcore", type=int, default=8000)
    a = ap.parse_args()
    if a.status:
        status()
    elif a.parse:
        parse()
    else:
        status()
        print()
        write_inputs(a.maxcore)


if __name__ == "__main__":
    main()
