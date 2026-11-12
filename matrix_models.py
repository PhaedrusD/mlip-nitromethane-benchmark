#!/usr/bin/env python
"""
matrix_models.py -- the five models evaluated on the 17-cell matrix at the
reference geometries, with the right multiplicity on every fragment
=============================================================================

WHY NOT REUSE results.json FROM THE HEATMAP REPO
    That run relaxed every fragment under each model with BFGS and asked
    for the energy of whatever it ended up as. For CH2OOH that is H2CO + OH,
    31 kcal/mol below the radical at DLPNO-CCSD(T), so the peroxide C-C
    cell measured the bond energy minus a decomposition. It also labelled
    the nitro N-O split [1,2,2]; the O atom is a triplet and nitrosoethane a
    singlet. Neither can be seen from results.json because no geometry was
    saved.

    Here every species is evaluated as a SINGLE POINT at its
    wB97X-D3/def2-TZVP geometry from orca_dlpno/ or orca_matrix/, the same
    geometry the DLPNO energy was computed on. No relaxation, so nothing can
    fall apart, and the comparison is model energy vs reference energy on
    identical nuclei, which is what De means.

ONE MODEL PER PROCESS
    mace-dox and uma cannot share a shell. Run each --model in its own srun
    block; --merge afterwards.

SETUP
    Copy matrix_models.py AND the matrix_geoms/ folder (25 xyz, species.csv,
    matrix_refs.csv) into the cluster MLIP directory, next to run_reactive_md.py.
    Nothing else from the laptop is needed.

USAGE (cluster)
    python matrix_models.py --model off23
    python matrix_models.py --model osaka24 --osaka-path .../mace-osaka24-medium-headfix.model
    python matrix_models.py --model omol
    python matrix_models.py --model ani --ani-info .../ani-1xnr.info
    python matrix_models.py --model uma --uma-path uma-s-1p1
    python matrix_models.py --merge          # -> matrix_errors_dlpno_sp.csv + bde_matrix_data_dlpno.py
"""
import argparse, csv, os, sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

KCAL = 23.0605
OUTDIR = "matrix_models"
GEOMS = "matrix_geoms"          # 25 xyz files + species.csv + matrix_refs.csv, made on the laptop

# (molecule, bond, environment, fragment A, fragment B): the 17 cells
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


def needed():
    return sorted({s for c in CELLS for s in (c[0], c[3], c[4])})


def species_table():
    f = os.path.join(_HERE, GEOMS, "species.csv")
    if not os.path.exists(f):
        sys.exit(f"  {GEOMS}/species.csv not found: copy the matrix_geoms/ folder next to this script")
    return {r["species"]: r for r in csv.DictReader(open(f))}
LABEL = {"off23": "MACE-OFF23", "osaka24": "MACE-Osaka24", "omol": "MACE-OMOL",
         "uma": "UMA-S-1P1", "ani": "ANI-1xnr"}


def geometry(name, table):
    """ASE Atoms at the reference geometry, charge 0, multiplicity from
    species.csv (odd electrons doublet, O atom triplet, nitrosoethane singlet)."""
    from ase.io import read
    at = read(os.path.join(_HERE, GEOMS, f"{name}.xyz"))
    at.info["charge"] = 0
    at.info["spin"] = int(table[name]["mult"])
    at.pbc = False
    return at


def evaluate(a):
    from run_reactive_md import setup_calculator
    calc = setup_calculator(a.model, osaka_path=a.osaka_path, ani_info=a.ani_info,
                            uma_path=a.uma_path, omol_path=a.omol_path, device=a.device)
    os.makedirs(os.path.join(_HERE, OUTDIR), exist_ok=True)
    table = species_table()
    rows = []
    for nm in needed():
        at = geometry(nm, table)
        at.calc = calc
        try:
            e = at.get_potential_energy()
        except Exception as exc:
            print(f"  !! {nm}: {exc}")
            e = float("nan")
        rows.append((nm, len(at), at.info["spin"], e))
        print(f"  {nm:<12} {len(at):>3} atoms  mult {at.info['spin']}   E = {e:14.6f} eV")
    out = os.path.join(_HERE, OUTDIR, f"energies_{a.model}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["species", "n_atoms", "mult", "E_eV", "model"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.6f}", a.model])
    print(f"  wrote {out}")


def merge():
    rf = os.path.join(_HERE, GEOMS, "matrix_refs.csv")
    refs = {(r["molecule"], r["bond"]): r for r in csv.DictReader(open(rf))}
    E = {}
    for m in LABEL:
        f = os.path.join(_HERE, OUTDIR, f"energies_{m}.csv")
        if os.path.exists(f):
            E[m] = {r["species"]: float(r["E_eV"]) for r in csv.DictReader(open(f))}
    if not E:
        sys.exit(f"  no energies_*.csv in {OUTDIR}/")
    models = [m for m in LABEL if m in E]
    print(f"\n  signed error, model minus DLPNO-CCSD(T) De, single points at the reference geometries (kcal/mol)")
    print(f"  {'cell':<24}" + "".join(f"{LABEL[m]:>14}" for m in models))
    out_rows = []
    for mol, bond, env, fa, fb in CELLS:
        ref = float(refs[(mol, bond)]["De_kcal"])
        row = dict(molecule=mol, bond=bond, environment=env, De_ref=ref)
        line = f"  {mol + ' ' + bond:<24}"
        for m in models:
            try:
                de = (E[m][fa] + E[m][fb] - E[m][mol]) * KCAL
                err = de - ref
            except KeyError:
                de = err = float("nan")
            row[LABEL[m] + "_De"] = f"{de:.2f}"
            row[LABEL[m]] = f"{err:.2f}"
            line += f"{err:>14.2f}"
        print(line)
        out_rows.append(row)
    out = os.path.join(_HERE, "matrix_errors_dlpno_sp.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print(f"\n  wrote {out}")

    # a drop-in replacement for the heatmap repo's bde_matrix_data.py arrays
    BONDS = ["C-H", "C-C", "C-O", "N-H", "C-N", "N-O", "O-O"]
    ENVS = ["Alkanes", "Alcohols", "Amines", "Nitro compounds", "Peroxides", "Carbonyls"]
    py = ["# generated by matrix_models.py --merge: errors vs DLPNO-CCSD(T) De at reference geometries",
          "import numpy as np", "", f"BONDS = {BONDS!r}",
          'ENVIRONMENTS = ["Alkanes", "Alcohols", "Amines", "Nitro\\nCompounds", "Peroxides", "Carbonyls"]', ""]
    var = {"off23": "data_mace_off23", "osaka24": "data_mace_osaka24", "ani": "data_ani_1xnr",
           "omol": "data_mace_omol", "uma": "data_uma_s_1p1"}
    for m in models:
        grid = np.full((7, 6), np.nan)
        for r in out_rows:
            grid[BONDS.index(r["bond"]), ENVS.index(r["environment"])] = float(r[LABEL[m]])
        py.append(f"{var[m]} = np.array([")
        for i, b in enumerate(BONDS):
            cells = ", ".join("np.nan" if np.isnan(v) else f"{v:7.2f}" for v in grid[i])
            py.append(f"    [{cells}],  # {b}")
        py.append("])\n")
    out2 = os.path.join(_HERE, "bde_matrix_data_dlpno.py")
    open(out2, "w").write("\n".join(py))
    print(f"  wrote {out2}  (paste its arrays over the ones in bde_matrix_data.py)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(LABEL))
    ap.add_argument("--osaka-path"); ap.add_argument("--ani-info")
    ap.add_argument("--uma-path"); ap.add_argument("--omol-path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        merge()
    elif a.model:
        evaluate(a)
    else:
        ap.error("--model or --merge")


if __name__ == "__main__":
    main()
