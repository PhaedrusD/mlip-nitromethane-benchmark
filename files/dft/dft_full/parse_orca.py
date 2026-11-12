#!/usr/bin/env python
"""
parse_orca.py
=============

Scrape ORCA single-point energies into a CSV ready to overlay on the MLIP
bond_pull_scan curves. Reads manifest.csv (from make_dft_snapshots.py) plus the
matching .out files.

Output columns:
  job,label,method,bond_A,natoms,E_hartree,E_eV,E_rel_eV,converged
E_rel_eV references each METHOD to its own lowest energy, so the DFT-RKS and
DFT-UKS curves plot on the same footing as the model E_rel curves.
"""

import argparse
import csv
from pathlib import Path

HARTREE_EV = 27.211386245988


def parse_energy(outpath):
    e, conv = None, False
    p = Path(outpath)
    if not p.exists():
        return None, False
    for line in p.read_text(errors="ignore").splitlines():
        if "FINAL SINGLE POINT ENERGY" in line:
            try:
                e = float(line.split()[-1])
            except ValueError:
                pass
        elif "ORCA TERMINATED NORMALLY" in line:
            conv = True
    return e, conv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="dft_snapshots")
    ap.add_argument("--out", default="dft_energies.csv")
    args = ap.parse_args()
    d = Path(args.dir)

    mf = d / "manifest_orca.csv"
    if not mf.exists():
        mf = d / "manifest.csv"
    rows = list(csv.DictReader(open(mf)))
    for m in rows:
        e, conv = parse_energy(d / f"{m['job']}.out")
        m["E_hartree"] = "" if e is None else f"{e:.8f}"
        m["E_eV"] = "" if e is None else f"{e * HARTREE_EV:.6f}"
        m["converged"] = int(conv)
        m["_e"] = e

    # per-method reference to that method's minimum energy
    for method in {m["method"] for m in rows}:
        es = [m["_e"] for m in rows if m["method"] == method and m["_e"] is not None]
        emin = min(es) if es else None
        for m in rows:
            if m["method"] == method and m["_e"] is not None and emin is not None:
                m["E_rel_eV"] = f"{(m['_e'] - emin) * HARTREE_EV:.6f}"
            elif m["method"] == method:
                m["E_rel_eV"] = ""

    fields = ["job", "label", "method", "bond_A", "natoms",
              "E_hartree", "E_eV", "E_rel_eV", "converged"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for m in rows:
            w.writerow(m)

    ok = sum(m["converged"] for m in rows)
    print(f"parsed {len(rows)} jobs, {ok} converged -> {args.out}")
    bad = [m["job"] for m in rows if not m["converged"]]
    if bad:
        print("NOT converged / missing:", ", ".join(bad[:12]),
              "..." if len(bad) > 12 else "")


if __name__ == "__main__":
    main()