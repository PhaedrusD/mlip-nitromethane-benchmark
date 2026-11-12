#!/usr/bin/env python
"""
prep_dense_box.py
=================

Build a CLEAN, overlap-free dense nitromethane box by staged compression with
relaxation between stages -- the defensible, model-neutral way to reach
1.97 g/cm^3 (replaces the single-shot COM scaling that caused interpenetration).

Protocol (mirrors the ReaxFF paper's compress-and-relax approach):
    1. Read a clash-free native-density box (Packmol ~1.13 liquid, or the
       native-density crystal). Fix molecule membership ONCE here.
    2. Compress isotropically toward the target density in N stages, moving only
       molecular centers of mass (molecules stay rigid).
    3. After each stage: FIRE minimization + a short low-T NVT, so atoms relax
       out of any mild overlap before the next compression step.
    4. Verify the final box reads as N0 intact CH3NO2 (maxsize ~7) and write it.

Box prep uses a NON-reactive potential (MACE-OFF23) at low T so molecules stay
intact and no chemistry is initiated during preparation -- the prepared box is
then the common starting point for every production model.

Run on a GPU node (srun/sbatch). Example:
    python prep_dense_box.py --in nm_box_native.xyz --target-density 1.97 \\
        --out nm_box_1p97_clean.xyz --device cuda
"""

import argparse
from collections import Counter
import numpy as np
from ase.io import read, write
from ase import units
from ase.optimize import FIRE
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.neighborlist import natural_cutoffs, NeighborList

AMU_TO_G = 1.66053906660e-24
A3_TO_CM3 = 1.0e-24
ATOMS_PER_NM = 7


def density(atoms):
    return (atoms.get_masses().sum() * AMU_TO_G) / (atoms.get_volume() * A3_TO_CM3)


def hill_formula(symbols):
    c = Counter(symbols); parts = []
    for el in ("C", "H"):
        if el in c:
            n = c.pop(el); parts.append(el if n == 1 else f"{el}{n}")
    for el in sorted(c):
        n = c[el]; parts.append(el if n == 1 else f"{el}{n}")
    return "".join(parts)


def connected_groups(atoms, scale=1.1):
    cut = natural_cutoffs(atoms, mult=scale)
    nl = NeighborList(cut, self_interaction=False, bothways=True)
    nl.update(atoms)
    n = len(atoms); parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for i in range(n):
        nbrs, _ = nl.get_neighbors(i)
        for j in nbrs:
            ra, rb = find(i), find(j)
            if ra != rb:
                parent[ra] = rb
    g = {}
    for i in range(n):
        g.setdefault(find(i), []).append(i)
    return [np.array(v) for v in g.values()]


def com_scale(atoms, groups, scale):
    """Move each group's COM toward the origin by `scale`, keep it rigid."""
    pos = atoms.get_positions(); m = atoms.get_masses(); new = pos.copy()
    for idx in groups:
        com = (pos[idx] * m[idx, None]).sum(0) / m[idx].sum()
        new[idx] = scale * com + (pos[idx] - com)
    atoms.set_positions(new)
    atoms.set_cell(atoms.cell[:] * scale, scale_atoms=False)
    atoms.wrap()


def load_calc(device):
    from mace.calculators import mace_off
    return mace_off(model="medium", device=device, default_dtype="float64")


def species_report(atoms, scale=1.05):
    comps = connected_groups(atoms, scale)
    counts = Counter(hill_formula([atoms[j].symbol for j in g]) for g in comps)
    return len(comps), counts.get("CH3NO2", 0), max(len(g) for g in comps)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="infile", required=True,
                    help="Clash-free native-density box")
    ap.add_argument("--target-density", type=float, default=1.97)
    ap.add_argument("--out", default="nm_box_clean.xyz")
    ap.add_argument("--stages", type=int, default=6)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--relax-T", type=float, default=50.0,
                    help="Low temperature (K) for the per-stage NVT relax")
    ap.add_argument("--relax-fs", type=float, default=500.0,
                    help="NVT relax duration per stage in fs")
    ap.add_argument("--fmax", type=float, default=0.5,
                    help="FIRE force convergence (eV/A)")
    ap.add_argument("--max-min-steps", type=int, default=200)
    args = ap.parse_args()

    atoms = read(args.infile)
    atoms.set_pbc([True, True, True])
    n0 = len(atoms) // ATOMS_PER_NM

    # Fix molecule membership from the (clean) native box ONCE.
    groups = connected_groups(atoms, 1.1)
    nmol, nm, mx = species_report(atoms)
    print(f"Native box: {len(atoms)} atoms, density {density(atoms):.4f} g/cm^3")
    print(f"  groups={len(groups)}  CH3NO2={nm}/{n0}  maxsize={mx}")
    if len(groups) != n0 or mx > ATOMS_PER_NM + 4:
        print("  !!! Input box is not clean (molecules already overlapping or "
              "merged). Start from a Packmol ~1.13 liquid or a native crystal.")
        # proceed anyway, but the warning stands

    calc = load_calc(args.device)
    atoms.calc = calc

    rho0 = density(atoms)
    densities = np.linspace(rho0, args.target_density, args.stages + 1)[1:]
    print(f"\nStaged compression {rho0:.3f} -> {args.target_density:.3f} g/cm^3 "
          f"in {args.stages} stages")

    n_nvt = int(args.relax_fs / 0.5)  # 0.5 fs steps
    for k, rho in enumerate(densities, 1):
        scale = (density(atoms) / rho) ** (1.0 / 3.0)
        com_scale(atoms, groups, scale)
        atoms.calc = calc

        FIRE(atoms, logfile=None).run(fmax=args.fmax, steps=args.max_min_steps)
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.relax_T)
        Stationary(atoms)
        dyn = Langevin(atoms, timestep=0.5 * units.fs,
                       temperature_K=args.relax_T, friction=0.05 / units.fs)
        dyn.run(n_nvt)

        nmol, nm, mx = species_report(atoms)
        print(f"  stage {k}/{args.stages}: rho={density(atoms):.3f}  "
              f"groups={nmol}  CH3NO2={nm}/{n0}  maxsize={mx}")

    nmol, nm, mx = species_report(atoms)
    print(f"\nFinal: density {density(atoms):.4f} g/cm^3  "
          f"CH3NO2={nm}/{n0}  maxsize={mx}")
    if nm == n0 and mx <= ATOMS_PER_NM + 4:
        print("  CLEAN: box reads as intact NM. Safe to use as the shared box.")
    else:
        print("  !!! NOT clean -- inspect before using. Try more --stages or a "
              "lower --relax-T, or check the native input.")

    write(args.out, atoms)
    print(f"\nWrote {args.out}")
    print(f"Verify independently:  python calibrate_bond_scale.py --struct {args.out}")


if __name__ == "__main__":
    main()