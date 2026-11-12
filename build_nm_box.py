#!/usr/bin/env python
"""
build_nm_box.py
===============

Build a periodic box of nitromethane molecules for reactive MD.
Uses packmol to create a physically reasonable dense packing.

Usage:
    python build_nm_box.py [--n-mol 32] [--density 1.137] [--output nm_box.xyz]

Requires:
    - packmol on PATH (conda install -c conda-forge packmol)
    - rdkit, ase, numpy
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write


def build_single_nitromethane():
    """Build CH3NO2 with RDKit, return ASE Atoms."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles("C[N+](=O)[O-]")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = np.array([
        [conf.GetAtomPosition(i).x,
         conf.GetAtomPosition(i).y,
         conf.GetAtomPosition(i).z]
        for i in range(mol.GetNumAtoms())
    ])
    return Atoms(symbols=symbols, positions=positions)


def compute_box_size(n_molecules, density_gcc, molecular_mass_gmol=61.04):
    """Return cubic box side length (Å) for N molecules at target density."""
    total_mass_g = n_molecules * molecular_mass_gmol / 6.022e23
    volume_cm3 = total_mass_g / density_gcc
    side_length_ang = (volume_cm3 ** (1.0 / 3.0)) * 1e8
    return side_length_ang


def run_packmol(monomer_xyz, n_mol, box_inner, output_xyz, work_dir, tolerance=2.0):
    """Write packmol input, run packmol, return path to output file."""
    if not shutil.which("packmol"):
        raise RuntimeError(
            "packmol not found on PATH.\n"
            "  Install: conda install -c conda-forge packmol\n"
            "  Then:    conda activate mace-dox"
        )

    inp = f"""tolerance {tolerance}
output {output_xyz}
filetype xyz

structure {monomer_xyz}
    number {n_mol}
    inside cube 0.0 0.0 0.0 {box_inner:.4f}
end structure
"""
    input_file = Path(work_dir) / "packmol.inp"
    input_file.write_text(inp)

    with open(input_file) as f:
        proc = subprocess.run(
            ["packmol"], stdin=f, cwd=work_dir,
            capture_output=True, text=True, timeout=300,
        )

    if proc.returncode != 0:
        print("packmol stdout:", proc.stdout, file=sys.stderr)
        print("packmol stderr:", proc.stderr, file=sys.stderr)
        raise RuntimeError(f"packmol failed with return code {proc.returncode}")

    # Print last bit of packmol output for sanity
    last_lines = proc.stdout.strip().splitlines()[-10:]
    print("  packmol tail:")
    for line in last_lines:
        print(f"    {line}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-mol", type=int, default=32,
                    help="Number of nitromethane molecules (default 32)")
    ap.add_argument("--density", type=float, default=1.137,
                    help="Target density in g/cm^3 (default 1.137, liquid NM)")
    ap.add_argument("--output", default="nm_box.xyz",
                    help="Output filename for the periodic box")
    ap.add_argument("--work-dir", default=".",
                    help="Working directory for intermediate files")
    ap.add_argument("--packmol-margin", type=float, default=2.5,
                    help="Inner-box margin so nothing sits on the cell edge")
    args = ap.parse_args()

    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(exist_ok=True)

    L = compute_box_size(args.n_mol, args.density)
    L_packmol = L - args.packmol_margin

    print(f"Building box:")
    print(f"  N molecules:      {args.n_mol}")
    print(f"  Target density:   {args.density} g/cm^3")
    print(f"  Box side length:  {L:.3f} Å")
    print(f"  Packmol region:   {L_packmol:.3f} Å (with {args.packmol_margin} Å margin)")

    # Build single molecule template
    monomer = build_single_nitromethane()
    monomer_xyz = work_dir / "nitromethane.xyz"
    write(str(monomer_xyz), monomer)
    print(f"\n  Wrote monomer template: {monomer_xyz}")
    print(f"  Monomer: {len(monomer)} atoms")

    # Pack with packmol
    print("\n  Running packmol...")
    run_packmol(
        monomer_xyz.name, args.n_mol, L_packmol, "packed.xyz", work_dir,
    )

    # Load packed system, set PBC, center
    atoms = read(str(work_dir / "packed.xyz"))
    atoms.set_cell([L, L, L])
    atoms.set_pbc([True, True, True])
    atoms.center()

    output_path = work_dir / args.output
    write(str(output_path), atoms)

    print(f"\nDone.")
    print(f"  Total atoms: {len(atoms)}")
    print(f"  Cell:        {atoms.cell.lengths()}")
    print(f"  PBC:         {atoms.pbc}")
    print(f"  Output:      {output_path}")


if __name__ == "__main__":
    main()
