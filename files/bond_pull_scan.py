#!/usr/bin/env python
"""
bond_pull_scan.py
=================

Rigid C-N dissociation scan of a single nitromethane molecule, to locate and
quantify PES cutoff seams in an MLIP. Unlike an NVE run (which samples whatever
bond lengths the dynamics wander through), this drives the C-N bond outward on a
fixed grid, so the SAME geometries are evaluated for every model -> directly
comparable seam diagnostics across ANI-1xnr, MACE-OMOL, and UMA.

Method (no dynamics, no thermostat, no timestep):
    - atom 0 = C, atom 1 = N (bond (0,1), matching the NVE convention).
    - The NO2 fragment (N + 2 O) is translated rigidly along the C->N axis so
      the C-N distance = r; CH3 stays fixed. This is the homolysis coordinate.
    - At each r: E(r), the analytic force projected on the pull axis, and the
      numerical -dE/dr by central difference.

Seam signatures:
    - E(r): a smooth model gives a smooth rise to a plateau; a seam is a kink/step.
    - F_numerical (= -dE/dr): a step in E becomes a SPIKE here -- the sharpest tell.
    - mismatch = F_analytic - F_numerical: ~0 for a conservative surface; it blows
      up at a seam, which is an unarguable statement that the PES is non-smooth/
      non-conservative there.

Per-model (run in the matching env: ani/omol -> mace-dox; uma -> uma env):
    python bond_pull_scan.py --model ani  --device cuda \\
        --ani-info /mfs/io/groups/nielsen/Cary/MLIP/ani-1xnr/model/ani-1xnr.info \\
        --outdir seam_scan --label ani
    python bond_pull_scan.py --model omol --device cuda \\
        --omol-path /mfs/io/groups/nielsen/Cary/MLIP/model_cache/MACE-omol-0-extra-large-1024.model \\
        --outdir seam_scan --label omol
    python bond_pull_scan.py --model uma  --device cuda \\
        --uma-path uma-s-1p1 --outdir seam_scan --label uma
"""

import argparse
import os
from pathlib import Path
import numpy as np
from ase import Atoms


def build_nm():
    """Single nitromethane, ordered so atom 0 = C and atom 1 = N."""
    symbols = ["C", "N", "O", "O", "H", "H", "H"]
    positions = [
        (0.000,  0.000,  0.000),   # 0 C
        (0.000,  0.000,  1.490),   # 1 N
        (0.000,  1.082,  2.054),   # 2 O
        (0.000, -1.082,  2.054),   # 3 O
        (1.028,  0.000, -0.363),   # 4 H (methyl)
        (-0.514, 0.890, -0.363),   # 5 H
        (-0.514, -0.890, -0.363),  # 6 H
    ]
    return Atoms(symbols=symbols, positions=positions)


def setup_calculator(model, device, ani_info, omol_path, uma_path, esen_path=None,
                     task="omol"):
    if model == "ani":
        import torch
        import torchani.neurochem as nc
        if not ani_info:
            raise ValueError("--ani-info required for ani")
        return nc.load_model_from_info_file(ani_info).to(torch.device(device)).ase()
    elif model == "off23":
        from mace.calculators import mace_off
        return mace_off(model="medium", device=device, default_dtype="float64")
    elif model == "omol":
        from mace.calculators import mace_omol
        src = omol_path if omol_path else "extra_large"
        return mace_omol(model=src, device=device, default_dtype="float32")
    elif model in ("uma", "esen"):
        from fairchem.core import FAIRChemCalculator
        src = uma_path if model == "uma" else esen_path
        if not src:
            raise ValueError(f"--{model}-path required for {model}")
        if os.path.exists(src):
            from fairchem.core.units.mlip_unit import load_predict_unit
            pred = load_predict_unit(path=src, device=device)
        else:
            from fairchem.core import pretrained_mlip
            pred = pretrained_mlip.get_predict_unit(src, device=device)
        # >>> CHANGED: task is now selectable. UMA is a Mixture of Linear Experts
        # whose routing weights alpha depend on composition, charge, spin AND task,
        # so changing the task changes which experts are active AND which DFT
        # reference the energy is on. The 6 A cutoff belongs to the shared
        # backbone, not the expert -- so if the seam is architectural it should
        # appear at the same r for every task.
        return FAIRChemCalculator(pred, task_name=task)
    raise ValueError(f"unknown model {model}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["ani", "omol", "uma", "off23", "esen"], required=True)
    # >>> CHANGED: "auto" picks cuda on the cluster, mps on Apple Silicon, else cpu.
    # Explicit --device cuda in the cluster scripts is unaffected.
    ap.add_argument("--device", default="cpu",
                    choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--ani-info", default=None)
    ap.add_argument("--omol-path", default=None)
    ap.add_argument("--uma-path", default=None)
    ap.add_argument("--esen-path", default=None)
    ap.add_argument("--outdir", default="seam_scan")
    ap.add_argument("--label", default=None)
    ap.add_argument("--r-min", type=float, default=1.2)
    ap.add_argument("--r-max", type=float, default=8.0)
    ap.add_argument("--dr", type=float, default=0.02)
    # >>> ADDED: explicit charge/spin for the charge-spin-aware models.
    # spin is the MULTIPLICITY (1 = singlet, 3 = triplet). Default 1 reproduces
    # every previously-run scan byte-for-byte.
    # VERIFIED 2026-08-04: MACE-OMOL *does* consume --spin. omol spin1 vs spin3
    # differ by up to 3.92 eV and the spin3 curve correlates with the UMA triplet
    # at r=0.977. An earlier note here claimed otherwise from OMol25 Appendix K --
    # that was a misreading: the paper restricts MACE to the "charge-NEUTRAL
    # split", which constrains CHARGE, not spin.
    # OPEN: does --charge do anything for omol? Run the charge control.
    ap.add_argument("--spin", type=int, default=1,
                    help="spin multiplicity: 1 singlet, 3 triplet")
    ap.add_argument("--charge", type=int, default=0)
    # >>> ADDED: UMA/eSEN DFT task. Ignored by mace-family models.
    ap.add_argument("--task", default="omol",
                    choices=["omol", "oc20", "omat", "odac", "omc"],
                    help="UMA MoLE task: selects the expert mix AND the DFT "
                         "reference the energy is on (omol=wB97M-V/def2-TZVPD, "
                         "oc20=RPBE, omat=PBE/PAW, odac=PBE-D3, omc=PBE-D3)")
    args = ap.parse_args()

    # >>> ADDED: resolve "auto" -> best available backend
    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            args.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
        print(f"  [device] auto -> {args.device}")
    if args.device == "cpu":
        import torch, os
        n = int(os.environ.get("OMP_NUM_THREADS", 0)) or os.cpu_count() or 1
        torch.set_num_threads(n)
        print(f"  [device] cpu with {n} threads")

    label = args.label or args.model
    # >>> FIXED 2026-08-03: previously this only fired when --label was omitted,
    # so `--label uma_m --spin 3` silently overwrote `--label uma_m --spin 1`.
    # Tag on the SPIN, regardless of whether a custom label was given.
    if args.spin != 1:
        label = f"{label}_spin{args.spin}"
    # >>> ADDED: tag non-default task too, same reasoning as spin
    if args.task != "omol" and args.model in ("uma", "esen"):
        label = f"{label}_{args.task}"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    needs_spin = args.model in ("omol", "uma", "esen")
    calc = setup_calculator(args.model, args.device, 
                            args.ani_info, args.omol_path, args.uma_path, args.esen_path,
                            task=args.task)

    base = build_nm()
    base_pos = base.get_positions()
    u = np.array([0.0, 0.0, 1.0])     # C->N axis (constant in this construction)
    no2 = [1, 2, 3]                   # rigidly translated fragment

    rs = np.arange(args.r_min, args.r_max + args.dr / 2, args.dr)
    print(f"[{label}] scanning C-N {args.r_min}->{args.r_max} A, "
          f"dr={args.dr} ({len(rs)} points)"
          + (f"  charge={args.charge} spin={args.spin}" if needs_spin else "")
          + (f"  task={args.task}" if args.model in ("uma","esen") else ""))
    if args.model == "omol" and args.spin != 1:
        print("  NOTE: MACE-OMOL is spin-aware (verified 2026-08-04). Charge-"
              "awareness is UNVERIFIED -- run the --charge control before assuming.")

    E = np.full(len(rs), np.nan)
    Fa = np.full(len(rs), np.nan)     # analytic -dE/dr = sum_{NO2} F.u
    for i, r in enumerate(rs):
        atoms = Atoms(symbols=base.get_chemical_symbols(),
                      positions=base_pos.copy())
        # place N at distance r along u; translate NO2 rigidly
        shift = (atoms.positions[0] + r * u) - atoms.positions[1]
        for k in no2:
            atoms.positions[k] += shift
        atoms.calc = calc
        if needs_spin:
            atoms.info["charge"] = args.charge      # >>> CHANGED
            atoms.info["spin"] = args.spin          # >>> CHANGED
        try:
            E[i] = atoms.get_potential_energy()
            F = atoms.get_forces()
            Fa[i] = sum(float(F[k] @ u) for k in no2)
        except Exception as exc:
            print(f"  r={r:.3f}: eval failed ({exc})")

    # numerical -dE/dr (central difference)
    Fn = np.full(len(rs), np.nan)
    Fn[1:-1] = -(E[2:] - E[:-2]) / (2 * args.dr)
    mismatch = Fa - Fn

    # locate the worst conservativeness violation
    valid = np.isfinite(mismatch)
    if valid.any():
        j = np.nanargmax(np.abs(mismatch))
        print(f"  largest |F_analytic - F_numerical| = {abs(mismatch[j]):.3f} eV/A "
              f"at r = {rs[j]:.3f} A")

    csv = outdir / f"pull_{label}.csv"
    with open(csv, "w") as fh:
        # >>> ADDED: provenance header. Pandas/np skip these with comment="#".
        # Never again be unable to tell which spin produced a curve.
        import datetime, platform
        srcpath = {"uma": args.uma_path, "esen": args.esen_path,
                   "omol": args.omol_path, "ani": args.ani_info}.get(args.model, "")
        fh.write(f"# model={args.model} path={srcpath}\n")
        fh.write(f"# charge={args.charge} spin={args.spin}"
                 f"{f' task={args.task}' if args.model in ('uma','esen') else ''}\n")
        fh.write(f"# device={args.device} r={args.r_min}:{args.r_max}:{args.dr} label={label}\n")
        try:
            import torch
            fh.write(f"# torch={torch.__version__}\n")
        except Exception:
            pass
        for _m in ("mace", "fairchem.core", "ase", "torchani"):
            try:
                _mod = __import__(_m, fromlist=["__version__"])
                fh.write(f"# {_m}={getattr(_mod, '__version__', '?')}\n")
            except Exception:
                pass
        fh.write(f"# run={datetime.datetime.now().isoformat(timespec='seconds')}"
                 f" host={platform.node()}\n")
        fh.write("bond_A,E_eV,E_rel_eV,F_analytic_eV_A,F_numerical_eV_A,mismatch_eV_A\n")
        e0 = E[np.isfinite(E)][0] if np.isfinite(E).any() else 0.0
        for i, r in enumerate(rs):
            fh.write(f"{r:.4f},{E[i]:.6f},{E[i]-e0:.6f},"
                     f"{Fa[i]:.6f},{Fn[i]:.6f},{mismatch[i]:.6f}\n")
    print(f"  wrote {csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        e0 = E[np.isfinite(E)][0]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        a1.plot(rs, E - e0, lw=1.4)
        a1.set_ylabel("E - E(r_min)  (eV)")
        a1.set_title(f"{label}: rigid C-N dissociation")
        a2.plot(rs, Fa, lw=1.2, label="analytic  (sum F.u)")
        a2.plot(rs, Fn, lw=1.0, alpha=0.8, label="numerical  (-dE/dr)")
        a2.plot(rs, mismatch, lw=1.0, color="crimson", label="mismatch")
        a2.axhline(0, color="k", lw=0.5)
        a2.set_xlabel("C-N distance (A)")
        a2.set_ylabel("force along bond (eV/A)")
        a2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"pull_{label}.png", dpi=150)
        print(f"  wrote {outdir / f'pull_{label}.png'}")
    except Exception as exc:
        print(f"  (plot skipped: {exc})")


if __name__ == "__main__":
    main()