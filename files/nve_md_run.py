#!/usr/bin/env python
"""
nve_md_run.py
=============================================================
Energy-conservation test for the cutoff-seam failure mode.

Protocol: Maxwell-Boltzmann init -> Berendsen NVT heat/equilibrate at T_ignite for
--heat-ps -> pure NVE (VelocityVerlet) for --prod-ps. Total energy MUST be conserved
in the NVE segment, even while exothermic chemistry happens (PE -> KE, total flat).
A model with a cutoff discontinuity will break that conservation when a dissociating
fragment crosses the cutoff (~5-6 A) -- showing up as a step in total energy and a
spike in max force exactly when the breaking bond crosses the seam.

WHY SINGLE-MOLECULE IS THE CLEAN TEST: the seam is two fragments isolated beyond
their MUTUAL cutoff. In a dense box, neighbor molecules sit inside the cutoff and
mask it. A single nitromethane (or very low density) lets the fragment actually
reach 5-6 A in vacuum, so the artifact is exposed and attributable to one bond.

Logged every --log-every steps to <outdir>/nve_log.csv:
    step, time_ps, phase, PE_eV, KE_eV, Etot_eV, T_K, maxF_eV_A, bond_A
A summary plot (Etot / maxF / bond vs time) is written to <outdir>/nve_summary.png.

Compare UMA / MACE-OMOL against an agnostic model (e.g. osaka24) run identically:
the seam should break conservation for the OMol models and not for the agnostic one.

Run (single molecule, the clean seam test):
    python nve_md_run.py --model uma --box nitromethane.xyz --outdir nve_uma_single \\
        --T-ignite 3000 --heat-ps 0.5 --prod-ps 20.0
"""

import argparse
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase.io import read, write
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.verlet import VelocityVerlet


# ---------------- calculators (lazy imports: only the one you ask for loads) -------
def get_calc(name, args, device):
    name = name.lower()
    if name == "osaka24":
        from mace.calculators import MACECalculator
        return MACECalculator(model_paths=[args.osaka_path], device=device,
                              default_dtype="float64")
    if name == "off23":
        from mace.calculators import mace_off
        return mace_off(model="medium", device=device, default_dtype="float64")
    if name == "omol":
        from mace.calculators import mace_omol
        return mace_omol(model="extra_large", device=device, default_dtype="float64")
    if name == "uma":
        import torch
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        _orig = torch.load
        torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
        predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=device)
        return FAIRChemCalculator(predictor, task_name="omol")
    if name == "ani":
        import torchani.neurochem as nc
        return nc.load_model_from_info_file(args.ani_info).ase()
    sys.exit(f"Unknown model '{name}'")


def autodetect_bond(atoms):
    """Default tracked bond = first C with first N (the C-N for nitromethane)."""
    syms = atoms.get_chemical_symbols()
    try:
        c = syms.index("C")
        n = syms.index("N")
        return (c, n)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    choices=["osaka24", "off23", "omol", "uma", "ani"])
    ap.add_argument("--osaka-path", default="mace-osaka24-medium.model")
    ap.add_argument("--ani-info", default="ani-1xnr/model/ani-1xnr.info")
    ap.add_argument("--box", required=True, help="input geometry (.xyz)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--T-ignite", type=float, required=True,
                    help="heating/equilibration target temperature (K)")
    ap.add_argument("--heat-ps", type=float, default=1.0)
    ap.add_argument("--prod-ps", type=float, default=15.0)
    ap.add_argument("--timestep", type=float, default=0.5, help="fs (default 0.5)")
    ap.add_argument("--thermostat-tau", type=float, default=100.0,
                    help="Berendsen tau in fs for the heating phase (default 100)")
    ap.add_argument("--log-every", type=int, default=20, help="steps between log rows")
    ap.add_argument("--track-bond", default=None,
                    help='"i,j" atom indices to log distance; default auto C-N')
    ap.add_argument("--charge", type=int, default=0, help="for UMA/OMOL")
    ap.add_argument("--spin", type=int, default=1, help="multiplicity for UMA/OMOL")
    ap.add_argument("--device", default=None, help="cuda or cpu (auto if unset)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    np.random.seed(args.seed)

    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"

    atoms = read(args.box)
    atoms.info["charge"] = args.charge      # read by UMA/OMOL; harmless otherwise
    atoms.info["spin"] = args.spin
    natoms = len(atoms)

    # which bond to track for seam-crossing correlation
    if args.track_bond:
        i, j = (int(x) for x in args.track_bond.split(","))
        bond = (i, j)
    else:
        bond = autodetect_bond(atoms)
    bond_msg = f"{bond}" if bond else "none"

    print(f"model={args.model}  device={args.device}  atoms={natoms}  "
          f"pbc={atoms.pbc.any()}  tracking bond={bond_msg}")
    print(f"dt={args.timestep} fs  T_ignite={args.T_ignite} K  "
          f"heat={args.heat_ps} ps  prod={args.prod_ps} ps")

    atoms.calc = get_calc(args.model, args, args.device)

    dt = args.timestep * units.fs
    n_heat = int(round(args.heat_ps * 1000.0 / args.timestep))
    n_prod = int(round(args.prod_ps * 1000.0 / args.timestep))

    # initialize velocities and strip net translation/rotation (clean for 1 molecule)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.T_ignite)
    Stationary(atoms)
    ZeroRotation(atoms)

    # ---- logging machinery -------------------------------------------------------
    log_path = os.path.join(args.outdir, "nve_log.csv")
    traj_path = os.path.join(args.outdir, "traj.xyz")
    logf = open(log_path, "w")
    logf.write("step,time_ps,phase,PE_eV,KE_eV,Etot_eV,T_K,maxF_eV_A,bond_A\n")
    rows = {"t": [], "Etot": [], "maxF": [], "bond": [], "phase": []}
    state = {"step": 0, "phase": "heat"}

    def log():
        a = atoms
        pe = a.get_potential_energy()
        ke = a.get_kinetic_energy()
        etot = pe + ke
        T = a.get_temperature()
        maxf = float(np.abs(a.get_forces()).max())
        d = a.get_distance(*bond) if bond else float("nan")
        t_ps = state["step"] * args.timestep / 1000.0
        logf.write(f"{state['step']},{t_ps:.5f},{state['phase']},{pe:.6f},{ke:.6f},"
                   f"{etot:.6f},{T:.1f},{maxf:.4f},{d:.4f}\n")
        logf.flush()
        rows["t"].append(t_ps); rows["Etot"].append(etot)
        rows["maxF"].append(maxf); rows["bond"].append(d); rows["phase"].append(state["phase"])
        write(traj_path, a, append=(state["step"] != 0))

    def bump():
        state["step"] += 1

    # ---- heating / equilibration (NVT Berendsen) ---------------------------------
    print(f"\nHeating/equilibrating at {args.T_ignite} K for {n_heat} steps...")
    heat = NVTBerendsen(atoms, timestep=dt, temperature_K=args.T_ignite,
                        taut=args.thermostat_tau * units.fs)
    heat.attach(log, interval=args.log_every)
    heat.attach(bump, interval=1)
    log()  # t=0 row
    heat.run(n_heat)

    # ---- production (pure NVE) ----------------------------------------------------
    state["phase"] = "prod"
    prod_start_step = state["step"]
    print(f"Production NVE for {n_prod} steps (energy must stay flat)...")
    nve = VelocityVerlet(atoms, timestep=dt)
    nve.attach(log, interval=args.log_every)
    nve.attach(bump, interval=1)
    nve.run(n_prod)
    logf.close()

    # ---- conservation metrics on the PRODUCTION segment --------------------------
    t = np.array(rows["t"]); E = np.array(rows["Etot"])
    ph = np.array(rows["phase"])
    pm = ph == "prod"
    if pm.sum() >= 2:
        tp, Ep = t[pm], E[pm]
        slope = np.polyfit(tp - tp[0], Ep, 1)[0]           # eV / ps
        drift_per_atom = slope * 1000.0 / natoms           # meV / atom / ps
        excursion = (Ep.max() - Ep.min()) * 1000.0 / natoms  # meV/atom peak-to-peak
        # largest single-step jump in Etot (seam crossings show up here)
        jumps = np.abs(np.diff(Ep)) * 1000.0 / natoms
        max_jump = jumps.max() if jumps.size else 0.0
        jump_at = tp[1:][np.argmax(jumps)] if jumps.size else float("nan")
    else:
        drift_per_atom = excursion = max_jump = jump_at = float("nan")

    print("\n" + "=" * 60)
    print(f"NVE ENERGY CONSERVATION  ({args.model})")
    print("=" * 60)
    print(f"production drift      : {drift_per_atom:+.4f} meV/atom/ps")
    print(f"peak-to-peak excursion: {excursion:.4f} meV/atom")
    print(f"largest single jump   : {max_jump:.4f} meV/atom  at t={jump_at:.3f} ps")
    print("Reference: a well-behaved NVE run holds drift to <~0.1-1 meV/atom/ps and")
    print("has no isolated steps. A cutoff-seam artifact shows a sharp jump that")
    print("coincides with the tracked bond crossing ~5-6 A.")

    # ---- summary plot ------------------------------------------------------------
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(t, (E - E[0]) * 1000.0 / natoms, color="black")
    ax[0].axvline(t[pm][0] if pm.any() else 0, color="gray", ls=":", lw=1)
    ax[0].set_ylabel("dEtot (meV/atom)")
    ax[0].set_title(f"NVE conservation: {args.model}  (dotted = NVE start)")
    ax[1].plot(t, rows["maxF"], color="tab:red")
    ax[1].set_ylabel("max |F| (eV/A)")
    if bond:
        ax[2].plot(t, rows["bond"], color="tab:blue")
        ax[2].axhspan(5.0, 6.0, color="orange", alpha=0.2)  # the seam window
        ax[2].set_ylabel(f"bond {bond} (A)")
    ax[2].set_xlabel("time (ps)")
    fig.tight_layout()
    png = os.path.join(args.outdir, "nve_summary.png")
    fig.savefig(png, dpi=160)
    print(f"\nLog : {log_path}\nTraj: {traj_path}\nPlot: {png}")


if __name__ == "__main__":
    main()