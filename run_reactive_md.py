#!/usr/bin/env python
"""
run_reactive_md.py
==================

Reactive MD of a periodic nitromethane box at high temperature.

Protocol:
    1. Load prebuilt periodic box from xyz
    2. Equilibrate at 300 K (Langevin NVT)
    3. Production run at 3000 K (Langevin NVT)
    4. Log energy, temperature, forces; save trajectory every N steps
    5. Detect blowup (NaN, runaway energy) and terminate gracefully

Usage:
    python run_reactive_md.py --model off23 --box nm_box.xyz --outdir off23_run
    python run_reactive_md.py --model osaka24 \\
        --osaka-path /path/to/mace-osaka24-medium.model \\
        --box nm_box.xyz --outdir osaka_run 

    # GPU (P100), reactive ANI-1xnr via torchani/NeuroChem
    python run_reactive_md.py --model ani --device cuda \\
        --ani-info /mfs/io/groups/nielsen/Cary/MLIP/ani-1xnr/model/ani-1xnr.info \\
        --box nm_box.xyz --outdir ani_run
"""

import argparse
import time
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary


# ------------------------------------------------------------------
# Calculator setup
# ------------------------------------------------------------------

def setup_calculator(model, osaka_path=None, ani_info=None, uma_path=None, omol_path=None, device="cpu"):
    if model == "off23":
        from mace.calculators import mace_off
        return mace_off(model="medium", device=device, default_dtype="float64")
    elif model == "omol":
        # MACE-omol-0 (extra-large, 1024-width), trained on OMol25 with charge+
        # spin embeddings. Load from a staged local file on group space so the
        # multi-GB weights never hit the home quota. float32 to fit the 16 GB V100.
        # Needs atoms.info["charge"]/["spin"] (set in main()).
        from mace.calculators import mace_omol
        src = omol_path if omol_path else "extra_large"   # path or size keyword
        calc = mace_omol(model=src, device=device, default_dtype="float32")
        calc.implemented_properties = ["energy", "free_energy", "forces"]
        return calc
    elif model == "uma":
        import os
        from fairchem.core import FAIRChemCalculator
        if not uma_path:
            raise ValueError("--uma-path required for uma model: a local .pt"
                            "checkpoint path, or a registered name like 'uma-s-1p1'")
        if os.path.exists(uma_path):
            from fairchem.core.units.mlip_unit import load_predict_unit
            predictor = load_predict_unit(path=uma_path, device=device)
        else:
            from fairchem.core import pretrained_mlip
            predictor = pretrained_mlip.get_predict_unit(uma_path, device=device)
        return FAIRChemCalculator(predictor, task_name="omol")
    elif model == "osaka24":
        from mace.calculators import MACECalculator
        if not osaka_path:
            raise ValueError("--osaka-path required for osaka24 model")
        # Try newer 'model_paths' API first, fall back to 'model_path'
        try:
            return MACECalculator(model_paths=[osaka_path],
                                  device=device, default_dtype="float64")
        except TypeError:
            return MACECalculator(model_path=osaka_path,
                                  device=device, default_dtype="float64")
    elif model == "ani":
        # Reactive ANI-1xnr (Zhang et al., Nat. Chem. 2024), loaded from the
        # NeuroChem .info file. This is NOT torchani.models.ANI1x (equilibrium);
        # the reactive model is required for bond-breaking at high T.
        #
        # Param-file resolution relies on the symlink:
        #   ~/.local/share/Torchani/Neurochem/ani-1xnr ->
        #       /mfs/io/groups/nielsen/Cary/MLIP/ani-1xnr/model/ani-1xnr
        import torch
        import torchani.neurochem as nc
        if not ani_info:
            raise ValueError("--ani-info required for ani model")
        dev = torch.device(device)
        ani_model = nc.load_model_from_info_file(ani_info).to(dev)
        # torchani models expose .ase() -> ase.calculators.calculator.Calculator.
        # Native dtype is float32 (what ANI-1xnr was trained/validated at);
        # leave it as-is rather than forcing float64.
        return ani_model.ase()
    elif model == "ani_mrr":
        # ANI-1xnr-Mrr: the 8 Morse-fine-tuned state dicts grafted into one
        # torchani Ensemble, so the AEV is computed once and shared (75 ms/step
        # vs 47.7 for a single member, not 8x).
        # >>> ani_info STAYS POINTED AT THE ORIGINAL ANI-1xnr .info FILE. It
        # supplies the architecture, AEV parameters and self-atomic energies;
        # the member*.pt checkpoints are state dicts alone and are found by the
        # loader through its own __file__, never named on the command line.
        import sys
        sys.path.insert(0, "/mfs/io/groups/nielsen/Cary/MLIP/ANI-1xnr-Mrr")
        from ani_1xnr_Mrr import get_calculator
        return get_calculator(info=ani_info, device=device)

    else:
        raise ValueError(f"Unknown model: {model}")


# ------------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------------

def check_sanity(atoms, e_initial, threshold_eV_per_atom=5.0):
    """Return (ok: bool, reason: str). Reports blowup via energy, NaN, Inf."""
    try:
        e = atoms.get_potential_energy()
    except Exception as exc:
        return False, f"exception computing energy: {exc}"

    if np.isnan(e) or np.isinf(e):
        return False, f"E_pot = {e}"

    per_atom = (e - e_initial) / len(atoms)
    if per_atom > threshold_eV_per_atom:
        return False, f"E_pot/atom excess = {per_atom:.2f} eV above initial"

    if np.any(np.isnan(atoms.get_positions())):
        return False, "NaN positions"

    return True, "ok"


# ------------------------------------------------------------------
# Main MD loop
# ------------------------------------------------------------------

def run_nvt(atoms, T_target, n_steps, dt_fs, friction_fs,
            log_interval, traj_path, log_path, e_initial, label="",
            thermostat="langevin", taut_fs=200.0):
    """
    Run NVT Langevin dynamics for n_steps with periodic logging.
    Returns (completed_successfully, steps_done).
    """
    dt = dt_fs * units.fs
    if thermostat == "berendsen":
        dyn = NVTBerendsen(atoms, timestep=dt, temperature_K=T_target,
                           taut=taut_fs * units.fs, fixcm=True)

    else:     
        friction_ase = friction_fs / units.fs  # convert fs^-1 -> ASE internal
        dyn = Langevin(atoms, timestep=dt, temperature_K=T_target,
                   friction=friction_ase, fixcm=True)

    traj = Trajectory(str(traj_path), "w", atoms)
    dyn.attach(traj.write, interval=log_interval)

    log_file = open(log_path, "w")
    log_file.write("# step  t_ps  T_K  E_pot_eV  E_tot_eV  Fmax_eV_per_A\n")

    header = f"  {'t/ps':>7} {'T/K':>8} {'E_pot/eV':>12} {'E_tot/eV':>12} {'wallclock':>10}"
    print(f"\n{label}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    t_start = time.time()
    completed = True
    steps_done = 0

    for chunk_start in range(0, n_steps, log_interval):
        chunk_size = min(log_interval, n_steps - chunk_start)
        try:
            dyn.run(chunk_size)
        except Exception as exc:
            print(f"  !!! MD integrator exception at step {chunk_start}: {exc}")
            completed = False
            break

        steps_done = chunk_start + chunk_size
        t_sim_ps = steps_done * dt_fs / 1000.0
        t_elapsed = time.time() - t_start

        T_inst = atoms.get_kinetic_energy() / (1.5 * units.kB * len(atoms))
        E_pot = atoms.get_potential_energy()
        E_tot = atoms.get_total_energy()
        forces = atoms.get_forces()
        F_max = np.max(np.linalg.norm(forces, axis=1))

        print(f"  {t_sim_ps:7.3f} {T_inst:8.1f} {E_pot:12.3f} {E_tot:12.3f} "
              f"{t_elapsed:9.0f}s")
        log_file.write(
            f"{steps_done} {t_sim_ps:.4f} {T_inst:.2f} "
            f"{E_pot:.4f} {E_tot:.4f} {F_max:.4f}\n"
        )
        log_file.flush()

        ok, reason = check_sanity(atoms, e_initial)
        if not ok:
            print(f"  !!! Blowup detected at step {steps_done}: {reason}")
            completed = False
            break

    traj.close()
    log_file.close()
    return completed, steps_done


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["off23", "osaka24", "ani", "omol", "uma", "ani_mrr"], required=True)
    ap.add_argument("--osaka-path", default=None,
                    help="Path to .model file (required for --model osaka24)")
    ap.add_argument("--ani-info", default=None,
                    help="Path to ani-1xnr.info file (required for --model ani). "
                         "e.g. /mfs/io/groups/nielsen/Cary/MLIP/ani-1xnr/model/ani-1xnr.info")
    ap.add_argument("--omol-path",
                    default="/mfs/io/groups/nielsen/Cary/MLIP/model_cache/MACE-omol-0-extra-large-1024.model",
                    help="Local MACE-omol checkpoint (avoids ~/.cache download)")
    ap.add_argument("--uma-path", default=None,
                    help="UMA checkpoint path or registered name (required for --model uma)")
    # >>> ZBL SHORT-RANGE WALL. MACE-Osaka24 collapsed an O-H contact to
    # 0.096 A and MACE-OFF23 to 0.062 A, both returning large NEGATIVE energies
    # where true nuclear repulsion is strongly positive: the learned surface is
    # unbounded below. ZBL restores a floor. It is additive, pairwise and
    # EXACTLY ZERO beyond r_cov_i + r_cov_j, so a --zbl none run and a --zbl
    # mace run are bit-identical everywhere except where atoms are already too
    # close. That is what makes the pair a controlled comparison rather than
    # two different force fields.
    ap.add_argument("--zbl", default="none",
                    choices=["none", "mace", "textbook"],
                    help="short-range repulsive wall; 'mace' reproduces MACE's "
                         "native ZBLBasis term exactly, 'textbook' is ~14x "
                         "stronger at 0.7 A and is the overdose control")
    ap.add_argument("--zbl-scale", type=float, default=1.0,
                    help="multiplier on the ZBL energy; dose for the ladder")
    ap.add_argument("--zbl-p", type=int, default=6,
                    help="envelope exponent, 6 matches MACE")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"],
            help="Compute device (default cuda). Pass cpu to reproduce the original CPU run.")
    ap.add_argument("--box", default="nm_box.xyz")
    ap.add_argument("--outdir", default="md_run")
    ap.add_argument("--eq-T", type=float, default=300.0,
                    help="Equilibration temperature in K (default 300)")
    ap.add_argument("--eq-ps", type=float, default=2.0,
                    help="Equilibration duration in ps (default 2.0)")
    ap.add_argument("--prod-T", type=float, default=3000.0,
                    help="Production temperature in K (default 3000)")
    ap.add_argument("--prod-ps", type=float, default=10.0,
                    help="Production duration in ps (default 10.0)")
    ap.add_argument("--dt-fs", type=float, default=0.5,
                    help="Timestep in fs. Use 0.25 for stability at high T")
    ap.add_argument("--friction-fs", type=float, default=0.02,
                    help="Langevin friction in fs^-1 (default 0.02 = tau 50 fs)")
    ap.add_argument("--thermostat", choices=["langevin", "berendsen"],
                    default="langevin",
                    help="Thermostat. 'berendsen' matches Han et al. (tau=200 fs).")
    ap.add_argument("--taut-fs", type=float, default=200.0,
                    help="Berendsen coupling time in fs (default 200, the paper's value)")
    ap.add_argument("--log-interval", type=int, default=100,
                    help="Save frame every N steps (default 100)")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for reproducible velocity init and Langevin noise. "
                         "If omitted, uses non-deterministic seeding.")
    args = ap.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"\nRandom seed: {args.seed}")
    
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    print(f"Output directory: {outdir.resolve()}")
    print(f"Model:            {args.model}")
    print(f"ZBL:              {args.zbl}"
          + (f"  (scale {args.zbl_scale}, p {args.zbl_p})"
             if args.zbl != "none" else ""))
    print(f"Device:           {args.device}")
    print(f"Timestep:         {args.dt_fs} fs")
    print(f"Friction:         {args.friction_fs} fs^-1 (tau = {1.0/args.friction_fs:.1f} fs)")

    # GPU sanity report
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but torch.cuda.is_available() "
                             "is False. Check the conda env / node allocation.")
        print(f"  CUDA device:    {torch.cuda.get_device_name(0)}")
        print(f"  Compute cap:    {torch.cuda.get_device_capability(0)}")

    # Load prebuilt box
    print(f"\nLoading box from {args.box}")
    atoms = read(args.box)
    atoms.set_pbc([True, True, True])
    print(f"  Atoms: {len(atoms)}")
    print(f"  Cell:  {atoms.cell.lengths()}")
    assert atoms.cell.volume > 0, "Cell volume is zero — box has no unit cell set."

    # >>> RECORD WHAT THIS RUN IS, BEFORE IT COSTS ANYTHING.
    # Written here rather than at the end: a run that dies in the first force
    # call is exactly the run whose provenance someone will want, and a
    # manifest written on success only is a manifest missing from every
    # failure. See run_manifest.py for what four separate provenance mistakes
    # this is meant to prevent.
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from run_manifest import write_manifest
    write_manifest(outdir, args, atoms)

    # Setup calculator
    print(f"\nSetting up calculator...")
    calc = setup_calculator(args.model, osaka_path=args.osaka_path,
                            ani_info=args.ani_info, uma_path=args.uma_path,
                            omol_path=args.omol_path, device=args.device)

    # >>> WRAP AFTER setup_calculator, NOT INSIDE IT. The wall is an ADDITIVE
    # calculator around whatever the model branch returned, so the same three
    # lines work for MACE, UMA and ANI alike. Patching each branch would make
    # the wall model-specific, which is the one thing it must not be if the
    # cross-model comparison is to mean anything.
    if args.zbl != "none":
        # >>> os IS NOT A MODULE-LEVEL IMPORT IN THIS FILE. It is imported
        # locally inside the uma branch of setup_calculator and nowhere else,
        # so referring to it here raised NameError. Import both locally.
        import sys as _s, os as _o
        _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
        from zbl_patch import ZBLCalculator
        calc = ZBLCalculator(calc, flavor=args.zbl, p=args.zbl_p,
                             scale=args.zbl_scale)
        print(f"  wrapped in {calc!r}")

    atoms.calc = calc

    if args.model in ("omol", "uma"):
        atoms.info["charge"] = 0
        atoms.info["spin"] = 1

    e_initial = atoms.get_potential_energy()
    print(f"  Initial E_pot:      {e_initial:.4f} eV")
    print(f"  Initial E_pot/atom: {e_initial / len(atoms):.4f} eV/atom")

    # Initialize velocities at equilibration T
    print(f"\nInitializing Maxwell-Boltzmann velocities at {args.eq_T} K")
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.eq_T)
    Stationary(atoms)

    # --- Equilibration ---
    n_eq_steps = int(args.eq_ps * 1000 / args.dt_fs)
    ok, steps = run_nvt(
        atoms,
        T_target=args.eq_T,
        n_steps=n_eq_steps,
        dt_fs=args.dt_fs,
        friction_fs=args.friction_fs,
        log_interval=args.log_interval,
        traj_path=outdir / "equilibration.traj",
        log_path=outdir / "equilibration.log",
        e_initial=e_initial,
        label=f"=== Equilibration: {args.eq_ps} ps at {args.eq_T} K "
              f"({n_eq_steps} steps) ===",
        thermostat=args.thermostat,
        taut_fs=args.taut_fs,
    )

    if not ok:
        print("\n!!! Equilibration did not complete cleanly — stopping.")
        write(str(outdir / "final.xyz"), atoms)
        return

    write(str(outdir / "equilibrated.xyz"), atoms)
    print(f"\n  Equilibration complete. Saved: {outdir / 'equilibrated.xyz'}")

    # --- Production ---
    n_prod_steps = int(args.prod_ps * 1000 / args.dt_fs)
    ok, steps = run_nvt(
        atoms,
        T_target=args.prod_T,
        n_steps=n_prod_steps,
        dt_fs=args.dt_fs,
        friction_fs=args.friction_fs,
        log_interval=args.log_interval,
        traj_path=outdir / "production.traj",
        log_path=outdir / "production.log",
        e_initial=e_initial,
        label=f"=== Production: {args.prod_ps} ps at {args.prod_T} K "
              f"({n_prod_steps} steps) ===",
        thermostat=args.thermostat,
        taut_fs=args.taut_fs,      
    )

    write(str(outdir / "final.xyz"), atoms)

    t_done_ps = steps * args.dt_fs / 1000.0
    print(f"\n--- Summary ---")
    print(f"  Status:         {'completed' if ok else 'BLEW UP'}")
    print(f"  Steps done:     {steps} / {n_prod_steps}")
    print(f"  Time simulated: {t_done_ps:.3f} ps")
    print(f"  Trajectory:     {outdir / 'production.traj'}")
    print(f"  Log:            {outdir / 'production.log'}")


if __name__ == "__main__":
    main()
