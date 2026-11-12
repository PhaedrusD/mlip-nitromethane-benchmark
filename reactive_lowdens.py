#!/usr/bin/env python
"""
reactive_lowdensity.py
=====================================================================
Runs the low-density / single-molecule reactive MD that feeds BOTH
  * point 2  (spin x density product slate)   -- via --mode box   or single
  * point 3  (unimolecular first-passage rate) -- via --mode single (replicates)
so the two analyses come out of one consistent trajectory set.

Modes
-----
  --mode single : one nitromethane in a large cell; run MANY replicates over a
                  grid of temperatures. This is the clean unimolecular regime
                  for first_passage.py (no bimolecular interference) and also
                  gives the unimolecular channel branching.
  --mode box    : N nitromethane at a chosen (low) density; run a few replicates
                  per temperature for the Han-style multimolecular product slate
                  where H2O/CO2/N2/NH3 chemistry can occur. Run this at low
                  density AND at 1.97 to expose vs. mask the spin/BDE artifacts.

Spin (Sterling's experiment)
----------------------------
  UMA and MACE-OMOL take integer charge/spin. --spin 1 is the default singlet
  (what Sterling asked us to run); --spin 3 gives the triplet for the
  lower-envelope contrast. Spin-agnostic models (off23, osaka24, ani) ignore it.

Thermostat caveat
-----------------
  Barrier-crossing RATES depend on thermostat friction (Kramers). For rate work
  (--mode single) the default is Langevin (canonical, ergodic) with a documented
  friction; for Han-matching product slates (--mode box) the default is Berendsen
  tau=200 fs as in Han et al. Keep friction FIXED across models when comparing.

NOTE: setup_calculator() below mirrors the conventions from the working
run_reactive_md.py (charge/spin on atoms.info for omol/uma, float64 option,
per-model model paths). If your tested copy differs, drop it in verbatim --
that is the only function that touches model internals.
"""

from __future__ import annotations
import argparse, os, json
import numpy as np
from ase import Atoms
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.io import read as ase_read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen
from ase import units

# molar mass of CH3NO2 (g/mol) for density -> box conversion
M_NM = 61.04
NA = 6.02214076e23


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def build_nitromethane():
    """Single CH3NO2, order C N O O H H H."""
    return Atoms("CNOOHHH", positions=[
        [0.000, 0.000, 0.000],
        [0.000, 0.000, 1.489],
        [1.028, 0.000, 2.110],
        [-1.028, 0.000, 2.110],
        [0.512, 0.887, -0.363],
        [0.512, -0.887, -0.363],
        [-1.024, 0.000, -0.363],
    ])


def box_length_for_density(n_mol, density_gcc):
    """Cubic side (A) giving n_mol nitromethane at density_gcc."""
    mass_g = n_mol * M_NM / NA
    vol_cm3 = mass_g / density_gcc
    vol_A3 = vol_cm3 * 1.0e24
    return vol_A3 ** (1.0 / 3.0)


def build_box(n_mol, density_gcc, rng, min_sep=2.2, pad=1.0):
    """
    Place n_mol randomly-oriented nitromethanes on a jittered grid in a cubic
    cell at the target density, rejecting placements whose COM is closer than
    min_sep+... to an existing molecule. At low density this is trivial; at
    1.97 it still succeeds because we seed on a grid.
    """
    L = box_length_for_density(n_mol, density_gcc)
    base = build_nitromethane()
    base.translate(-base.get_center_of_mass())

    ncell = int(np.ceil(n_mol ** (1.0 / 3.0)))
    spacing = L / ncell
    slots = [(i, j, k) for i in range(ncell)
             for j in range(ncell) for k in range(ncell)]
    rng.shuffle(slots)

    box = Atoms(cell=[L, L, L], pbc=True)
    coms = []
    for idx in range(n_mol):
        i, j, k = slots[idx]
        mol = base.copy()
        # random rotation
        phi, theta, psi = rng.uniform(0, 360, 3)
        mol.euler_rotate(phi, theta, psi, center="COM")
        com = (np.array([i, j, k]) + 0.5) * spacing \
            + rng.uniform(-0.15, 0.15, 3) * spacing
        mol.translate(com)
        box += mol
        coms.append(com)
    box.wrap()
    return box


# ---------------------------------------------------------------------------
# Calculator  --  matches the working loader from run_reactive_md.py.
# Each model gets an explicit path arg (omol/uma default to a named pretrained
# source if no path is given). charge/spin are attached to atoms.info at run
# time via attach_spin(); nothing to do here except build the calc.
# ---------------------------------------------------------------------------
def setup_calculator(model, device="cuda", dtype="float64", charge=0, spin=1,
                     off23_path=None, osaka_path=None, omol_path=None,
                     uma_path=None, ani_path=None):
    model = model.lower()

    if model == "off23":
        from mace.calculators import mace_off
        if off23_path:
            from mace.calculators import MACECalculator
            return MACECalculator(model_paths=[off23_path], device=device,
                                  default_dtype="float64")
        return mace_off(model="medium", device=device, default_dtype="float64")

    if model in ("osaka", "osaka24"):
        from mace.calculators import MACECalculator
        try:
            return MACECalculator(model_paths=[osaka_path], device=device,
                                  default_dtype="float64")
        except TypeError:
            return MACECalculator(model_path=osaka_path, device=device,
                                  default_dtype="float64")

    if model == "omol":
        from mace.calculators import mace_omol
        src = omol_path if omol_path else "extra_large"
        return mace_omol(model=src, device=device, default_dtype="float32")

    if model == "uma":
        from fairchem.core import FAIRChemCalculator
        if not uma_path:
            raise ValueError("--uma-path required for uma")
        if os.path.exists(uma_path):
            from fairchem.core.units.mlip_unit import load_predict_unit
            pred = load_predict_unit(path=uma_path, device=device)
        else:
            from fairchem.core import pretrained_mlip
            pred = pretrained_mlip.get_predict_unit(uma_path, device=device)
        return FAIRChemCalculator(pred, task_name="omol")

    if model == "ani":
        # Reactive ANI-1xnr via NeuroChem .info file (NOT torchani.models.ANI1x).
        # Matches run_reactive_md.py. Native float32.
        import torch
        import torchani.neurochem as nc
        if not ani_path:
            raise ValueError("--ani-path (the ani-1xnr.info file) required for ani")
        return nc.load_model_from_info_file(ani_path).to(torch.device(device)).ase()

    raise ValueError(f"unknown model {model!r}")


def attach_spin(atoms, model, charge=0, spin=1):
    """Set charge/spin for the spin-aware models; no-op otherwise."""
    if model.lower() in ("omol", "uma"):
        atoms.info["charge"] = int(charge)
        atoms.info["spin"] = int(spin)
    return atoms


# ---------------------------------------------------------------------------
# One MD run
# ---------------------------------------------------------------------------
def run_one(atoms, calc, T, dt_fs, n_steps, traj_path,
            thermostat="langevin", friction_ps=1.0, tau_fs=200.0,
            log_stride=40, init_temp_factor=1.0, seed=0,
            stop_on_fragment=False, scale=1.2,
            frag_check_fs=50.0, frag_persist=3, frag_buffer_fs=500.0):
    rng = np.random.default_rng(seed)
    atoms.calc = calc
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=T * init_temp_factor,
        rng=np.random.RandomState(seed))
    atoms.set_momenta(atoms.get_momenta()
                      - atoms.get_momenta().mean(axis=0))  # zero COM drift

    dt = dt_fs * units.fs
    if thermostat == "langevin":
        friction = friction_ps / 1000.0 / units.fs      # 1/ps -> 1/(ASE time)
        dyn = Langevin(atoms, dt, temperature_K=T, friction=friction,
                       fixcm=False,
                       rng=np.random.RandomState(seed + 1))
    elif thermostat == "berendsen":
        dyn = NVTBerendsen(atoms, dt, temperature_K=T,
                           taut=tau_fs * units.fs)
    else:
        raise ValueError(thermostat)

    traj = Trajectory(traj_path, "w", atoms)
    dyn.attach(traj.write, interval=log_stride)

    if not stop_on_fragment:
        dyn.run(n_steps)
        traj.close()
        return traj_path

    # Early stop: run in chunks, stop once fragmentation has PERSISTED for
    # `frag_persist` consecutive checks, then run a short buffer and quit.
    # For first-passage we only need up to the first sustained break.
    from species_analysis import adjacency, connected_components
    chunk = max(int(round(frag_check_fs / dt_fs)), 1)
    buffer_steps = int(round(frag_buffer_fs / dt_fs))
    done = 0
    consec = 0
    frag_step = None
    while done < n_steps:
        step = min(chunk, n_steps - done)
        dyn.run(step)
        done += step
        fragmented = len(connected_components(adjacency(atoms, scale))) >= 2
        consec = consec + 1 if fragmented else 0
        if frag_step is None and consec >= frag_persist:
            frag_step = done
        if frag_step is not None and (done - frag_step) >= buffer_steps:
            break
    traj.close()
    return traj_path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["single", "box", "dilute"], required=True,
                   help="single: 1 molecule first-passage; box: N-mol product slate; "
                        "dilute: Messerly-style ultra-dilute rate (population decay)")
    p.add_argument("--model", required=True,
                   choices=["off23", "osaka24", "ani", "omol", "uma"])
    p.add_argument("--temps", type=float, nargs="+",
                   default=[2000, 2500, 3000, 3500])
    p.add_argument("--replicates", type=int, default=200,
                   help="single: replicates/T (200+); box: 3-10; dilute: 8 (Messerly)")
    p.add_argument("--n-mol", type=int, default=16,
                   help="box/dilute mode: molecules per cell (Messerly dilute = 125)")
    p.add_argument("--density", type=float, default=0.10,
                   help="g/cc for box mode; run low AND 1.97")
    p.add_argument("--dilute-density", type=float, default=1e-4,
                   help="g/cc for dilute mode (Messerly CH4 = 1e-4; independent "
                        "unimolecular reactors, no bi/termolecular interference)")
    p.add_argument("--steps", type=int, default=600000,
                   help="0.25 fs * 600000 = 150 ps (Han)")
    p.add_argument("--dt", type=float, default=0.25)
    p.add_argument("--log-stride", type=int, default=40, help="frames every N steps")
    p.add_argument("--thermostat", choices=["langevin", "berendsen"],
                   default=None, help="default: single->langevin, box->berendsen")
    p.add_argument("--friction", type=float, default=1.0, help="Langevin 1/ps")
    p.add_argument("--tau", type=float, default=200.0, help="Berendsen tau (fs)")
    p.add_argument("--spin", type=int, default=1, help="1 singlet (Sterling), 3 triplet")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float64")
    p.add_argument("--single-cell", type=float, default=30.0,
                   help="cubic cell (A) for the lone molecule in single mode")
    p.add_argument("--stop-on-fragment", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="stop a run after first sustained fragmentation + buffer "
                        "(default: on for single, off for box)")
    p.add_argument("--frag-buffer", type=float, default=500.0,
                   help="fs to keep running after fragmentation (single mode)")
    p.add_argument("--start-xyz", default=None,
                   help="read the starting box from a prepared file instead of "
                        "building one. REQUIRED for crystal runs and strongly "
                        "preferred at high density: build_box() places "
                        "molecules randomly and cannot make a crystal, while "
                        "prep_dense_box.py compresses in stages to stay "
                        "clash-free. --n-mol and --density are ignored when "
                        "this is given; the file's cell defines both.")
    p.add_argument("--zbl", choices=["none", "mace", "textbook"], default="none",
                   help="add a ZBL screened-Coulomb wall on top of the model. "
                        "Zero beyond the covalent contact radius, so --zbl none "
                        "vs --zbl mace is a controlled pair. See zbl_patch.py.")
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--outdir", default="lowdens_runs")
    p.add_argument("--off23-path", default=None)
    p.add_argument("--osaka-path", default=None)
    p.add_argument("--omol-path", default=None, help="path or omit for 'extra_large'")
    p.add_argument("--uma-path", default=None, help="path or a pretrained name")
    p.add_argument("--ani-path", default=None)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    thermostat = args.thermostat or ("langevin" if args.mode == "single"
                                     else "berendsen")
    init_factor = 1.0 if args.mode == "single" else 2.0   # Han: box init at 2T
    # early-stop only makes sense for the lone-molecule first-passage mode
    stop_frag = (args.stop_on_fragment if args.stop_on_fragment is not None
                 else (args.mode == "single"))

    calc = setup_calculator(args.model, device=args.device, dtype=args.dtype,
                            charge=args.charge, spin=args.spin,
                            off23_path=args.off23_path, osaka_path=args.osaka_path,
                            omol_path=args.omol_path, uma_path=args.uma_path,
                            ani_path=args.ani_path)

    # >>> OPTIONAL ZBL SHORT-RANGE WALL (added 2026-08-10) <<<
    # Purely additive, and identically ZERO beyond r_cov(i)+r_cov(j), so a run
    # with --zbl none and one with --zbl mace are the same simulation everywhere
    # except inside a covalent contact. That makes the pair a CONTROLLED
    # EXPERIMENT rather than two unrelated runs.
    #
    # The motivating case is MACE-OFF23, which diverged at ~1.5-1.9 ps with
    # T -> 4.7e25 K. A smaller timestep alone only tests "was it integration
    # error"; adding ZBL also tests "was it a hole in the repulsive wall", which
    # is the more likely cause and the one we can actually fix. Run both and the
    # answer is unambiguous:
    #   smaller dt alone survives      -> integration
    #   only dt + ZBL survives         -> the wall
    #   neither survives               -> something else; look elsewhere
    if args.zbl != "none":
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        _sys.path.insert(0, os.path.abspath(".."))
        from zbl_patch import ZBLCalculator
        calc = ZBLCalculator(calc, flavor=args.zbl)
        print(f"  ZBL wall active: flavor={args.zbl}")

    manifest = []
    seed = args.seed0
    for T in args.temps:
        nrep = args.replicates if args.mode == "single" else max(args.replicates, 1)
        for rep in range(nrep):
            seed += 1
            if args.mode == "single":
                atoms = build_nitromethane()
                atoms.set_cell([args.single_cell] * 3)
                atoms.center()
                atoms.pbc = False
            elif args.mode == "dilute":
                # Messerly protocol: many nitromethanes at ultra-low density so
                # each is an independent unimolecular reactor. Rate comes from
                # the population decay N(t)=N0 exp(-k t), not per-molecule events.
                atoms = build_box(args.n_mol, args.dilute_density,
                                  np.random.default_rng(seed))
            elif args.start_xyz:
                # >>> PREPARED BOX FROM FILE (added 2026-08-12) <<<
                # build_box() places molecules randomly on a jittered grid. That
                # is fine at low density, but it CANNOT produce a crystal, and
                # at 1.97 g/cc a randomly packed box is not the same object as
                # one compressed in stages by prep_dense_box.py -- the stagewise
                # compression is what removes the clashes that would otherwise
                # sit right in the pathological 0.3-0.6 A region we are studying.
                # Starting from a clash-free prepared box (nm_box_1p97_clean.xyz,
                # nm_box_1p97_clean_crystal.xyz) means any close contact that
                # appears during the run was PRODUCED BY THE MODEL, not by the
                # box builder -- which is the entire claim.
                atoms = ase_read(args.start_xyz)
                if atoms.cell.rank < 3:
                    raise SystemExit(
                        f"{args.start_xyz} has no periodic cell; a condensed-"
                        f"phase run needs one. Check the file was written with "
                        f"its lattice.")
                atoms.pbc = True
            else:
                atoms = build_box(args.n_mol, args.density,
                                  np.random.default_rng(seed))
            attach_spin(atoms, args.model, args.charge, args.spin)
            if args.start_xyz:
                # density comes from the FILE, not the flag -- record what was
                # actually simulated, and say so if the flag disagrees
                vol = atoms.get_volume()
                mass = float(sum(atoms.get_masses()))
                dens_out = mass / vol / 0.6022140857
                if rep == 0 and T == args.temps[0]:
                    print(f"  start box: {args.start_xyz}")
                    print(f"    {len(atoms)} atoms, {dens_out:.3f} g/cc, "
                          f"cell {np.round(atoms.cell.lengths(), 2)}")
                    if abs(dens_out - args.density) > 0.02:
                        print(f"    NOTE --density {args.density} is IGNORED; "
                              f"the file's cell sets {dens_out:.3f} g/cc")
            else:
                dens_out = (None if args.mode == "single"
                            else (args.dilute_density if args.mode == "dilute"
                                  else args.density))
            tag = f"{args.model}_{args.mode}_T{int(T)}_s{args.spin}_r{rep:03d}"
            tpath = os.path.join(args.outdir, tag + ".traj")
            run_one(atoms, calc, T, args.dt, args.steps, tpath,
                    thermostat=thermostat, friction_ps=args.friction,
                    tau_fs=args.tau, log_stride=args.log_stride,
                    init_temp_factor=init_factor, seed=seed,
                    stop_on_fragment=stop_frag, frag_buffer_fs=args.frag_buffer)
            manifest.append({"path": tpath, "model": args.model, "mode": args.mode,
                             "T": T, "spin": args.spin, "rep": rep,
                             "density": dens_out, "n_mol": args.n_mol,
                             "dt_fs": args.dt, "log_stride": args.log_stride})
    with open(os.path.join(args.outdir,
              f"manifest_{args.model}_{args.mode}_s{args.spin}.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {len(manifest)} trajectories to {args.outdir}")


if __name__ == "__main__":
    main()