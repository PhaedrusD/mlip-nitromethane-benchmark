"""
bde_scan.py
==================================================
Adiabatic dissociation scan for ANI-1xnr, MACE-OFF23, and MACE-Osaka24 on any molecule, for any bond.

Usage: Modify values in the molecule-specific section and then run 'python bde_scan.py' with arguments if desired.

Note: Make sure you have the models downloaded and the necessary libraries installed, and
    make sure to change the file paths to match your system.
"""

import argparse
import json
import os
import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from ase.constraints import FixInternals
from ase.visualize import view
from ase.io import write

# >>> CHANGED 2026-08-04: these were TOP-LEVEL imports, which meant the whole
# script failed to load in any environment missing torchani OR fairchem. Since
# mace and fairchem cannot coexist in one conda env, that made cross-env runs
# impossible. They are now imported lazily inside the model getters.
import torch

import matplotlib.pyplot as plt

from ase.geometry import get_distances
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import natural_cutoffs, NeighborList, neighbor_list
import traceback

EV_TO_KCAL = 23.0609
DEVICE = "cpu"

# ---- model variants, aligned to the nitromethane paper (2026-08-04) ---------
# These MUST match files/bond_pull_scan.py or the rigid rescans are not
# comparable with seam_scan/pull_*.csv.
OFF23_VARIANT = "medium"     # paper uses medium; Johnny's default was "large"
OMOL_DTYPE    = "float32"    # paper uses float32; Johnny's default was float64
# Osaka24: see the --osaka-model default below (medium, not the large in-repo file)  # Change to "cuda" for GPU if available

# Dictionary to hold values for the final graph
results = {}

# >>> CHANGED 2026-08-04: send_email pulls in python-dotenv and needs a
# configured .env with mail credentials. Nothing in the science depends on it,
# so it is now optional and OFF by default. Set SEND_EMAIL=True (and install
# python-dotenv) to re-enable.
try:
    import send_email
    _HAVE_EMAIL = True
except Exception as _e:
    send_email = None
    _HAVE_EMAIL = False

SEND_EMAIL = False # If True, confirmation email is sent upon completion with the following data:
process_name = "SET PROCESS NAME IF USING" # Name of the process to be included in the email

# [from main, Jonny's peroxide charge/spin investigation] If True, a custom
# charge and spin are stamped onto atoms.info in build_molecule(). Only the
# spin-aware models (OMOL, UMA, AIMNet2-NSE) read them; the rest ignore them.
SET_CUSTOM_CHARGE_SPIN = False
CUSTOM_CHARGE = 5
CUSTOM_SPIN = 1


''' ---------- Molecule-Specific Values ---------- '''

# CHANGE ONLY THE FOLLOWING WHEN TRYING OUT NEW MOLECULES:
NAME = "Nitromethane"
EXP_BDE_KCAL = 60.0  # Expected BDE Value, D0 at 298 K, kcal/mol
SMILES_STRING = "C[N+](=O)[O-]"
BOND_TO_SCAN = "C-N" # As an example, format like this: "C-N"
SCAN_NUMBER = 2 # Use this when scanning multiple of the same bond in a molecule (prevents graphs/results from overriding one another), set to 0 when not in use
INDICES_OF_BOND = [0, 1] # Can be found through ase gui, run 'python bde_scan.py --findindices True' and highlight bonding atoms
# Note: for INDICES_OF_BOND, the first index is fixed and the second is moving. Make sure INDICES_OF_BOND has exactly 2 inputs (both within range of molecule)!
USE_GPAW = False # >>> CHANGED: GPAW is a slow DFT calculator; the rigid
                 # subset does not need it. run() overrides this per-scan anyway.
EXTEND_GRAPH = True  # If True, graph will extend to 7.0 Å
HYPER_EXTEND_GRAPH = False # Extend graph to 9.0 Å
# [from main] Module-level default for rigid mode. --rigid on the command line
# still works and wins; this just sets the default so run() and bare invocations
# behave like main's version did.
DO_RIGID = True

''' ---------------------------------------------- '''


# Initializes starting values and runs main (used when starting script from ANOTHER PYTHON FILE)
def run(name, exp_bde_kcal, smiles_string, bond_to_scan, scan_number, indices_of_bond, use_gpaw, extend_graph, hyper_extend_graph, do_rigid=None):

    global NAME, EXP_BDE_KCAL, SMILES_STRING, BOND_TO_SCAN, SCAN_NUMBER, INDICES_OF_BOND, USE_GPAW, EXTEND_GRAPH, HYPER_EXTEND_GRAPH # So the global variables are used
    NAME = name
    EXP_BDE_KCAL = exp_bde_kcal
    SMILES_STRING = smiles_string
    BOND_TO_SCAN = bond_to_scan
    SCAN_NUMBER = scan_number
    INDICES_OF_BOND = indices_of_bond
    USE_GPAW = use_gpaw
    EXTEND_GRAPH = extend_graph
    HYPER_EXTEND_GRAPH = hyper_extend_graph

    global results
    results = {} # Resetting results dictionary so data doesn't leak between iterations.

    try:
        main()
        return True
    except Exception:
        return False


# Builds the molecule with rdkit's Chem module
def build_molecule(calc_name):
    """Build molecule using RDKit, return as ASE Atoms."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(SMILES_STRING)
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
    atoms = Atoms(symbols=symbols, positions=positions)
    if calc_name == "GPAW":
        atoms.center(vacuum=10.0) # If building for GPAW (DFT Calculator), add vacuum around molecule
    atoms.set_pbc(False)
    if SET_CUSTOM_CHARGE_SPIN:                      # [from main]
        atoms.info['charge'] = CUSTOM_CHARGE
        atoms.info['spin'] = CUSTOM_SPIN
    return atoms


def main():

    # Tell user which molecule and bond are being scanned
    print("\n----------------------------------------------")
    print(f"Running {BOND_TO_SCAN} dissociation scan on {NAME}.")
    print("-----------------------------------------------\n")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--json', default=None)
    ap.add_argument('--graph', default=None)
    ap.add_argument('--info', default='ani-1xnr/model/ani-1xnr.info')
    ap.add_argument('--view', action='store_true')
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--rigid', action='store_true')
    ap.add_argument('--findindices', action='store_true')
    ap.add_argument('--fmax', type=float, default=0.05,
                    help='Force convergence threshold in eV/Å (default 0.05)')
    ap.add_argument('--steps', type=int, default=500,
                    help='Max FIRE steps per scan point (default 500)')
    ap.add_argument('--fresh', action='store_true',
                    help="overwrite the results file instead of merging into it")
    ap.add_argument('--models', default='auto',
                    help="comma list from ani,off23,osaka,omol,uma,gpaw -- or 'all'. "
                         "Default 'omol,uma': the two OMol25-trained models that show the seam.")
    # >>> CHANGED: was "mace-osaka24-large.model" (63.5 MB, shipped in this repo).
    # The paper's Osaka24 results (D = 2.870 eV, cutoff 4.5 A, 65.8 ps collapse)
    # are the MEDIUM model living in ~/ase.
    ap.add_argument("--osaka-model", default="../mace-osaka24-medium.model",
                    help="Path to MACE-Osaka24 model file (.model). If omitted, default path is used.")
    args = ap.parse_args()

    # [merged] --rigid wins if given; otherwise fall back to the DO_RIGID module
    # constant that main's version used. Both mechanisms now drive one flag.
    if not args.rigid and DO_RIGID:
        args.rigid = True

    # >>> ADDED: rigid runs get their own filenames so they never clobber the
    # relaxed results already in the repo.
    tag = "_rigid" if args.rigid else ""
    if args.json is None:
        args.json = f"json_results/scan_results_{NAME}_{BOND_TO_SCAN}{tag}.json"
    if args.graph is None:
        args.graph = f"graphs/bde_scan_{NAME}_{BOND_TO_SCAN}{tag}.png"

    # SCAN_NUMBER is used when scanning multiple of the same bond in a molecule
    if not SCAN_NUMBER == 0:
        args.json = f"json_results/scan_results_{NAME}_{BOND_TO_SCAN}_{SCAN_NUMBER}{tag}.json"
        args.graph = f"graphs/bde_scan_{NAME}_{BOND_TO_SCAN}_{SCAN_NUMBER}{tag}.png"

    # Let user find indices of bonding atoms if necessary
    if args.findindices:
        indices_gui = view(build_molecule("No Calc (ignore this)"))
        print("Highlight the bonding atoms individually to find their indices.")
        print("Then, run the program normally to run the BDE scan.")
        exit()

    # Call other main function to calculate scan results
    run_models(args)

    # Make and save graph:
    plot_graph(args)

    if SEND_EMAIL and _HAVE_EMAIL: # Send completion email if desired
        send_email.send_completion_email(process_name)
    elif SEND_EMAIL and not _HAVE_EMAIL:
        print("  (email requested but send_email unavailable -- pip install python-dotenv)")


# Find moving fragments in a compress/stretch and return their indices
def find_fragment_indices(atoms, idx_fixed, idx_moving):
    """Finds all atoms connected to idx_moving, without crossing into idx_fixed."""
    # Build a neighbor list based on automatic chemical cutoffs
    cutoffs = natural_cutoffs(atoms)
    nl = NeighborList(cutoffs, bothways=True, self_interaction=False)
    nl.update(atoms)
    
    # Standard BFS graph search
    fragment = set([idx_moving])
    queue = [idx_moving]
    
    while queue:
        current = queue.pop(0)
        indices, offsets = nl.get_neighbors(current)
        for nbr in indices:
            # Crucial: Do not cross back into the fixed atom side of the bond
            if nbr != idx_fixed and nbr not in fragment:
                fragment.add(nbr)
                queue.append(nbr)
                
    return list(fragment)


# Stretch the fragments returned by the 'find_fragment_indices' function
def stretch_any_bond(atoms, idx_fixed, idx_moving, target_r):
    """
    Stretches the bond between idx_fixed and idx_moving to target_r.
    Rigidly translates the entire fragment attached to idx_moving.
    """
    # 1. Get the bond vector (points from fixed atom to moving atom)
    axis = atoms.get_distance(idx_fixed, idx_moving, vector=True)
    current_r = np.linalg.norm(axis)
    axis_unit = axis / current_r # unit vector
    shift = (target_r - current_r) * axis_unit

    # 2. Find all atoms in the moving fragment
    moving_fragment_indices = find_fragment_indices(atoms, idx_fixed, idx_moving)

    # 3. Apply the vectorized shift to the entire fragment
    atoms.positions[moving_fragment_indices] += shift


# This is the function that runs the actual scan
def run_scan(calc, calc_name, r_values, args):

    """ Run an adiabatic relaxed scan of the bond with the given calculator. """

    print(f"\n{'=' * 60}")
    print(f"Running model: {calc_name}")
    print(f"{'=' * 60}")

    atoms = build_molecule(calc_name)

    # Load in proper indices for the atoms in the bond that is being scanned
    index_a = INDICES_OF_BOND[0]
    index_b = INDICES_OF_BOND[1]

    atoms.calc = calc

    # Initial full optimization (no constraint)
    print("Initial unconstrained optimization...")
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=0.01, steps=400)
    r_eq = atoms.get_distance(index_a, index_b)
    e_eq = atoms.get_potential_energy()
    print(f"  r_eq({BOND_TO_SCAN}) = {r_eq:.4f} Å")
    print(f"  E_eq       = {e_eq:.4f} eV")

    energies = []
    trajectories = []

    print(f"\n{'r(C-N) / Å':>12} {'E / eV':>14} {'ΔE / kcal/mol':>16}  {'Num of Steps':>16}")
    print("-" * 62)

    # Show molecule to users if they want to see it:
    if args.view:
        pre_gui = view(atoms)

    for r in r_values:
        # Reset to equilibrium geometry and clear constraints
        atoms.set_constraint()

        # FOR DEBUGGING
        if args.debug:
            moving_fragment_indices = find_fragment_indices(atoms, index_a, index_b)
            print(f"DEBUG:   idx_fixed = {index_a}   fragment_moving_idx = {moving_fragment_indices}")

        # Initial guess: translate NO2 to target distance
        stretch_any_bond(atoms, index_a, index_b, r)
        
        if not args.rigid:

            # ---------------- Dynamic Internal Constraint Manager ----------------

            cutoffs = natural_cutoffs(atoms, mult=1.10)
            nl_i, nl_j = neighbor_list('ij', atoms, cutoff=cutoffs)
        
            # Extract all atoms bonded to index_a, excluding the leaving atom (index_b)
            index_a_neighbors = [
                nj for ni, nj in zip(nl_i, nl_j) if ni == index_a and nj != index_b
            ]

            bonds = [[r, (index_a, index_b)]]
            angles = None
            dihedrals = None

            if len(index_a_neighbors) > 0:
                # Anchor 1: First available neighbor attached to index_a
                anchor_1 = index_a_neighbors[0]
            
                # Lock Angle: anchor_1 - index_a - index_b
                current_angle = atoms.get_angle(anchor_1, index_a, index_b)
                angles = [[current_angle, (anchor_1, index_a, index_b)]]

                # Look deeper for a Dihedral anchor to freeze 3D swinging
                anchor_1_neighbors = [
                    nj for ni, nj in zip(nl_i, nl_j) if ni == anchor_1 and nj not in [index_a, index_b]
                ]
            
                if len(anchor_1_neighbors) > 0:
                    # Anchor 2: A neighbor of neighbor_1 to map the dihedral path
                    anchor_2 = anchor_1_neighbors[0]
                
                    # Lock Dihedral:
                    current_dihedral = atoms.get_dihedral(anchor_2, anchor_1, index_a, index_b)
                    dihedrals = [[current_dihedral, (anchor_2, anchor_1, index_a, index_b)]]

            # Construct and apply the generalized constraint
            constraints = FixInternals(
                bonds=bonds, 
                angles_deg=angles,        
                dihedrals_deg=dihedrals   
            )
            atoms.set_constraint(constraints)

            # --------------------------------------------------------------------

            # atoms.set_constraint(FixBondLengths(pairs=[[index_a, index_b]], tolerance=1e-6)) # OLD CONSTRAINTS

            opt = BFGS(atoms, logfile=None)
            try:
                opt.run(fmax=args.fmax, steps=args.steps)
                num_of_steps = opt.get_number_of_steps()
                e = atoms.get_potential_energy()
                de_kcal = (e - e_eq) * EV_TO_KCAL
                print(f"  {r:10.3f}   {e:12.4f}   {de_kcal:14.2f}  {num_of_steps:14d}")
            except Exception as exc:
                print(f"  {r:10.3f}   FAILED: {exc}")
                e = np.nan
        else: # Run rigid scan
            e = atoms.get_potential_energy()
            de_kcal = (e - e_eq) * EV_TO_KCAL
            print(f"  {r:10.3f}   {e:12.4f}   {de_kcal:14.2f}")

        energies.append(e)
        trajectories.append(atoms.copy())
    
    # Show molecule to users if they want to see it:
    if args.view:
        post_gui = view(atoms)

    # Save trajectory for visualization (add number if doing multiple)
    # >>> FIXED 2026-08-05 (Cary). The json filenames already carry a "_rigid"
    # tag (see main()), but these trajectory filenames did NOT -- so a rigid run
    # silently OVERWROTE the relaxed .xyz trajectories from a previous run.
    # 30 of Johnny's relaxed trajectories were clobbered this way locally before
    # anyone noticed. Same tag here, same behaviour as the json.
    _tag = "_rigid" if getattr(args, "rigid", False) else ""
    if SCAN_NUMBER == 0:
        traj_path = f"xyz_output/scan_{calc_name}_{NAME}_{BOND_TO_SCAN}{_tag}.xyz"
    else:
        traj_path = (f"xyz_output/scan_{calc_name}_{NAME}_{BOND_TO_SCAN}"
                     f"_{SCAN_NUMBER}{_tag}.xyz")
    write(str(traj_path), trajectories)
    print(f"\n  Trajectory saved: {traj_path}")

    return np.array(energies), e_eq


def get_ani_1xnr(args):
    import torchani.neurochem as nc
    # Loading ANI-1xnr
    model = nc.load_model_from_info_file(args.info)
    return model.ase()


def get_mace_off23():
    # >>> CHANGED 2026-08-04: was "large". Every MACE-OFF23 result in the
    # nitromethane paper (D = 3.710 eV, cutoff 5.0 A, 1.9 ps MD divergence) is
    # the MEDIUM variant, and only medium/small are cached locally. "large"
    # would silently download a different model and give incomparable numbers.
    from mace.calculators import mace_off
    return mace_off(model=OFF23_VARIANT, device=DEVICE, default_dtype="float64")


def get_mace_osaka24(model_path):
    # MACE-Osaka24 from a local .model file
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=[model_path], device=DEVICE, default_dtype="float64")


def get_mace_omol():
    # Checkpoint matches (extra_large = MACE-omol-0-extra-large-1024.model).
    # >>> CHANGED: dtype was float64; the paper's OMOL runs are float32 and
    # Methods records that. float64 is arguably better but breaks bit-level
    # comparability with seam_scan/pull_omol.csv. Set OMOL_DTYPE to switch.
    from mace.calculators import mace_omol
    return mace_omol(model="extra_large", device=DEVICE, default_dtype=OMOL_DTYPE)


def get_uma():
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    # Intercept torch.load to bypass the weights_only safety restriction
    _original_torch_load = torch.load
    torch.load = lambda *args, **kwargs: _original_torch_load(*args, **{**kwargs, 'weights_only': False})

    # Load UMA small model (since large is too big at 12gb)
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=DEVICE)
    return FAIRChemCalculator(predictor, task_name="omol")


def get_aimnet2(model="aimnet2-nse", charge=0, mult=None, ensemble=True):
    """AIMNet2 family (Isayev lab).   pip install "aimnet[ase]"   -- MIT, on PyPI.

    Imported lazily for the same reason as fairchem: aimnet pins its own torch and
    will not share an env with mace. Run these from the `aimnet` env; the results
    merge into the same JSON.

    MODELS
      aimnet2-nse  open-shell/radicals. Takes SEPARATE alpha and beta charges,
                   ab = 0.5*[q - mult + 1, q + mult - 1], instead of one global
                   spin integer -- exactly the representational fix our ablation
                   says OMOL/UMA lack. Kalita et al., Angew. Chem. 2026, e16763.
      aimnet2-rxn  reactive / TS / NEB. H, C, N, O ONLY, net-neutral ONLY -- both
                   satisfied by nitromethane. Direct competitor to ANI-1xnr.

    >>> THE ENSEMBLE IS NOT LOADED BY DEFAULT <<<
    Each family has FOUR members (0-3) and a short alias silently loads MEMBER 0.
    We were bitten by exactly this with ANI-1xnr, where member 0 was the
    least-broken member and every number quoted from it understated the problem
    AND the fix. ensemble=True averages all four.
    """
    from aimnet.calculators import AIMNet2ASE

    def _one(member=None):
        kw = {"charge": charge}
        if member is not None:
            kw["ensemble_member"] = member
        try:
            c = AIMNet2ASE(model, **kw)
        except TypeError:
            from aimnet.calculators import AIMNet2Calculator
            core = (AIMNet2Calculator(model, ensemble_member=member)
                    if member is not None else AIMNet2Calculator(model))
            c = AIMNet2ASE(core, charge=charge)
        if mult is not None:
            if not hasattr(c, "set_mult"):
                raise RuntimeError(
                    f"{type(c).__name__} has no set_mult(); check the signature with "
                    f"help(AIMNet2ASE) and see isayevlab.github.io/aimnetcentral/calculator/")
            c.set_mult(mult)
        return c

    if not ensemble:
        print(f"  [aimnet2] {model}: MEMBER 0 ONLY -- not the ensemble.")
        return _one()

    members = [_one(i) for i in range(4)]
    print(f"  [aimnet2] {model}: averaging {len(members)} members"
          f"{'' if mult is None else f', mult={mult}'}")

    class _AIMNet2Ensemble(Calculator):
        implemented_properties = ["energy", "free_energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",),
                      system_changes=all_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
            es, fs = [], []
            for c in members:
                a = atoms.copy(); a.calc = c
                es.append(a.get_potential_energy()); fs.append(a.get_forces())
            es = np.asarray(es)
            self.last_std_eV = float(es.std())
            self.results = {"energy": float(es.mean()),
                            "free_energy": float(es.mean()),
                            "forces": np.mean(fs, axis=0)}

    return _AIMNet2Ensemble()


def get_gpaw():
    # Load GPAW (GPAW is DFT-based and used to compare against the MLIP models)
    from gpaw import GPAW
    return GPAW(mode='lcao', basis='dzp', xc='RPBE', spinpol=True, symmetry={'point_group': False}, h=0.20, txt='gpaw_output.txt')


# Note: Make sure the default path under the --osaka-model argument is correct since that's the one that is used.
def run_models(args):

    # Define r values (denser near equilibrium, coarser at dissociation)
    _base = [1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,
             2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,
             3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5]
    if EXTEND_GRAPH:      # [from main] out to 7.0 A -- past every model's cutoff
        r_values = _base + [4.7,4.9,5.1,5.3,5.5,5.7,5.9,6.1,6.3,6.5,6.7,7.0]
    else:
        r_values = _base

    if HYPER_EXTEND_GRAPH:
        r_values = [1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.7,4.9,5.1,5.3,5.5,5.7,5.9,6.1,6.3,6.5,6.7,7.0]

    global results

    # --- GPAW ---
    # Runs first so that errors can be caught quickly
    if USE_GPAW:
        try:
            calc = get_gpaw()
            E, E_eq = run_scan(calc, "GPAW", r_values, args)
            results["GPAW"] = {"r": r_values, "E": ((np.array(E.tolist()) - E_eq) * EV_TO_KCAL).tolist(), "E_eq": E_eq}
        except Exception:
            print("GPAW FAILED:")
            traceback.print_exc()

    # ---- model selection (added 2026-08-04) --------------------------------
    import importlib.util as _ilu
    HAVE_MACE = _ilu.find_spec("mace") is not None
    HAVE_FAIR = _ilu.find_spec("fairchem") is not None
    HAVE_ANI  = _ilu.find_spec("torchani") is not None
    HAVE_AIMNET = _ilu.find_spec("aimnet") is not None

    sel = {m.strip().lower() for m in args.models.split(',')}
    if 'all' in sel:
        sel = {'ani', 'off23', 'osaka', 'omol', 'uma', 'nse', 'rxn'}
    if 'auto' in sel:
        # pick whatever THIS environment can run -- mace and fairchem cannot
        # coexist, so this resolves to the mace family OR uma, never both.
        sel = set()
        if HAVE_MACE: sel |= {'off23', 'osaka', 'omol'}
        if HAVE_FAIR: sel |= {'uma'}
        if HAVE_ANI:  sel |= {'ani'}
        if HAVE_AIMNET: sel |= {'nse', 'rxn'}
        print(f"  [auto] this env provides: "
              f"{'mace ' if HAVE_MACE else ''}{'fairchem ' if HAVE_FAIR else ''}"
              f"{'torchani' if HAVE_ANI else ''}")

    # refuse impossible combinations loudly rather than failing halfway
    if {'omol', 'off23', 'osaka'} & sel and not HAVE_MACE:
        print("  !! mace not importable here -- dropping mace-family models. "
              "Run them from the `mace` env.")
        sel -= {'omol', 'off23', 'osaka'}
    if 'uma' in sel and not HAVE_FAIR:
        print("  !! fairchem not importable here -- dropping uma. "
              "Run it from the `uma` env.")
        sel -= {'uma'}
    if 'ani' in sel and not HAVE_ANI:
        sel -= {'ani'}
    if {'nse', 'rxn'} & sel and not HAVE_AIMNET:
        print("  !! aimnet not importable here -- dropping AIMNet2 models. "
              "Run them from the `aimnet` env (pip install \"aimnet[ase]\").")
        sel -= {'nse', 'rxn'}
    if not sel:
        print("  !! no runnable models in this environment -- nothing to do.")
        return

    registry = [
        ('ani',    "ANI-1xnr",     lambda: get_ani_1xnr(args)),
        ('off23',  "MACE-OFF23",   lambda: get_mace_off23()),
        ('osaka',  "MACE-Osaka24", lambda: get_mace_osaka24(args.osaka_model)),
        ('omol',   "MACE-OMOL",    lambda: get_mace_omol()),
        ('uma',    "UMA-S-1P1",    lambda: get_uma()),
        # AIMNet2-NSE represents spin as SEPARATE alpha/beta charges rather than
        # one global label -- the architectural fix our ablation says OMOL/UMA
        # lack. Run both multiplicities: if NSE works they converge at large r
        # and the asymptote is FLAT where OMOL/UMA rise ~28 kcal/mol.
        ('nse',    "AIMNet2-NSE (S=1)", lambda: get_aimnet2("aimnet2-nse", mult=1)),
        ('nse',    "AIMNet2-NSE (S=3)", lambda: get_aimnet2("aimnet2-nse", mult=3)),
        # aimnet2-rxn: reactive, H/C/N/O only, net-neutral only. Both satisfied by
        # nitromethane. The direct competitor to ANI-1xnr.
        ('rxn',    "AIMNet2-rxn",       lambda: get_aimnet2("aimnet2-rxn")),
    ]

    print(f"  [models] {', '.join(k for k, _, _ in registry if k in sel)}"
          + ("   [RIGID: no optimiser, no constraints]" if args.rigid
             else f"   [relaxed: BFGS fmax={args.fmax} steps={args.steps}]"))

    for key, label, getter in registry:
        if key not in sel:
            continue
        if key == 'osaka' and not args.osaka_model:
            print("MACE-Osaka24 skipped (no --osaka-model path).")
            continue
        try:
            calc = getter()
            E, E_eq = run_scan(calc, label, r_values, args)
            results[label] = {"r": r_values,
                              "E": ((np.array(E.tolist()) - E_eq) * EV_TO_KCAL).tolist(),
                              "E_eq": E_eq,
                              "rigid": bool(args.rigid)}
        except Exception:
            print(f"{label} FAILED:")
            traceback.print_exc()

    # ---- MERGE, do not overwrite (added 2026-08-04) -------------------------
    # MACE and fairchem cannot live in one conda env, so `omol` and `uma` must be
    # run from DIFFERENT environments. Each run therefore has to fold its models
    # into any existing file rather than replacing it, or the second env wipes
    # the first env's results.
    merged = {}
    if os.path.exists(args.json) and not getattr(args, "fresh", False):
        try:
            with open(args.json) as f:
                merged = json.load(f)
            if merged:
                print(f"  [merge] existing models in {args.json}: "
                      f"{', '.join(sorted(merged))}")
        except Exception:
            print(f"  [merge] could not read {args.json}; starting fresh")
            merged = {}
    merged.update(results)
    results.clear(); results.update(merged)      # so plot_graph sees everything

    with open(args.json, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nResults saved: {args.json}   ({', '.join(sorted(merged))})")


# This function graphs the dissociation curves and then saves them to the specified file
def plot_graph(args):

    # Define the file in which to save
    save_file = args.graph

    # Setting title, axis labels, grid lines, and formatting
    font = {'size':10}
    plt.title(NAME + " " + BOND_TO_SCAN + " Dissociation: MLIP Benchmark", fontdict = font)
    plt.xlabel(f"r({BOND_TO_SCAN}) / Å", fontdict = font)
    plt.ylabel("ΔE / kcal $\mathregular{mol^{-1}}$", fontdict = font)
    plt.grid(color = 'lightgrey', linestyle = '-', linewidth = 0.4)
    #plt.ylim(-15, 100)
    
    # >>> REWRITTEN: was five hardcoded plt.plot calls inside ONE try/except.
    # Two failure modes, both silent:
    #   1. a model absent from `results` raised KeyError on the FIRST missing one
    #      and skipped EVERY plot after it -- order-dependent data loss, which
    #      matters now that models are merged in from different conda envs;
    #   2. any model not in that hardcoded list (AIMNet2, anything future) simply
    #      never appeared, with no warning.
    # Now: iterate whatever is actually in `results`, one try per curve, and
    # anything unstyled still plots with a default colour rather than vanishing.
    PLOT_STYLE = {
        "ANI-1xnr":          dict(color='darkcyan',    linestyle='-'),
        "MACE-OFF23":        dict(color='orange',      linestyle='-'),
        "MACE-Osaka24":      dict(color='green',       linestyle='-'),
        "MACE-OMOL":         dict(color='blue',        linestyle='-'),
        "UMA-S-1P1":         dict(color='violet',      linestyle='-'),
        # AIMNet2-NSE: same colour, dashed for the triplet, so the two spin states
        # read as one model in two states rather than two unrelated curves.
        "AIMNet2-NSE (S=1)": dict(color='crimson',     linestyle='-'),
        "AIMNet2-NSE (S=3)": dict(color='crimson',     linestyle='--'),
        "AIMNet2-rxn":       dict(color='saddlebrown', linestyle='-'),
        "GPAW":              dict(color='grey',        linestyle='-'),
    }

    # stable legend order: known models first, then anything new, alphabetically
    order = [k for k in PLOT_STYLE if k in results]
    order += sorted(k for k in results if k not in PLOT_STYLE)

    plotted, skipped = [], []
    for label in order:
        d = results.get(label)
        if not isinstance(d, dict) or not d.get("r") or not d.get("E"):
            skipped.append(label)
            continue
        try:
            plt.plot(d["r"], d["E"], marker='.', label=label,
                     **PLOT_STYLE.get(label, {}))
            plotted.append(label)
        except Exception as e:
            skipped.append(f"{label} ({type(e).__name__})")

    print(f"  [plot] {len(plotted)} curves: {', '.join(plotted)}")
    if skipped:
        print(f"  [plot] SKIPPED (no usable r/E): {', '.join(skipped)}")

    # Plotting the line for expected BDE:
    plt.axline((2, EXP_BDE_KCAL), (4, EXP_BDE_KCAL), color = 'grey', linestyle = '--', linewidth = 1.0, label = 'Expected BDE')

    # Adding the legend for the the graph:
    plt.legend()

    plt.savefig(save_file, dpi=1200, bbox_inches='tight')
    plt.close("all")
    print(f"Successfully saved graph to '{save_file}'.")


if __name__ == '__main__':
    main()
