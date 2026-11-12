#!/usr/bin/env python
"""
orca_molecules.py -- ORCA inputs for every reference species, from the
geometries the pipeline already uses
======================================================================

NO NEW TOOL IS NEEDED FOR GEOMETRY
----------------------------------
ani_bde_replicate.build() already returns a structure for all 38 species (ASE's
G2 collection, plus hand-built CN / C2H / COOH). And the starting geometry
barely matters, because job 1 below OPTIMISES it -- it only has to have the
right connectivity and isomer. Avogadro or `obabel -:"SMILES" --gen3D -O x.xyz`
are fine for one-offs, but generating from build() means the ORCA calculation
and the MLIP evaluation start from literally the same structure.

TWO JOBS PER SPECIES, CHAINED IN ONE FILE
-----------------------------------------
  job 1  wB97X-D3 / def2-TZVP  Opt Freq   -> geometry + ZPE
  job 2  DLPNO-CCSD(T) / aug-cc-pVQZ      -> the energy, on that geometry

This is the standard composite recipe: correlated energies on DFT geometries
with DFT frequencies. ZPE is far less method-sensitive than the electronic
energy, so spending CCSD(T) on frequencies would cost more than everything else
combined and buy nothing.

DLPNO rather than canonical CCSD(T) because it scales near-linearly: it makes
CH3CH2NH2 (10 atoms) as tractable as H2O, so ALL 38 species can share ONE level.
That uniformity is the point -- it is what makes the BDE constraint system
solvable, and it removes the CCCBDB coverage lottery. Validate against CCCBDB on
the seven diatomics; agreement within ~1 kcal/mol certifies the whole set.

FrozenCore FC_NONE matches CCCBDB's "=FULL" (all electrons correlated).

>>> MULTIPLICITY IS ASSIGNED, NOT GUESSED, AND O2 IS THE TRAP <<<
Odd electron count -> doublet. Even -> singlet, EXCEPT O2, whose ground state is
a TRIPLET (3-Sigma-g-minus) despite having an even number of electrons. Running
O2 as a singlet gives a bound-but-wrong answer that will quietly corrupt every
O-O and O-atom quantity downstream. Hydrogen is the other exception: one
electron has no correlation energy, so CCSD(T) is undefined and UHF is exact.

>>> CONFORMERS: THE ONE PLACE THE STARTING GEOMETRY REALLY MATTERS <<<
Several species have distinct minima separated by more than the precision we
are chasing:
    CH3ONO     cis (Z) vs trans (E) methyl nitrite, ~1 kcal/mol apart
    CH3COOH    syn vs anti carboxyl, ~6 kcal/mol -- syn is right
    COOH       cis vs trans radical, ~2 kcal/mol
    CH3CH2NH2  anti vs gauche
The optimiser finds the nearest minimum, not the global one. G2 structures are
the conventional lowest-energy forms, so starting from build() is usually
correct -- but for these four, check the final geometry is the isomer you meant
before trusting the energy.

USAGE
    python orca_molecules.py --max-atoms 4        # the 25-species first tranche
    python orca_molecules.py                      # all 38
    python orca_molecules.py --parse orca_mol/    # -> refs_energies.csv
"""
import argparse, csv, glob, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ani_bde_replicate import REF, FRAGS, ATOMS, build

LEVEL = "DLPNO-CCSD(T)=FULL/aug-cc-pVQZ//wB97X-D3/def2-TZVP"
LEVEL_CANON = "CCSD(T)=FULL/aug-cc-pVQZ//wB97X-D3/def2-TZVP"

# even-electron species whose ground state is NOT a singlet
TRIPLETS = {"O2"}

# >>> FREE-ATOM GROUND STATES ARE NOT GIVEN BY ELECTRON PARITY <<<
# The first version of this file applied the parity rule to atoms too and
# produced C and O as SINGLETS and N as a DOUBLET -- all three wrong. Hund's
# rule maximises spin in a partly filled p shell:
#   H  2S  mult 2     C  3P  mult 3 (p2)
#   N  4S  mult 4     O  3P  mult 3 (p4)
# A wrong atomic state is the most damaging single error available here: the
# atom energy enters EVERY bond that element touches, and the result looks
# entirely plausible.
ATOM_MULT = {"H": 2, "C": 3, "N": 4, "O": 3}
ATOM_STATE = {"H": "2S", "C": "3P", "N": "4S", "O": "3P"}


def multiplicity(atoms, name):
    if name in ATOM_MULT and len(atoms) == 1:
        return ATOM_MULT[name]
    if name in TRIPLETS:
        return 3
    n_el = int(sum(atoms.get_atomic_numbers()))
    return 2 if n_el % 2 else 1


# >>> TWO SEPARATE FILES, NOT A $new_job COMPOUND INPUT (fixed 2026-08-12) <<<
# The first version chained Opt and the single point in one file with
#     * xyzfile 0 2 CH3.xyz
#     *
# and ORCA died with "found a coordinate defintion line (* ctyp charge mult) but
# could not read ctyp, charge and multiplicity". Two things were wrong:
#
#  1. `* xyzfile ...` is SELF-TERMINATING -- it names a file, so there is no
#     coordinate block to close. The trailing `*` is parsed as the start of a
#     new block and the parse fails. Only the inline `* xyz ... *` form takes a
#     closing asterisk.
#  2. Chaining made the handoff filename implicit and gave no opportunity to
#     LOOK at the optimised geometry before spending coupled-cluster time on it
#     -- which is exactly what the conformer cases (CH3COOH, CH3ONO, COOH,
#     CH3CH2NH2) need, since the optimiser finds the nearest minimum, not the
#     global one.
#
# So: {name}_opt.inp writes {name}_opt.xyz, and {name}_sp.inp reads it by name.
OPT_TEMPLATE = """# {name}: {natoms} atoms, charge 0, multiplicity {mult}{extra}
# STEP 1 of 2 -- geometry + ZPE. Writes {name}_opt.xyz for the single point.
! wB97X-D3 def2-TZVP Opt Freq TightSCF
%maxcore {maxcore}
* xyz 0 {mult}
{geom}*
"""

SP_TEMPLATE = """# {name}: STEP 2 of 2 -- energy at
# {level}
# Reads the optimised geometry from {name}_opt.xyz (run {name}_opt.inp first).
# NOTE: `* xyzfile` takes NO closing asterisk.
! {method} TightSCF
%maxcore {maxcore}
%method
  FrozenCore FC_NONE
end
* xyzfile 0 {mult} {name}_opt.xyz
"""

# A single atom has nothing to optimise and no vibrations, so the Opt/Freq job
# is skipped entirely -- ORCA would either error or waste time on it. ZPE = 0.
ATOM_TEMPLATE = """# {name} atom, ground state {state}, multiplicity {mult}
# single point only: nothing to optimise, no vibrations, ZPE = 0
! {method} TightSCF
%maxcore {maxcore}
%method
  FrozenCore FC_NONE
end
%scf
  STABPerform true      # open-shell p atoms can converge to a broken solution
  MaxIter 300
end
* xyz 0 {mult}
{sym}   0.0  0.0  0.0
*
"""


def method_line(name, atoms, pno="TightPNO", canonical=False):
    """
    >>> MEMORY (%maxcore) IS PER CORE, IN MB, AND OPEN-SHELL DLPNO IS HUNGRY <<<
    The first O2 run died with

        not enough memory for Triples evaluation
        MemNeeded 3785.7 MB > MemAvailable 2357.7 MB   [mdci_pno_openshell]

    at %maxcore 3000 -- ORCA keeps some for itself, so 3000 became 2358 usable.
    Three ways out, in order of preference:

      1. RAISE --maxcore (now 8000 by default). Serial runs use one core, so
         maxcore is effectively the total. Keep it below about half the
         machine's RAM.
      2. LOOSEN --pno (TightPNO -> NormalPNO). Tight buys roughly 0.3 vs 1.0
         kcal/mol on reaction energies and costs a lot of memory; Normal is
         adequate for reference BDEs.
      3. --canonical for the small tranche. Below ~5 atoms, canonical CCSD(T)
         is entirely tractable, has no PNO truncation at all, and matches
         CCCBDB's method exactly -- DLPNO exists for the big species, and its
         overhead is wasted on a diatomic.
    """
    if len(atoms) == 1 and atoms.get_chemical_symbols()[0] == "H":
        return "UHF aug-cc-pVQZ"     # one electron: no correlation to recover
    if canonical:
        return "CCSD(T) aug-cc-pVQZ"
    return f"DLPNO-CCSD(T) aug-cc-pVQZ aug-cc-pVQZ/C {pno}"


def write_inputs(outdir, max_atoms=None, only=None, maxcore=8000,
                 pno="TightPNO", canonical=False):
    os.makedirs(outdir, exist_ok=True)
    names = sorted(set(REF) | set(ATOMS) |
                   {f for p in FRAGS.values() for f in p})
    if only:
        names = [n for n in names if n in only]
    made = []
    for nm in names:
        a = build(nm)
        if max_atoms and len(a) > max_atoms:
            continue
        mult = multiplicity(a, nm)
        extra = ""
        if nm in TRIPLETS:
            extra = "   <-- TRIPLET ground state, not a singlet"
        elif nm in ("CH3ONO", "CH3COOH", "COOH", "CH3CH2NH2"):
            extra = "   <-- has conformers; verify the optimised isomer"
        geom = "".join(f"{s:<3}{p[0]:>14.8f}{p[1]:>14.8f}{p[2]:>14.8f}\n"
                       for s, p in zip(a.get_chemical_symbols(),
                                       a.get_positions()))
        if len(a) == 1:
            # atoms: single point only, nothing to optimise, no vibrations
            with open(os.path.join(outdir, f"{nm}_sp.inp"), "w") as fh:
                fh.write(ATOM_TEMPLATE.format(
                    name=nm, sym=a.get_chemical_symbols()[0], mult=mult,
                    state=ATOM_STATE.get(nm, "?"), maxcore=maxcore,
                    method=method_line(nm, a, pno, canonical)))
        else:
            with open(os.path.join(outdir, f"{nm}_opt.inp"), "w") as fh:
                fh.write(OPT_TEMPLATE.format(name=nm, natoms=len(a), mult=mult,
                                             extra=extra, geom=geom,
                                             maxcore=maxcore))
            with open(os.path.join(outdir, f"{nm}_sp.inp"), "w") as fh:
                fh.write(SP_TEMPLATE.format(
                    name=nm, mult=mult, maxcore=maxcore,
                    level=(LEVEL_CANON if canonical else LEVEL),
                    method=method_line(nm, a, pno, canonical)))
        made.append((nm, len(a), mult))

    sh = os.path.join(outdir, "run_orca_mol.sh")
    with open(sh, "w") as fh:
        fh.write(
            '#!/bin/bash\n: "${ORCA:?set ORCA to the orca binary}"\n'
            'cd "$(dirname "$0")"\n'
            'ok () { grep -q "ORCA TERMINATED NORMALLY" "$1" 2>/dev/null; }\n'
            'run () {   # $1 = basename\n'
            '    ok "$1.out" && { echo "skip $1"; return 0; }\n'
            '    echo "=== $1  $(date +%H:%M:%S)"\n'
            '    "$ORCA" "$1.inp" > "$1.out" 2>&1\n'
            '    ok "$1.out" || { echo "    FAILED: see $1.out"; return 1; }\n'
            '}\n\n'
            '# PASS 1: optimisations (smallest first, so failures surface fast)\n'
            'for f in $(ls -S -r *_opt.inp 2>/dev/null); do run "${f%.inp}"; done\n\n'
            '# PASS 2: single points. Each needs its _opt.xyz, so it is checked\n'
            '# for -- a missing geometry means the optimisation failed and the\n'
            '# coupled-cluster job would silently use the wrong structure.\n'
            'for f in $(ls -S -r *_sp.inp 2>/dev/null); do\n'
            '    b=${f%_sp.inp}\n'
            '    if [ -f "${b}_opt.inp" ] && [ ! -s "${b}_opt.xyz" ]; then\n'
            '        echo "SKIP ${b}_sp: no ${b}_opt.xyz (optimisation failed?)"\n'
            '        continue\n'
            '    fi\n'
            '    run "${b}_sp"\n'
            'done\n\n'
            'echo; echo "done. Check for failures:  grep -L \\"TERMINATED NORMALLY\\" *.out"\n')
    os.chmod(sh, 0o755)

    print(f"  {len(made)} inputs in {outdir}/")
    print(f"  {'species':<12}{'atoms':>6}{'mult':>6}")
    for nm, n, m in sorted(made, key=lambda x: (x[1], x[0])):
        flag = "  TRIPLET" if nm in TRIPLETS else ""
        print(f"  {nm:<12}{n:>6}{m:>6}{flag}")
    print(f"\n  run:  ORCA=/path/to/orca bash {sh}")


_PATS = [re.compile(r"E\(CCSD\(T\)\)\s*\.*\s*(-?\d+\.\d+)"),
         re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")]
_ZPE = re.compile(r"Zero point energy\s*\.*\s*(-?\d+\.\d+)\s*Eh")
HARTREE_KCAL = 627.509474


def parse(outdir, energies_csv):
    """Energy from {name}_sp.out, ZPE from {name}_opt.out (atoms: ZPE = 0)."""
    got = {}
    for out in sorted(glob.glob(os.path.join(outdir, "*_sp.out"))):
        nm = os.path.basename(out)[:-7]
        txt = open(out, errors="ignore").read()
        if "ORCA TERMINATED NORMALLY" not in txt:
            print(f"  !! {nm}: single point did not terminate normally"); continue
        e = None
        for p in _PATS:
            m = p.findall(txt)
            if m:
                e = float(m[-1]); break
        if e is None:
            print(f"  !! {nm}: no energy in {os.path.basename(out)}"); continue

        zpe = 0.0 if nm in ATOMS else None
        opt = os.path.join(outdir, f"{nm}_opt.out")
        if os.path.exists(opt):
            otxt = open(opt, errors="ignore").read()
            if "ORCA TERMINATED NORMALLY" not in otxt:
                print(f"  !! {nm}: optimisation did not terminate normally -- "
                      f"the geometry behind this energy is suspect")
            z = _ZPE.findall(otxt)
            if z:
                zpe = float(z[-1]) * HARTREE_KCAL
            if "**   IMAGINARY MODE" in otxt or "***imaginary mode***" in otxt.lower():
                print(f"  !! {nm}: IMAGINARY FREQUENCY -- not a true minimum, "
                      f"likely the wrong conformer or a saddle point")
        # The level string must reflect what was ACTUALLY run: --canonical and
        # DLPNO are different methods, and recording the wrong one would make a
        # mixed set look uniform. Read it back from the input rather than
        # assuming, since the inputs may have been generated in several passes.
        inp = os.path.join(outdir, f"{nm}_sp.inp")
        lvl = LEVEL
        if os.path.exists(inp):
            head = open(inp, errors="ignore").read()
            mline = next((l for l in head.splitlines() if l.startswith("!")), "")
            if "DLPNO" not in mline:
                lvl = LEVEL_CANON
            for tag in ("NormalPNO", "LoosePNO"):
                if tag in mline:
                    lvl = lvl.replace("DLPNO-CCSD(T)", f"DLPNO-CCSD(T)[{tag}]")
        got[nm] = (e, zpe, lvl)
        print(f"  {nm:<12}{e:>16.6f} Ha   ZPE "
              f"{'--' if zpe is None else f'{zpe:7.2f}'} kcal/mol")

    if not got:
        print("  nothing parsed"); return
    fields = ["species", "E_hartree", "ZPE_kcal", "level", "source"]
    rows, keyed = [], {}
    if os.path.exists(energies_csv):
        with open(energies_csv) as fh:
            rd = csv.DictReader(fh)
            fields = rd.fieldnames or fields
            rows = list(rd)
    keyed = {(r["species"], r.get("level", "")): r for r in rows}
    for nm, (e, zpe, lvl) in got.items():
        row = keyed.get((nm, lvl)) or keyed.get((nm, ""))
        if row is None:
            row = {f: "" for f in fields}; row["species"] = nm; rows.append(row)
        row["E_hartree"] = f"{e:.6f}"
        if zpe is not None:
            row["ZPE_kcal"] = f"{zpe:.2f}"
        row["level"] = lvl
        row["source"] = "ORCA (this work)"
    with open(energies_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"\n  merged {len(got)} species into {energies_csv}")
    print("  next:  python make_refs.py --status")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="orca_mol")
    ap.add_argument("--max-atoms", type=int, default=None)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--maxcore", type=int, default=8000,
                    help="MB PER CORE. O2 needed ~3800 for open-shell "
                         "DLPNO triples; keep under half your RAM.")
    ap.add_argument("--pno", default="TightPNO",
                    choices=["TightPNO", "NormalPNO", "LoosePNO"],
                    help="loosen if memory is still short")
    ap.add_argument("--canonical", action="store_true",
                    help="canonical CCSD(T) instead of DLPNO -- best for "
                         "the <=4-atom tranche: no PNO truncation and it "
                         "matches CCCBDB's method exactly")
    ap.add_argument("--parse", metavar="DIR")
    ap.add_argument("--energies", default="refs_energies.csv")
    a = ap.parse_args()
    if a.parse:
        parse(a.parse, a.energies)
    else:
        write_inputs(a.outdir, a.max_atoms, a.only, a.maxcore,
                     a.pno, a.canonical)


if __name__ == "__main__":
    main()
