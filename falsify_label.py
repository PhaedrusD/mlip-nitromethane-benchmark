#!/usr/bin/env python
"""
falsify_label.py
================

THE EXPERIMENT DESIGNED TO KILL FINDING 20.

Finding 20 claims the spurious dissociation ramp needs TWO ingredients:
the radial envelope closing at r_max = 6 A (the trigger), and a global label
that has become FALSE for the separated fragments (the thing exposed).

Every system we have measured so far breaks a bond homolytically:

    CH3NO2  ->  CH3(doublet) + NO2(doublet),   total label says "singlet"

so the label is false at large separation in every single one. Envelope and
false-label are perfectly confounded across our entire dataset — exactly the
confound we spent two weeks removing between models, reintroduced through the
choice of chemistry.

THE FIX: scan a bond whose fragments are BOTH GENUINELY CLOSED-SHELL SINGLETS.

    C2F4  ->  CF2(singlet) + CF2(singlet),     total label says "singlet"  ✓ TRUE

Same envelope. Same cutoff. Same extrapolation into a region with no training
data. The ONLY thing that changes is that the label stays true all the way out.

    no ramp   -> a false label is NECESSARY. Finding 20 survives its hardest test.
    ramp      -> the label is not the mechanism; it is the cutoff or the edge of
                 the training distribution. Finding 20 is WRONG, and Finding 19's
                 ablation was measuring something else.

⚠️⚠️ THE TRAP IN THIS EXPERIMENT — READ BEFORE INTERPRETING ⚠️⚠️
-----------------------------------------------------------------
The obvious closed-shell test is CH4 + CH4. It is worthless, and here is why.

Two methanes at 4.5 A interact by a few tenths of a kcal/mol. There is nothing
left for the envelope to switch off, so "no ramp" is guaranteed REGARDLESS of
whether the theory is right. A null result from a system with no live
interaction energy proves nothing at all.

The test is only fair if the closed-shell system's BOND IS AS STRONG as the
homolytic control's. Otherwise "no ramp" can always be dismissed as "your test
system was too small to show one".

That rules out the obvious dative adducts. NH3-BH3 is only ~30 kcal/mol against
nitromethane's 59, because dative bonds are intrinsically weak. The way out is a
bond whose fragments are STABILISED SINGLET CARBENES:

    C2F4  -> 2 CF2   ~69 kcal/mol     CF2 singlet-triplet gap ~57 kcal/mol
    C2Cl4 -> 2 CCl2  ~65 kcal/mol

Now the strengths match (69 vs 59) and essentially one variable differs.

So the systems below form a STRENGTH LADDER, and the verdict is judged on total
bond strength -- NOT on the energy still live at 4.5 A. That distinction matters:
for the OMol models, live@4.5 is largely the artifact itself, so using it to
decide whether a test was fair would be circular. Bond strength is a clean
property of the chemistry. live@4.5 is still printed, as context only.

Usage
-----
    conda activate uma  && python falsify_label.py --model uma
    conda activate mace && python falsify_label.py --model omol
    python falsify_label.py --model uma --only C2F4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.optimize import BFGS

sys.path.insert(0, str(Path(__file__).parent))
from ablate_embedding import load_uma, load_omol, step_stats   # noqa: E402

EV_TO_KCAL = 23.0605


def c2f4():
    """Tetrafluoroethylene -> 2 CF2.  ~69 kcal/mol.  THE BEST TEST CASE.

    >>> WHY THIS BEATS NH3-BH3 (2026-08-05)
    A closed-shell dissociation is normally weak, because strong bonds are
    covalent electron-pair bonds and those break homolytically into RADICALS.
    Dative adducts avoid radicals but top out around 20-45 kcal/mol, which is
    weak enough that a null result invites "your test system was too small".

    The exception is a bond whose fragments are STABILISED SINGLET CARBENES.
    CF2 has a singlet ground state with a ~57 kcal/mol singlet-triplet gap --
    it is unambiguously closed-shell, not a radical. So

        C2F4  ->  CF2(singlet) + CF2(singlet)     ~69 kcal/mol

    is comparable in strength to nitromethane C-N (59) and ethane C-C (90),
    while the global "neutral singlet" label stays TRUE for both fragments at
    every separation. That removes the strength objection completely.

    Only C and F, both ubiquitous in OMol25. Six atoms. Planar D2h.
    NOTE: ethylene itself will NOT do -- CH2 has a TRIPLET ground state, so
    H2C=CH2 dissociates to triplets and the label would be false again.
    """
    s, p = ["C", "C"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.31]]
    for z, off in ((-0.738, 0.0), (0.738, 1.31)):
        for x in (+1.094, -1.094):
            s.append("F"); p.append([x, 0.0, off + z])
    return (Atoms(symbols=s, positions=np.array(p)),
            [0, 2, 3], [1, 4, 5], (0, 1))


def c2cl4():
    """Tetrachloroethylene -> 2 CCl2. ~65 kcal/mol. Same idea, replication in a
    different halogen -- CCl2 is also a singlet-ground-state carbene."""
    s, p = ["C", "C"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.34]]
    for z, off in ((-0.962, 0.0), (0.962, 1.34)):
        for x in (+1.426, -1.426):
            s.append("Cl"); p.append([x, 0.0, off + z])
    return (Atoms(symbols=s, positions=np.array(p)),
            [0, 2, 3], [1, 4, 5], (0, 1))


def ethane():
    """C2H6 -> 2 CH3.  ~90 kcal/mol.  THE CONTROL THAT ISOLATES SPIN.

    >>> WHY THIS ROW MATTERS MORE THAN THE CARBENES (2026-08-06)
    C2F4 and C2Cl4 show that a bond with a TRUE label does not ramp. But in those
    systems EVERY global label stays true -- a symmetric homolysis into two
    identical halves preserves elemental composition exactly, so the composition
    channel is as unperturbed as the spin channel. The carbenes therefore prove
    "all labels true -> no ramp". They cannot say WHICH label matters.

    Ethane closes that gap. C2H6 -> 2 CH3 is the SAME structural situation --
    symmetric homolysis, composition preserved exactly (25% C / 75% H either
    way) -- but the fragments are RADICALS, so only the SPIN label goes false.

        C2Cl4  composition preserved, spin TRUE  -> flat
        C2F4   composition preserved, spin TRUE  -> flat
        C2H6   composition preserved, spin FALSE -> ramps hard

    Same symmetry, same preserved composition, opposite outcome. The only thing
    that changed is whether the fragments carry unpaired electrons. That isolates
    SPIN in a way the ablation cannot, because ablation perturbs channels that
    all feed the same downstream sum.
    """
    d, a = 0.767, 1.093          # half C-C, C-H;  staggered D3d
    s, p = ["C", "C"], [[0.0, 0.0, -d], [0.0, 0.0, d]]
    for sign, off in ((-1, 0.0), (+1, 60.0)):
        z = sign * (d + a * np.cos(np.radians(70.5)))
        rad = a * np.sin(np.radians(70.5))
        for k in range(3):
            t = np.radians(120 * k + off)
            s.append("H"); p.append([rad * np.cos(t), rad * np.sin(t), z])
    return (Atoms(symbols=s, positions=np.array(p)),
            [0, 2, 3, 4], [1, 5, 6, 7], (0, 1))


def nh3bh3():
    """Ammonia borane, C3v staggered. B-N dative bond, ~30 kcal/mol.
    BOTH fragments are closed-shell singlets -- this is the whole point."""
    p, s = [], []
    s.append("N"); p.append([0.0, 0.0, 0.0])
    s.append("B"); p.append([0.0, 0.0, 1.66])
    for k in range(3):                       # NH3 hydrogens, pointing away from B
        a = np.radians(120 * k)
        s.append("H"); p.append([0.964 * np.cos(a), 0.964 * np.sin(a), -0.333])
    for k in range(3):                       # BH3 hydrogens, staggered
        a = np.radians(120 * k + 60)
        s.append("H"); p.append([1.169 * np.cos(a), 1.169 * np.sin(a), 1.973])
    return Atoms(symbols=s, positions=np.array(p)), [0, 2, 3, 4], [1, 5, 6, 7], (0, 1)


def ch4_dimer():
    def ch4(o):
        d = 0.629
        v = [[d, d, d], [-d, -d, d], [-d, d, -d], [d, -d, -d]]
        return ["C"] + ["H"] * 4, [o] + [[o[0] + x, o[1] + y, o[2] + z] for x, y, z in v]
    sa, pa = ch4([0, 0, 0]); sb, pb = ch4([0, 0, 3.9])
    return (Atoms(symbols=sa + sb, positions=np.array(pa + pb)),
            list(range(5)), list(range(5, 10)), (0, 5))


def h2o_dimer():
    s = ["O", "H", "H", "O", "H", "H"]
    p = [[0, 0, 0], [0.757, 0, 0.587], [-0.757, 0, 0.587],
         [0, 0, 2.98], [0, 0.757, 3.567], [0, -0.757, 3.567]]
    return Atoms(symbols=s, positions=np.array(p)), [0, 1, 2], [3, 4, 5], (0, 3)


def nitromethane():
    """POSITIVE CONTROL. Homolytic: CH3(doublet) + NO2(doublet). Label FALSE."""
    from bond_pull_scan import build_nm
    a = build_nm()
    return a, [0, 4, 5, 6], [1, 2, 3], (0, 1)


#   name        builder       label true at inf?   dissociates to        strength
SYSTEMS = [
    # --- the real tests: strong bonds, both fragments closed-shell singlets ---
    ("C2F4",    c2f4,         True,  "2 x CF2 singlet carbene   ~69 kcal/mol"),
    ("C2Cl4",   c2cl4,        True,  "2 x CCl2 singlet carbene  ~65 kcal/mol"),
    # --- positive controls: the label goes FALSE ------------------------------
    # ETHANE is the matched partner for the carbenes: same symmetric homolysis,
    # composition preserved identically, but the fragments are radicals. It is
    # the row that isolates SPIN rather than "global labels" collectively.
    ("C2H6",    ethane,       False, "2 x CH3 radical           ~90 kcal/mol"),
    ("CH3NO2",  nitromethane, False, "CH3 + NO2, both doublets  ~59 kcal/mol"),
    # --- secondary: weaker closed-shell, kept for the dose-response ----------
    ("NH3-BH3", nh3bh3,       True,  "NH3 + BH3 dative          ~30 kcal/mol"),
    ("H2O-H2O", h2o_dimer,    True,  "H-bond                    ~5 kcal/mol"),
    ("CH4-CH4", ch4_dimer,    True,  "dispersion only           ~0.5 kcal/mol"),
]

# >>> the systems form a STRENGTH LADDER on purpose. If the ramp tracks bond
# strength rather than label truth, that shows up as C2F4 ramping and CH4-CH4
# not. If it tracks label truth, C2F4 (69 kcal/mol, label true) stays flat while
# CH3NO2 (59 kcal/mol, label false) ramps -- a near-matched pair differing in
# essentially one variable. That pair is the experiment; the rest is context.


def scan(calc, atoms0, B, ia, ib, rs):
    """Rigid: translate fragment B along the A->B axis. No optimisation."""
    u = atoms0.get_positions()[ib] - atoms0.get_positions()[ia]
    r0 = np.linalg.norm(u)
    u = u / r0
    E = np.full(len(rs), np.nan)
    for i, r in enumerate(rs):
        a = atoms0.copy()
        a.positions[B] += (r - r0) * u
        a.info = {"charge": 0, "spin": 1}
        a.calc = calc
        try:
            E[i] = a.get_potential_energy()
        except Exception as exc:
            print(f"    r={r:.2f} failed: {type(exc).__name__}")
    return E


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["uma", "omol"], required=True)
    ap.add_argument("--uma-ckpt", default="uma-s-1p1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", default=None)
    ap.add_argument("--r-max", type=float, default=8.0)
    ap.add_argument("--dr", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"MLIP_paper/falsify_label_{args.model}.csv"

    print(f"loading {args.model} ...")
    calc, _ = (load_uma(args.device, "float32", args.uma_ckpt) if args.model == "uma"
               else load_omol(args.device, "float32"))

    todo = [s for s in SYSTEMS if not args.only or args.only.lower() in s[0].lower()]
    rows, curves, res = [], [], []

    for name, build, label_true, note in todo:
        print(f"\n{'='*82}\n{name}   ->  {note}\n"
              f"label at infinity: {'TRUE  (both fragments really are singlets)' if label_true else 'FALSE (fragments are doublets)'}\n{'='*82}")
        atoms0, A, B, (ia, ib) = build()
        atoms0.info = {"charge": 0, "spin": 1}
        atoms0.calc = calc
        try:
            BFGS(atoms0, logfile=None).run(fmax=0.02, steps=400)
        except Exception as exc:
            print(f"  optimisation FAILED ({type(exc).__name__}) -- skipping")
            continue
        atoms0.calc = None
        req = float(np.linalg.norm(atoms0.get_positions()[ib] - atoms0.get_positions()[ia]))
        rs = np.arange(max(0.8, req - 0.4), args.r_max + args.dr / 2, args.dr)
        print(f"  optimised {name}: axis distance = {req:.3f} A;  scanning "
              f"{rs[0]:.1f} - {rs[-1]:.1f} A ({len(rs)} points)")

        E = scan(calc, atoms0, B, ia, ib, rs)
        rel = (E - np.nanmin(E)) * EV_TO_KCAL
        rise, step_r, jump, _ = step_stats(rs, rel)
        far = rs >= 4.0
        drop = float(np.nanmax(rel[far]) - rel[-1]) if far.any() else np.nan
        live = float(rel[-1] - np.interp(4.5, rs, rel))
        amp = max(abs(rise), abs(drop))

        print(f"\n  binding energy (rigid)      {rel[-1]:8.1f} kcal/mol")
        print(f"  energy still live at 4.5 A  {live:8.1f} kcal/mol   <-- fairness check")
        print(f"  upward rise beyond 4 A      {rise:8.1f}")
        print(f"  downward drop beyond 4 A    {drop:8.1f}")
        print(f"  largest single-step jump    {jump:8.1f} kcal/mol at {step_r:.1f} A")
        rows.append(f"{name},{label_true},{rel[-1]:.2f},{live:.2f},{rise:.2f},"
                    f"{drop:.2f},{jump:.2f},{step_r:.2f}")
        curves += [f"{name},{label_true},{rs[i]:.2f},{E[i]:.6f},{rel[i]:.3f}"
                   for i in range(len(rs))]
        res.append((name, label_true, live, amp, abs(jump), step_r, abs(rel[-1])))

    # >>> FIXED 2026-08-06. This used to OVERWRITE both files on every run, so
    # `--only C2H6` destroyed the carbene and nitromethane rows that had taken
    # the previous run to produce. Same class of bug as bde_scan's --out
    # collision. Now it MERGES: rows for systems in THIS run replace their old
    # versions, everything else is preserved.
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ran = {r.split(",")[0] for r in rows}

    def merge(path, header, new):
        keep = []
        if Path(path).exists():
            old = Path(path).read_text().splitlines()
            if old and old[0] == header:
                keep = [ln for ln in old[1:]
                        if ln.strip() and ln.split(",")[0] not in ran]
        if keep:
            print(f"  [merge] preserving {len({k.split(',')[0] for k in keep})} "
                  f"system(s) already in {Path(path).name}")
        Path(path).write_text(header + "\n" + "\n".join(keep + new) + "\n")

    merge(out, "system,label_true,binding_kcal,live_at_4.5,rise_kcal,"
               "drop_kcal,max_jump_kcal,step_r_A", rows)
    merge(out.replace(".csv", "_curves.csv"),
          "system,label_true,r_A,E_eV,E_rel_kcal", curves)

    # ---------------- the verdict ------------------------------------------
    print(f"\n{'='*82}\nVERDICT\n{'='*82}")
    print(f"{'system':<12}{'label':<8}{'bond':>8}{'live@4.5':>10}{'|ramp|':>9}"
          f"{'|jump|':>9}{'at':>7}   fair test?")
    print("-" * 90)
    LIVE_MIN, RAMP_MIN = 15.0, 8.0
    for name, lt, live, amp, jmp, sr, bond in res:
        inf = ("positive control" if not lt else
               "YES -- strength matched" if bond >= 40 else
               "weak -- a null here proves little")
        print(f"{name:<12}{'TRUE' if lt else 'FALSE':<8}{bond:>8.1f}{live:>10.1f}"
              f"{amp:>9.1f}{jmp:>9.1f}{sr:>7.1f}   {inf}")

    # >>> fairness is judged on BOND STRENGTH, not on live@4.5.
    # live@4.5 is contaminated: for the OMol models it is largely the artifact
    # itself, so using it to decide whether a test was fair is circular. Total
    # binding energy is a clean property of the chemistry.
    fair = [x for x in res if x[1] and x[6] >= 40.0]   # true label, strong bond
    ctrl = [x for x in res if not x[1]]
    print()
    if not fair:
        print("  ⚠️ INCONCLUSIVE. No closed-shell system had a bond strong enough")
        print("     (>= 40 kcal/mol) to match the homolytic control. A null result")
        print("     here proves little -- it could just mean the system was too weak.")
    elif all(x[3] < RAMP_MIN for x in fair):
        print("  ✅ FINDING 20 SURVIVES. Closed-shell systems with comparable live")
        print("     interaction energy show NO ramp, while the homolytic control does.")
        print("     A false label is NECESSARY -- the envelope alone is not sufficient,")
        print("     and neither is being outside the training distribution.")
    else:
        print("  ❌ FINDING 20 IS FALSIFIED. A system whose label stays TRUE still")
        print("     ramps. The mechanism is not the label going stale -- it is the")
        print("     cutoff, or the edge of the training data. Finding 19's ablation")
        print("     removed the ramp by some other route and must be reinterpreted.")
        print("     Report this immediately; do not present Finding 20 as it stands.")
    if ctrl:
        print(f"\n  positive control: {ctrl[0][0]} bond {ctrl[0][6]:.0f} kcal/mol, "
              f"|ramp| = {ctrl[0][3]:.1f} at {ctrl[0][5]:.1f} A")
        if ctrl[0][3] < RAMP_MIN:
            print("  ⚠️ THE POSITIVE CONTROL DID NOT RAMP. Something is wrong with the")
            print("     setup itself -- fix that before reading anything above.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
