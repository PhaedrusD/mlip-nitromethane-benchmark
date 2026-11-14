#!/usr/bin/env python
"""
make_refs.py -- build refs.csv from total energies, not from hand-typed BDEs
============================================================================

WHY THIS EXISTS
---------------
Entering a De per bond types some species' energies TWICE. NO2 is a parent
(NO2 -> NO + O) and a fragment (CH3NO2 -> CH3 + NO2); OH appears in four bonds.
Two copies of a number drift apart, and the fine-tune then receives
contradictory statements about what a species is worth -- the most likely cause
of NO2 swinging -42 kcal/mol in the first successful run.

Store ONE total energy per species and derive every BDE from it, and the
inconsistency becomes structurally impossible: NO2 has one energy, so its use
as a parent and as a fragment cannot disagree.

    De  = E(A) + E(B) - E(AB)                      [Hartree -> kcal/mol]
    D0  = De - [ZPE(AB) - ZPE(A) - ZPE(B)]         [ZPE in kcal/mol, atoms 0]

INPUT: refs_energies.csv
    species,E_hartree,ZPE_kcal,level,source
Atoms (H, C, N, O) are species too and are REQUIRED -- every diatomic BDE needs
them. ZPE may be left blank; D0 is then omitted for bonds touching that species.

>>> ONE LEVEL WITHIN EACH BDE -- NOT ONE LEVEL EVERYWHERE <<<
De = E(A) + E(B) - E(AB) is only meaningful if all three energies come from the
same method and basis. It does NOT require every bond to share a level: dos
Santos used three (35 bonds at CCSD(T)=FULL/aug-cc-pVQZ, one at aug-cc-pVTZ,
seven at cc-pVTZ) because CCCBDB coverage is not uniform, and each of their BDEs
is still internally consistent.

So refs_energies.csv is keyed on (species, level) and may hold a species several
times. For each bond the script picks the highest-ranked level at which ALL
THREE of its species exist; a bond with no common level stays provisional rather
than being computed from a mixture. It then reports which level every bond used,
so the manuscript can state them.

ZPE also matters more than it looks: dos Santos compare electronic energies on
both sides (De vs De), while the ANI-1xnr paper compares its electronic 116.6
to an experimental D0 of 103.3 -- about 9 kcal/mol of ZPE. Keeping both columns
means we never have to guess which convention a downstream number used.

USAGE
    python make_refs.py --status          # what is still missing
    python make_refs.py                   # write refs.csv
"""
import argparse, csv, os, sys

HARTREE_KCAL = 627.509474

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ani_bde_replicate import REF, FRAGS, ATOMS


def required_species():
    """Every species whose energy is needed: parents, fragments, atoms."""
    need = set(REF) | set(ATOMS)
    for fa, fb in FRAGS.values():
        need |= {fa, fb}
    return sorted(need)


def load_energies(path):
    """
    Returns {(species, level): {...}}.

    >>> KEYED ON (SPECIES, LEVEL), NOT SPECIES (fixed 2026-08-17) <<<
    The first version keyed on species alone, so a file holding the same species
    at two levels -- canonical CCSD(T) from the <=4-atom tranche and
    DLPNO-CCSD(T) for the larger ones -- would silently keep whichever row came
    last. That is the mixed-set-masquerading-as-uniform failure this whole file
    exists to prevent, hidden one layer deeper.

    CCCBDB has the same non-uniform coverage, which is why dos Santos used three
    different levels across their 43 bonds. The rule is not "one level
    everywhere" but "one level WITHIN each BDE": De = E(A) + E(B) - E(AB) is only
    meaningful if all three come from the same method and basis.
    """
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path) as fh:
        for r in csv.DictReader(fh):
            sp = (r.get("species") or "").strip()
            if not sp or sp.startswith("#"):
                continue
            try:
                e = float((r.get("E_hartree") or "").strip())
            except ValueError:
                continue                       # blank / placeholder -> not yet
            try:
                z = float((r.get("ZPE_kcal") or "").strip())
            except ValueError:
                z = None
            lvl = (r.get("level") or "").strip()
            rows[(sp, lvl)] = dict(E=e, ZPE=z, level=lvl,
                                   source=(r.get("source") or "").strip())
    return rows


# >>> ONE-ELECTRON SPECIES ARE LEVEL-INDEPENDENT (fixed 2026-08-18) <<<
# Hydrogen has a single electron, so there is no correlation energy to recover:
# UHF is EXACT, and CCSD(T) and DLPNO-CCSD(T) return the identical number at the
# same basis. orca_molecules therefore emits a plain `! UHF aug-cc-pVQZ` job for
# H in every pass, and the parser -- which infers the level from the "!" line --
# labelled it canonical in both. H ended up with no DLPNO row, so CH4 -> CH3 + H
# and C2H4 -> C2H3 + H had "no COMMON level" and stayed provisional, including
# the CH4 ANCHOR bond.
#
# The fix is not to relabel H arbitrarily but to state the physics: its energy
# is valid at every level sharing that basis. Empirically confirmed here --
# H came out -0.499948 Ha in both the canonical and DLPNO passes, identical to
# six decimals.
EXACT_SPECIES = {"H"}
_BASES = ("aug-cc-pV5Z", "aug-cc-pVQZ", "aug-cc-pVTZ", "aug-cc-pVDZ",
          "cc-pV5Z", "cc-pVQZ", "cc-pVTZ", "cc-pVDZ")


def basis_of(lvl):
    """Basis set named in a level string, or None."""
    up = lvl.upper()
    for b in _BASES:
        if b.upper() in up:
            return b
    return None


def broadcast_exact(E):
    """
    Replicate one-electron species across every level that shares their basis.

    Returns the number of rows added. Only touches EXACT_SPECIES, so it cannot
    silently paper over a genuine level mismatch for anything correlated.
    """
    levels = {lvl for (_, lvl) in E}
    added = []
    for (sp, lvl) in list(E):
        if sp not in EXACT_SPECIES:
            continue
        b = basis_of(lvl)
        if b is None:
            continue
        for other in levels:
            if other != lvl and basis_of(other) == b and (sp, other) not in E:
                E[(sp, other)] = dict(E[(sp, lvl)], level=other)
                added.append((sp, other))
    return added


def level_rank(lvl):
    """
    Preference score for a level of theory. Higher is better.

    Canonical CCSD(T) beats DLPNO (no PNO truncation, and it is what CCCBDB
    reports); a quadruple-zeta basis beats triple; TightPNO beats Normal beats
    Loose. Unknown strings rank lowest so they are never silently preferred.
    """
    s = 0
    if "DLPNO" not in lvl.upper():
        s += 100
    elif "TIGHTPNO" in lvl.upper():
        s += 10
    elif "NORMALPNO" in lvl.upper():
        s += 5
    if "PVQZ" in lvl.upper():
        s += 50
    elif "PVTZ" in lvl.upper():
        s += 25
    return s


def best_common_level(trio, E):
    """
    The highest-ranked level at which ALL THREE species of a bond are available.
    None if there is no such level -- in which case the bond stays provisional
    rather than being computed from a mixture.
    """
    sets = []
    for sp in trio:
        sets.append({lvl for (s, lvl) in E if s == sp})
    common = set.intersection(*sets) if sets else set()
    if not common:
        return None
    # >>> DETERMINISTIC TIE-BREAK (fixed 2026-08-17) <<<
    # "CCSD(T)=FULL/aug-cc-pVQZ" and
    # "CCSD(T)=FULL/aug-cc-pVQZ//wB97X-D3/def2-TZVP" both score 150, so a plain
    # max() picked between them by set-iteration order -- which made CO's De move
    # 257.8 -> 258.51 between otherwise identical runs. Break ties on string
    # length, which prefers the MORE FULLY SPECIFIED level (the one that records
    # the geometry it was evaluated on), then alphabetically so the result is
    # reproducible.
    return max(common, key=lambda l: (level_rank(l), len(l), l))


def any_zpe(sp, E):
    """
    ZPE for a species from whichever level has it.

    Legitimate here because our ZPE always comes from the same wB97X-D3/def2-TZVP
    frequency job regardless of which single-point level was run on top -- that
    is the point of the composite recipe. Atoms have ZPE = 0 by definition.
    """
    if sp in ATOMS:
        return 0.0
    for (s, lvl), v in E.items():
        if s == sp and v["ZPE"] is not None:
            return v["ZPE"]
    return None


def make_template(path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["species", "E_hartree", "ZPE_kcal", "level", "source"])
        for sp in required_species():
            kind = ("atom" if sp in ATOMS else
                    "parent" if sp in REF else "fragment")
            w.writerow([sp, "", "", "", f"TODO ({kind})"])
    print(f"wrote template {path} with {len(required_species())} species")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--energies", default="refs_energies.csv")
    ap.add_argument("--out", default="refs.csv")
    ap.add_argument("--status", action="store_true",
                    help="report progress and exit")
    ap.add_argument("--template", action="store_true",
                    help="write a blank refs_energies.csv and exit")
    ap.add_argument("--allow-mixed-levels", action="store_true")
    args = ap.parse_args()

    if args.template:
        make_template(args.energies); return

    E = load_energies(args.energies)
    _bcast = broadcast_exact(E)
    if _bcast:
        print(f"  one-electron species broadcast across matching bases "
              f"(UHF = CCSD(T) exactly for H):")
        for sp, lvl in _bcast:
            print(f"    {sp} -> {lvl}")
    need = required_species()
    have = sorted({sp for (sp, _) in E})
    missing = [s for s in need if s not in have]
    levels = sorted({lvl for (_, lvl) in E if lvl})
    if len(levels) > 1:
        print(f"  {len(levels)} levels of theory present:")
        for lvl in sorted(levels, key=lambda x: -level_rank(x)):
            n = sum(1 for (_, l) in E if l == lvl)
            print(f"    rank {level_rank(lvl):>3}  {n:>2} species  {lvl}")
        print("  -> each BDE will use the highest-ranked level where ALL THREE")
        print("     of its species exist. Bonds with no common level stay")
        print("     provisional rather than being computed from a mixture.")

    print(f"  energies present: {len(have)}/{len(need)}")
    if missing:
        atoms_m = [s for s in missing if s in ATOMS]
        if atoms_m:
            print(f"\n  !! MISSING ATOMS: {atoms_m}")
            print("     Every diatomic BDE needs these. De(N2) = 2*E(N) - E(N2);")
            print("     without E(N) the molecular energy cannot be converted.")
        frag_m = [s for s in missing if s not in ATOMS and s not in REF]
        par_m = [s for s in missing if s in REF]
        if frag_m:
            print(f"\n  missing fragments ({len(frag_m)}): {frag_m}")
        if par_m:
            print(f"  missing parents   ({len(par_m)}): {par_m}")

    # ---- derive every BDE we can ------------------------------------------
    out, done, blocked, used_levels = [], 0, [], {}
    print(f"\n  {'species':<11}{'bond':>5}{'De':>9}{'was':>8}{'shift':>8}"
          f"{'D0':>8}   level")
    for sp in sorted(REF):
        bond, de_old, d0_old, note = REF[sp]
        fa, fb = FRAGS[sp]
        trio = (sp, fa, fb)
        lvl = best_common_level(trio, E)
        if lvl is None:
            have_any = {x for x in trio if any(s2 == x for (s2, _) in E)}
            lack = [x for x in trio if x not in have_any]
            blocked.append((sp, lack or ["no COMMON level for " + "/".join(trio)]))
            out.append([sp, bond, de_old, d0_old, "PROVISIONAL-hand-entered",
                        "", note])
            continue
        de = (E[(fa, lvl)]["E"] + E[(fb, lvl)]["E"]
              - E[(sp, lvl)]["E"]) * HARTREE_KCAL
        zs = [any_zpe(x, E) for x in trio]
        d0 = de - (zs[0] - zs[1] - zs[2]) if all(z is not None for z in zs) else ""
        out.append([sp, bond, round(de, 2),
                    round(d0, 2) if d0 != "" else "",
                    E[(sp, lvl)]["source"] or "CCCBDB", lvl, note])
        done += 1
        used_levels[lvl] = used_levels.get(lvl, 0) + 1
        d_old = de - de_old
        flag = "  <-- shifts >5" if abs(d_old) > 5 else ""
        short = ("canon" if "DLPNO" not in lvl.upper() else "DLPNO") + \
                ("/QZ" if "PVQZ" in lvl.upper() else "/TZ" if "PVTZ" in lvl.upper() else "")
        print(f"    {sp:<11}{bond:>5}{de:>9.2f}{de_old:>8.1f}{d_old:>+8.1f}"
              f"{(f'{d0:.2f}' if d0 != '' else '--'):>8}   {short}{flag}")

    if len(used_levels) > 1:
        print(f"\n  !! {len(used_levels)} DIFFERENT LEVELS across bonds:")
        for lvl, n in sorted(used_levels.items(), key=lambda kv: -kv[1]):
            print(f"       {n:>2} bonds  {lvl}")
        print("     Each BDE is internally consistent, which is what matters --")
        print("     but the manuscript must state the per-bond level, as dos")
        print("     Santos do (35 bonds at QZ, 1 at aug-TZ, 7 at cc-pVTZ).")

    if blocked:
        print(f"\n  {len(blocked)} bonds still blocked, keeping provisional values:")
        for sp, lack in blocked[:12]:
            print(f"    {sp:<11} needs {lack}")

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["species", "bond", "De_kcal", "D0_kcal", "source",
                    "level", "note"])
        w.writerows(out)
    print(f"\n  wrote {args.out}: {done} derived, {len(blocked)} provisional")
    print("  now run:  python ani_bde_replicate.py --check-refs")


if __name__ == "__main__":
    main()
