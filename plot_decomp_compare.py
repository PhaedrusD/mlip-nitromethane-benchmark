#!/usr/bin/env python
"""
plot_decomp_compare.py -- Han Fig. 5-style species populations, side by side
=============================================================================

The 150 ps versions of these plots are dominated by the steady-state plateau,
and the thing they are supposed to show -- how fast nitromethane is consumed --
happens in the first few picoseconds and is invisible. This plots the SAME data
over a short window, one panel per run, with the reactant curve emphasised.

>>> REWRITTEN 2026-09-24 FOR THE CLEAN BOX, AND THE RESULT CHANGED SIGN
    The previous version read two hardcoded trajectories, ani_liq_prod.traj and
    files/dft/omol_run/production_omol.traj, and its docstring carried this
    claim:

        "ANI-1xnr destroys the box before production begins; its production
         file starts 90% decomposed."

    That is an artifact of the STARTING STRUCTURE, not of ANI-1xnr. Those runs
    began from nm_box_1p97.xyz, which fuses two nitromethanes at an O-O contact
    of 1.469 A and has a closest intermolecular approach of 1.290 A. Rerun from
    nm_box_1p97_clean.xyz -- same cell, same 1.970 g/cc, closest approach
    1.759 A -- ANI-1xnr arrives at production 91% INTACT and half-consumes
    0.38 ps into production.

    So the figure no longer says "one model cannot survive its own
    equilibration". It says something narrower and more defensible: from an
    identical clean box, ANI-1xnr consumes nitromethane about ten times faster
    than MACE-OMOL-0, in the direction its C-N error predicts.

    Nothing about the script trusted the box before. Now it checks, refuses to
    draw two panels from different boxes, and prints what it used.

>>> WHY THE REACTANT CURVE IS DRAWN THICK
    Every other curve is a product, and the products all end up in much the same
    place. The only curve that separates the models is CH3NO2 itself. Han's
    Fig. 5 has the same property: the reactant decay IS the result; the product
    curves are context.

>>> TWO SMOOTHING WINDOWS, AND WHY
    Han uses a 2 ps rolling average, which is right for the product plateau and
    WRONG for this reactant. The window averages FORWARD, so on a decay with a
    sub-picosecond half-life the smoothed value at t=0 is already below 0.5.
    Measuring "half of the t=0 value" against that returns t_1/2 = 0.00 ps and
    reports a starting population well under 1.0 when the box verifiably starts
    at exactly 1.000 per molecule.

    So: 2.0 ps for the products, 0.25 ps for the reactant, and t_1/2 is measured
    against 1.0 -- the KNOWN initial condition -- not against a smoothed
    estimate of it.

>>> THE NORMALISER
    species_populations() defaults n_nitromethane to the count of INTACT CH3NO2
    in frame 0. On a run that has already decomposed, that divides every curve
    by a small number and produces impossible populations -- the old figure
    showed N2 at 1.0 per nitromethane, when each molecule carries one nitrogen
    and N2 therefore cannot exceed 0.5. Here the normaliser is the number of
    molecules actually in the box, len(frame0) // 7, read from the trajectory.

>>> TIMING IS A PROPERTY OF THE RUN, NOT OF THE COMMAND LINE
    osaka_run's liquid production used dt = 0.5 fs while every other run used
    0.25 fs. A single --dt-fs applied to both draws one time axis at half its
    true length, and nothing errors. Each run's spacing is read from its own
    log; see plot_reactant_survival.frame_spacing_ps.

USAGE
    python plot_decomp_compare.py --runs ani_run omol_run_cleanbox \\
        --tmax 30 --out MLIP_paper/decomp_0_30ps_cleanbox.png
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "files"))
sys.path.insert(0, _HERE)

# >>> ONE SOURCE OF TRUTH FOR RUN BOOKKEEPING.
# The timing, density and box-identity logic all had to be fixed once already
# (see PROJECT_NOTES SS13, SS14, SS16). Re-implementing any of it here would
# guarantee the two scripts drift apart, and the drift would be silent -- this
# figure and the survival figure would quote different half-lives for the same
# run and both would look fine. So they are imported.
try:
    from plot_reactant_survival import (        # noqa: E402
        ATOMS_PER_NM, REACT_ROLL, classify, density_g_cc, find,
        frame_spacing_ps, half_life, poster_style,
    )
except ImportError as exc:
    # >>> THE CLUSTER COPY IS NOT ALWAYS THE COPY YOU EDITED.
    # This is the third time a stale deployed module has surfaced as an
    # ImportError in the middle of a run (export_finetuned, the ANI-1xnr-Mrr
    # package, now this). A bare traceback does not say which machine is
    # behind, so say it.
    sys.exit(
        f"\n{'!' * 72}\n"
        f"  {exc}\n\n"
        f"  plot_reactant_survival.py next to this script is an OLDER copy than\n"
        f"  the one this figure was written against. Push the current one:\n\n"
        f"      scp plot_reactant_survival.py <cluster>:$MLIP/\n\n"
        f"  Back up the remote copy first if you are unsure which is newer --\n"
        f"  a get/put in the wrong direction has cost this file once already.\n"
        f"{'!' * 72}")

# >>> THESE COME FROM run_manifest DIRECTLY, NOT VIA THE SURVIVAL SCRIPT.
# Routing them through plot_reactant_survival coupled this figure to that
# script's version on whatever machine it runs on, which is exactly how this
# script first failed on the cluster. run_manifest.py is written by the MD
# driver itself, so if a run has a manifest, the module that reads it is there.
try:
    from run_manifest import read_manifest, fingerprint as box_fingerprint
except ImportError:
    def read_manifest(_):
        return None

    def box_fingerprint(atoms):
        """Fragment count and closest intermolecular contact, from geometry.

        Fallback for a checkout without run_manifest.py. Must stay numerically
        identical to run_manifest.fingerprint(), because the two are compared
        against each other across runs: FIND_SCALE 1.15, minimum image, BFS.
        """
        from ase.data import covalent_radii
        d = atoms.get_all_distances(mic=True)
        np.fill_diagonal(d, np.inf)
        rc = np.array([covalent_radii[z] for z in atoms.get_atomic_numbers()])
        A = d < 1.15 * (rc[:, None] + rc[None, :])
        seen, comps = np.zeros(len(atoms), bool), []
        for i in range(len(atoms)):
            if seen[i]:
                continue
            stack, comp = [i], []
            seen[i] = True
            while stack:
                k = stack.pop()
                comp.append(k)
                for j in np.where(A[k] & ~seen)[0]:
                    seen[j] = True
                    stack.append(j)
            comps.append(comp)
        inter = d.copy()
        for c in comps:
            inter[np.ix_(c, c)] = np.inf
        return {"n_fragments": len(comps),
                "min_intermolecular_A": float(inter.min())}
    print("  !! run_manifest.py not found; box identity will fall back to "
          "equilibration frame 0")
from species_analysis import species_populations, SPECIES_ORDER   # noqa: E402

REACTANT = "CH3NO2 (NM+isomers)"
COLORS = {REACTANT: "#1f4fd8", "H2O": "#2e8b30", "CO2": "#c62828",
          "N2": "#8e6bbf", "NH3": "#8a5a3c", "CH3NOOH (+H)": "#e8860c",
          "CH4 (methane)": "#e377c2"}

# >>> THESE ARE QUOTED ON THE FIGURE, SO THEY ARE PRINTED TOO.
# Signed C-N dissociation error against the CCSD(T) reference, rigid scan --
# the same quantity and sign convention as the Figure 2 caption. The paper has
# three different C-N numbers for nitromethane in it (rigid scan, relaxed BDE,
# ensemble-mean relaxed) and they are NOT interchangeable. If the number below
# does not match what Figure 2 prints, fix it here or pass --notes; do not
# assume the panel subtitle is right because it looks plausible.
CN_ERROR = {"ANI-1xnr": -47.4, "ANI-1xnr-Mrr": -0.6, "MACE-OFF23": +10.4,
            "MACE-Osaka24": -10.8, "MACE-OMOL-0": +26.0, "UMA-S-1P1": +25.0}


def load(d, a):
    """Species populations for one run directory, on that run's own clock."""
    from ase.io.trajectory import Trajectory

    prod = find(d, ("production.traj", "production_*.traj", "*prod*.traj"))
    if prod is None:
        return None
    name = os.path.basename(d.rstrip("/"))
    warn = []

    # >>> READ ONLY THE FRAMES WE PLOT, BUT WE CANNOT KNOW HOW MANY UNTIL THE
    # SPACING IS KNOWN, AND THE SPACING CHECK WANTS THE FRAME COUNT. Trajectory
    # supports len() without materialising the frames, so ask it first.
    tr = Trajectory(prod)
    n_total = len(tr)
    dt_prod, note = frame_spacing_ps(d, "production", n_total)
    if dt_prod is None:
        dt_prod = a.dt_fs * a.log_stride / 1000.0
        warn.append(f"production timing ASSUMED from the command line "
                    f"({dt_prod*1000:.0f} fs/frame): {note}")
    elif "WARNING" in note:
        warn.append(note)

    nframe = min(n_total, int(a.tmax / dt_prod) + 1)
    prod_frames = [tr[i] for i in range(nframe)]
    rho = density_g_cc(prod_frames[0])
    finished = os.path.exists(os.path.join(d, "final.xyz"))

    # >>> THE equilibration.traj IN THIS FOLDER IS NOT ALWAYS THIS RUN'S.
    # osaka_run holds a 4-frame equilibration of the 1.97 g/cc box that blew up
    # at 0.05 ps, sitting next to a production trajectory of the 1.137 g/cc box
    # that ran cleanly. Concatenated, the abandoned explosion is drawn as the
    # production run's first picosecond of chemistry. The cell is the exact
    # discriminator: NVT cannot change its own cell.
    eq_frames = []
    eq = find(d, ("equilibration.traj", "*eq*.traj")) if not a.no_eq else None
    if eq:
        cand = list(Trajectory(eq))
        c_eq = np.asarray(cand[0].cell.lengths())
        c_pr = np.asarray(prod_frames[0].cell.lengths())
        dt_e, _ = frame_spacing_ps(d, "equilibration", len(cand))
        if not np.allclose(c_eq, c_pr, atol=1e-3):
            warn.append(f"DROPPED {os.path.basename(eq)}: cell {c_eq[0]:.3f} A "
                        f"({density_g_cc(cand[0]):.3f} g/cc) against production "
                        f"{c_pr[0]:.3f} A ({rho:.3f} g/cc). Different box.")
        elif dt_e is not None and abs(dt_e - dt_prod) > 1e-9:
            warn.append(f"DROPPED {os.path.basename(eq)}: {dt_e*1000:.0f} "
                        f"fs/frame against production's {dt_prod*1000:.0f}. "
                        f"One time axis cannot carry two frame spacings.")
        else:
            eq_frames = cand

    # >>> WHICH BOX, NOT JUST WHICH DENSITY.
    # nm_box_1p97.xyz and nm_box_1p97_clean.xyz are both 11.808 A and both
    # 1.970 g/cc. This whole rewrite exists because of the difference between
    # them, so identifying it is not optional here.
    start_box, man = None, read_manifest(d)
    if man and man.get("fingerprint"):
        f = man["fingerprint"]
        start_box = dict(key=(f["n_fragments"], round(f["min_intermolecular_A"], 3)),
                         source="run_manifest.json", n_frag=f["n_fragments"],
                         min_inter=f["min_intermolecular_A"],
                         name=os.path.basename(man.get("box", {}).get("path") or "?"))
    elif eq_frames:
        f = box_fingerprint(eq_frames[0])
        start_box = dict(key=(f["n_fragments"], round(f["min_intermolecular_A"], 3)),
                         source="equilibration frame 0", n_frag=f["n_fragments"],
                         min_inter=f["min_intermolecular_A"], name="(not recorded)")
    else:
        warn.append("starting box UNKNOWN: no run_manifest.json and no usable "
                    "equilibration trajectory. Production frame 0 has already "
                    "been equilibrated and cannot identify the box.")

    traj = eq_frames + prod_frames
    traj = [at for at in traj if np.isfinite(at.get_positions()).all()]
    n_nm = len(traj[0]) // ATOMS_PER_NM
    kw = dict(dt_fs=dt_prod * 1000.0, log_stride=1, scale=a.scale,
              n_nitromethane=n_nm)

    tp, pops = species_populations(traj, roll_ps=a.roll_ps, **kw)
    tr_, popr = species_populations(traj, roll_ps=REACT_ROLL, **kw)

    # t = 0 is the START OF PRODUCTION; equilibration runs at negative time.
    off = -len(eq_frames) * dt_prod
    tp = np.asarray(tp, float) + off
    tr_ = np.asarray(tr_, float) + off
    y = np.asarray(popr[REACTANT], float)

    return dict(name=name, label=classify(name)[1], t_prod=tp, t_react=tr_,
                pops=pops, y=y, n_mol=n_nm, rho=rho, frame_ps=dt_prod,
                eq_ps=-off, t_end=float(tp[-1]), n_total=n_total,
                finished=finished, warn=warn, start_box=start_box,
                t_half=half_life(tr_, y),
                y_at_prod=float(y[np.argmin(np.abs(tr_))]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run directories, one panel each, in the order given")
    ap.add_argument("--tmax", type=float, default=30.0)
    ap.add_argument("--roll-ps", type=float, default=2.0,
                    help="product smoothing, as in Han et al. The reactant uses "
                         f"{REACT_ROLL} ps regardless; see the docstring.")
    ap.add_argument("--scale", type=float, default=1.2,
                    help="bond threshold as a multiple of summed covalent radii")
    ap.add_argument("--dt-fs", type=float, default=0.25,
                    help="fallback only; timing is read from each run's log")
    ap.add_argument("--log-stride", type=int, default=100, help="fallback only")
    ap.add_argument("--no-eq", action="store_true")
    ap.add_argument("--notes", nargs="+", default=None,
                    help="panel subtitles; default is the C-N error from "
                         "CN_ERROR, which you should check against Figure 2")
    ap.add_argument("--mixed-box", action="store_true",
                    help="allow panels from different starting structures; "
                         "only with a caption that says so")
    ap.add_argument("--out", default="MLIP_paper/decomp_0_30ps_cleanbox.png")
    ap.add_argument("--annotate", action="store_true",
                    help="slide version: suptitle with the half-life ratio")
    a = ap.parse_args()

    runs = []
    for d in a.runs:                      # order given, NOT sorted: it is layout
        if not os.path.isdir(d):
            print(f"  [{d}] not a directory")
            continue
        print(f"  [{os.path.basename(d.rstrip('/')):<22}] ", end="", flush=True)
        r = load(d, a)
        if r is None:
            print("no production trajectory")
            continue
        print(f"{r['n_mol']} mol, {r['rho']:.3f} g/cc, "
              f"{r['frame_ps']*1000:.0f} fs/frame, eq {r['eq_ps']:.2f} ps, "
              f"{r['n_total']} frames total, plotted to {r['t_end']:.1f} ps,"
              f"  t(1/2) = {r['t_half']:+.2f} ps,"
              f"  intact at production start = {r['y_at_prod']:.2f}"
              + ("" if r["finished"] else "   <-- NO final.xyz: STILL RUNNING?"))
        for w in r["warn"]:
            print(f"      !! {w}")
        runs.append(r)
    if not runs:
        sys.exit("  nothing to plot")

    # >>> TWO PANELS FROM TWO DIFFERENT BOXES IS NOT A MODEL COMPARISON.
    # This is the exact mistake the old version of this figure made, and it
    # inverted the ANI-1xnr result. It stops rather than warns, because the
    # resulting figure looks entirely reasonable and there is nothing on it to
    # give the mistake away.
    boxes = {}
    for r in runs:
        if r["start_box"]:
            boxes.setdefault(r["start_box"]["key"], []).append(r)
    if len(boxes) > 1 and not a.mixed_box:
        print("\n" + "!" * 72)
        print("  PANELS STARTED FROM DIFFERENT BOXES:")
        for key, rs in sorted(boxes.items(), key=lambda kv: -kv[0][1]):
            b = rs[0]["start_box"]
            print(f"    {b['n_frag']} fragments, closest intermolecular "
                  f"{b['min_inter']:.3f} A   [{b['source']}, {b['name']}]")
            for r in rs:
                print(f"       {r['name']}")
        print("  This is what made the old version of this figure wrong.")
        print("  Rerun the odd one, or pass --mixed-box with a caption.")
        print("!" * 72)
        sys.exit(1)
    rhos = sorted({round(r["rho"], 3) for r in runs})
    if len(rhos) > 1:
        print(f"\n  !! DIFFERENT DENSITIES: {rhos}. These are different "
              f"experiments, not different models.")

    notes = a.notes if a.notes else [
        (f"{'underbinds' if CN_ERROR[r['label']] < 0 else 'overbinds'} C–N by "
         f"{abs(CN_ERROR[r['label']]):.1f} kcal/mol"
         if r["label"] in CN_ERROR else "") for r in runs]

    # >>> A SINGLE PANEL NEEDS MORE THAN HALF OF THE TWO-PANEL WIDTH.
    # poster_style scales every font by a fixed factor, so at 6.6 in the
    # suptitle runs off both edges. The two-panel case keeps the original
    # 13.2 in, which is the width the type was tuned against.
    n = len(runs)
    width = 9.0 if n == 1 else 6.6 * n
    fig, axes = plt.subplots(1, n, figsize=poster_style.figsize(width, 5.6),
                             sharey=True, squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(top=0.80, bottom=0.135, left=0.085 if n > 1 else 0.13,
                        right=0.985, wspace=0.06)

    for ax, r, note in zip(axes, runs, notes):
        t0 = -r["eq_ps"]
        if r["eq_ps"] > 0:
            ax.axvspan(t0, 0, color="0.85", alpha=.55, zorder=0)
            ax.text(t0 / 2, -0.055, "equil.", fontsize=9, color="0.35",
                    ha="center", va="top", fontweight="bold",
                    transform=ax.get_xaxis_transform())

        for s in SPECIES_ORDER:
            if s not in r["pops"] or s == "other":
                continue
            is_r = s == REACTANT
            y = np.asarray(r["pops"][s], float)
            if not is_r and np.nanmax(y) < 0.02:
                continue
            xx, yy = ((r["t_react"], r["y"]) if is_r else (r["t_prod"], y))
            ax.plot(xx, yy, color=COLORS.get(s, "0.5"),
                    lw=3.4 if is_r else 1.5, zorder=5 if is_r else 3,
                    label=("CH₃NO₂  (reactant)" if is_r else s.split(" (")[0]))

        if np.isfinite(r["t_half"]):
            ax.axvline(r["t_half"], color="#1f4fd8", ls=":", lw=1.8, zorder=2)
            ax.annotate(f"half gone by\n{r['t_half']:+.2f} ps",
                        xy=(r["t_half"], 0.5),
                        xytext=(r["t_half"] + a.tmax * 0.14, 0.68),
                        fontsize=11, color="#1f4fd8", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="#1f4fd8", lw=1.8))
        ax.axhline(0.5, color="0.55", ls="--", lw=1.1, zorder=1)

        # >>> A RUN THAT ENDED INSIDE THE WINDOW MUST LOOK LIKE ONE.
        # MACE-OFF23 stops at 2.4 ps. Drawn on a 30 ps axis with no marker it
        # reads as a plotting failure, or worse, as a flat line at whatever
        # value it happened to stop at.
        if r["t_end"] < a.tmax * 0.98:
            ax.plot([r["t_end"]], [r["y"][-1]],
                    "o" if r["finished"] else ">", ms=10,
                    color=(COLORS[REACTANT] if r["finished"] else "white"),
                    markeredgecolor=COLORS[REACTANT], markeredgewidth=2.0,
                    zorder=7)
            ax.text(r["t_end"], r["y"][-1] + 0.05,
                    f"run ends\n{r['t_end']:.1f} ps", fontsize=9.5,
                    color=COLORS[REACTANT], ha="left", va="bottom",
                    fontweight="bold")

        ax.set_xlim(t0, a.tmax)
        ax.set_xlabel("time from start of production (ps)", fontsize=11)
        ax.set_title(f"{r['label']}\n{note}", fontsize=12, fontweight="bold")
        ax.grid(alpha=.2)
        ax.tick_params(labelsize=10)

    axes[0].set_ylabel("population per nitromethane", fontsize=11)
    axes[0].set_ylim(0, 1.08)
    axes[-1].legend(fontsize=10.5, ncol=2, framealpha=.95, loc="upper right")

    ts = [r["t_half"] for r in runs if np.isfinite(r["t_half"]) and r["t_half"] > 0]
    ratio = (f", {max(ts)/min(ts):.0f}x difference in half-life"
             if len(ts) == len(runs) and len(ts) > 1 else "")
    if a.annotate:
        fig.suptitle(f"How fast is nitromethane consumed? First {a.tmax:.0f} ps"
                     f"{ratio}", fontsize=14.5, fontweight="bold", y=0.975)
    else:
        fig.subplots_adjust(top=0.86)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=300)
    print(f"\n  wrote {a.out}")

    # >>> THE CAPTION IS NOT ON THE FIGURE, SO THE SCRIPT OWES THE NUMBERS.
    print("\n  CAPTION MUST STATE (measured from the runs just plotted):")
    print(f"    {runs[0]['n_mol']} molecules, "
          f"{rhos[0] if len(rhos) == 1 else 'MIXED'} g/cc, 3000 K production")
    if len(boxes) == 1:
        b = next(iter(boxes.values()))[0]["start_box"]
        print(f"    starting box   all panels agree: {b['n_frag']} fragments, "
              f"closest intermolecular {b['min_inter']:.3f} A"
              + (f", {b['name']}" if b["name"] != "(not recorded)" else ""))
    print(f"    connectivity   {a.scale}x the summed covalent radii")
    print(f"    smoothing      products {a.roll_ps:.0f} ps as in Han et al.; "
          f"reactant {REACT_ROLL} ps so a sub-picosecond decay survives")
    print( "    time zero      START OF PRODUCTION; shaded strip is equilibration")
    print( "    dashed line    half the initial nitromethane; t(1/2) is measured")
    print( "                   against 1.0, the known initial condition")
    print("\n  PANEL SUBTITLES QUOTE (check these against the Figure 2 caption):")
    for r, note in zip(runs, notes):
        print(f"    {r['label']:<16} {note}")

    print(f"\n  {'run':<24}{'t(1/2) ps':>11}{'at prod start':>15}"
          f"{'eq ps':>8}{'plotted to':>12}")
    for r in runs:
        print(f"  {r['name']:<24}{r['t_half']:>11.2f}{r['y_at_prod']:>15.2f}"
              f"{r['eq_ps']:>8.2f}{r['t_end']:>12.1f}")

    unfinished = [r["name"] for r in runs if not r["finished"]]
    if unfinished:
        print(f"\n  !! NOT FINISHED (no final.xyz): {', '.join(unfinished)}")
        print("     A partial trajectory is indistinguishable from a stability")
        print("     failure once it is a line on a plot. Do not publish this yet.")


if __name__ == "__main__":
    main()
