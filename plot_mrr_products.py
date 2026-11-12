#!/usr/bin/env python
"""
plot_mrr_products.py -- what the Morse remediation actually changed
====================================================================

The ANI-1xnr-Mrr section needs a figure that shows a DIFFERENCE. The reactant
survival curves do not: 0.25 ps against 0.28 ps, one 25 fs frame apart, two
lines on top of each other. That is a real result but it is a null one, and a
null result cannot be the only panel in a section about a remediation.

The difference is in the PRODUCTS, and it is large.

>>> WHY AN ATOM BUDGET AND NOT A BAR CHART OF FRAGMENT COUNTS
    Counting fragments makes the reader do conservation arithmetic in their
    head: N2 carries two nitrogens, NH3 carries one, so "N2 went from 5.0 to
    9.6 fragments" and "NH3 went from 3.4 to 1.6" are not on the same scale and
    cannot be added up. Worse, nothing tells the reader whether the columns are
    complete or whether some large sink has been left off.

    Counting NITROGEN ATOMS fixes both. Every bar sums to 32, which is the
    number of nitrogens in the box and a number the reader can verify at a
    glance. Rerouting then looks like exactly what it is: the same material
    moved from one block to another. A sink that has been omitted would make
    the bar short, so the figure cannot quietly hide one.

>>> THE SECOND PANEL IS THE NULL, AND IT BELONGS HERE
    Shown alone, panel (a) invites "so the remediation worked". It did not, in
    the sense the section cares about: the rate is untouched. Putting the
    survival curves next to the budget makes the actual claim visible in one
    view -- the correction reroutes the chemistry without changing the kinetics.
    A null result is worth a panel when it is half of the argument.

>>> GROUPING, AND WHY THE TAIL IS SHOWN RATHER THAN DROPPED
    About nine of the 32 nitrogens sit in a long tail of minor fragments. Those
    are collected into "other", which is plotted, not discarded. Dropping them
    would break the conservation the figure relies on for its credibility.

>>> ONE SEED
    One trajectory per model. The error bars are the fluctuation of each sink
    across the averaging window, which measures how steady the steady state is,
    NOT run-to-run reproducibility. Say so in the caption; do not let a reader
    take them for confidence intervals on the difference.

USAGE
    python plot_mrr_products.py --runs ani_run ani_mrr_run \\
        --out MLIP_paper/fig_mrr_products.png
"""

import argparse
import collections
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "files"))
sys.path.insert(0, _HERE)

try:
    import poster_style
    poster_style.apply()
except ModuleNotFoundError:
    class _NoStyle:
        @staticmethod
        def figsize(w, h):
            return (w, h)
    poster_style = _NoStyle()
    print("\n" + "!" * 72)
    print("  poster_style.py NOT FOUND -- type will not match the other")
    print("  paper figures. Copy it next to this script and rerun.")
    print("!" * 72 + "\n")

from species_analysis import (adjacency, connected_components,   # noqa: E402
                              hill_formula, species_populations)

REACTANT = "CH3NO2 (NM+isomers)"
ELEMENT = "N"
ATOMS_PER_NM = 7

# >>> GROUPS ARE BY CHEMISTRY, NOT BY ABUNDANCE.
# Ordered bottom to top. The point of the figure is that nitrogen leaves the
# reduced and the C-bonded sinks and arrives in N2, so those are the blocks.
GROUPS = [
    ("N₂", ["N2"], "#1f4fd8"),
    ("NHₓ  (NH₃, NH₂, NH₄⁺)", ["H3N", "H2N", "H4N", "HN"], "#8a5a3c"),
    ("C–N–O fragments  (HNCO, CNO, HCN)",
     ["CHNO", "CNO", "CHN", "CN", "CNO2", "CHNO2"], "#e8860c"),
    ("CHₓNOₓ remnants", ["CH2NO2", "CH3NO2", "CH3NO", "CH2NO", "CH4NO2",
                                  "CH3NO3", "CH2N"], "#2e8b30"),
    ("NO, NO₂, N₂O", ["NO", "NO2", "N2O", "HNO", "HNO2", "HNO3"], "#8e6bbf"),
    ("larger clusters + minor species", None, "0.62"),
]


def element_budget(run, frames, element=ELEMENT, scale=1.2):
    """Mean and s.d. of <element> atoms held by each fragment formula.

    Counts ATOMS, not fragments, so the total is conserved and equal to the
    number of that element in the box.
    """
    from ase.io.trajectory import Trajectory
    tr = Trajectory(os.path.join(run, "production.traj"))
    per = collections.defaultdict(list)
    seen = []
    for i in frames:
        a = tr[i]
        sym = np.array(a.get_chemical_symbols())
        A = adjacency(a, scale)
        f = collections.Counter()
        for comp in connected_components(A):
            s = sym[list(comp)]
            n = int((s == element).sum())
            if n:
                f[hill_formula(s)] += n
        seen.append(sum(f.values()))
        for k, v in f.items():
            per[k].append(v)
    n = len(frames)
    # a formula absent from a frame contributes zero to that frame, so pad
    out = {k: (float(np.mean(v + [0] * (n - len(v)))),
               float(np.std(v + [0] * (n - len(v))))) for k, v in per.items()}
    return out, float(np.mean(seen))


def grouped(budget):
    named = {lab: 0.0 for lab, _, _ in GROUPS}
    sd = {lab: 0.0 for lab, _, _ in GROUPS}
    claimed = set()
    for lab, keys, _ in GROUPS:
        if keys is None:
            continue
        for k in keys:
            if k in budget:
                named[lab] += budget[k][0]
                sd[lab] += budget[k][1] ** 2
                claimed.add(k)
    other = [k for k in budget if k not in claimed]
    named[GROUPS[-1][0]] = sum(budget[k][0] for k in other)
    sd[GROUPS[-1][0]] = sum(budget[k][1] ** 2 for k in other)
    return named, {k: v ** 0.5 for k, v in sd.items()}, other


def survival(run, tmax, scale=1.2, roll=0.25):
    from ase.io.trajectory import Trajectory
    tr = Trajectory(os.path.join(run, "production.traj"))
    # frame spacing from the run's own log; never assumed (PROJECT_NOTES §13)
    log = os.path.join(run, "production.log")
    ts = []
    for line in open(log):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        if len(p) >= 2:
            ts.append(float(p[1]))
        if len(ts) >= 2:
            break
    fps = ts[1] - ts[0]
    n = min(len(tr), int(tmax / fps) + 1)
    t, pops = species_populations([tr[i] for i in range(n)],
                                  dt_fs=fps * 1000.0, log_stride=1,
                                  roll_ps=roll, scale=scale,
                                  n_nitromethane=len(tr[0]) // ATOMS_PER_NM)
    t = np.asarray(t, float)
    y = np.asarray(pops[REACTANT], float)
    w = np.where(y <= 0.5)[0]
    return t, y, (float(t[w[0]]) if len(w) else np.nan), fps



def _panel_letter(ax, letter):
    """Bold panel letter just outside the top-left corner, journal style.
    Outside the axes so it can never land on data, a legend or a colourbar."""
    ax.text(0.0, 1.015, f"({letter})", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=13, fontweight="bold")

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs=2, default=["ani_run", "ani_mrr_run"],
                    help="baseline first, remediated second")
    ap.add_argument("--labels", nargs=2, default=["ANI-1xnr", "ANI-1xnr-Mrr"])
    ap.add_argument("--from-ps", type=float, default=120.0,
                    help="start of the averaging window for the budget")
    ap.add_argument("--to-ps", type=float, default=150.0)
    ap.add_argument("--every", type=int, default=20, help="frames between samples")
    ap.add_argument("--frame-ps", type=float, default=0.025)
    ap.add_argument("--tmax", type=float, default=6.0, help="panel (b) window")
    ap.add_argument("--scale", type=float, default=1.2)
    ap.add_argument("--annotate", action="store_true",
                    help="slide version: suptitle, panel titles, on-axes notes")
    ap.add_argument("--out", default="MLIP_paper/fig_mrr_products.png")
    ap.add_argument("--csv", default="MLIP_paper/mrr_nitrogen_budget.csv")
    a = ap.parse_args()

    frames = list(range(int(a.from_ps / a.frame_ps),
                        int(a.to_ps / a.frame_ps) + 1, a.every))
    print(f"  budget window {a.from_ps:.0f}-{a.to_ps:.0f} ps, "
          f"{len(frames)} samples per run")

    budgets, totals, tails = {}, {}, {}
    for r in a.runs:
        b, tot = element_budget(r, frames, scale=a.scale)
        budgets[r], totals[r] = b, tot
        print(f"  [{r:<14}] {ELEMENT} atoms accounted = {tot:.2f}")

    # >>> CONSERVATION IS THE FIGURE'S OWN CHECK. FAIL LOUDLY.
    exp = round(max(totals.values()))
    for r, tot in totals.items():
        if abs(tot - exp) > 0.02:
            sys.exit(f"  {r} accounts for {tot:.2f} {ELEMENT} atoms, not {exp}. "
                     f"The budget is not conserved and the figure would be wrong.")

    gs, sds = {}, {}
    for r in a.runs:
        gs[r], sds[r], tails[r] = grouped(budgets[r])

    print(f"\n  {ELEMENT}-atom budget (of {exp})")
    print(f"  {'sink':<40}{a.labels[0]:>14}{a.labels[1]:>16}{'change':>10}")
    for lab, _, _ in GROUPS:
        x, y = gs[a.runs[0]][lab], gs[a.runs[1]][lab]
        ch = f"{y/x:.2f}x" if x > 0.05 else "--"
        print(f"  {lab:<40}{x:>14.2f}{y:>16.2f}{ch:>10}")

    surv = {r: survival(r, a.tmax, scale=a.scale) for r in a.runs}
    for r in a.runs:
        print(f"  [{r:<14}] t(1/2) = {surv[r][2]:.3f} ps "
              f"at {surv[r][3]*1000:.0f} fs resolution")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=poster_style.figsize(13.6, 6.4),
                                   gridspec_kw=dict(width_ratios=[1.3, 1.0]))
    # bottom holds the two-column legend under panel (a); top holds a suptitle
    # that poster_style renders about 40%% wider than default matplotlib
    fig.subplots_adjust(top=0.82 if a.annotate else 0.92, bottom=0.30,
                        left=0.075, right=0.985, wspace=0.26)

    # ---- (a) the atom budget ---------------------------------------------
    xs = np.arange(2)
    bottom = np.zeros(2)
    for lab, _, col in GROUPS:
        vals = np.array([gs[r][lab] for r in a.runs])
        ax1.bar(xs, vals, 0.58, bottom=bottom, color=col, edgecolor="white",
                linewidth=1.4, label=lab, zorder=3)
        for k, v in enumerate(vals):
            if v >= 1.4:
                ax1.text(xs[k], bottom[k] + v / 2, f"{v:.1f}", ha="center",
                         va="center", fontsize=10, fontweight="bold",
                         color="white" if col != "0.62" else "0.2", zorder=5)
        bottom += vals
    ax1.axhline(exp, color="0.35", ls="--", lw=1.4, zorder=4)
    # inside the axes: at the right edge this collided with panel (b)'s ylabel
    # the "all 32 N atoms" label moved to the caption; the dashed line stays
    if a.annotate:
        ax1.text(-0.50, exp * 0.975, f"all {exp} {ELEMENT} atoms in the box",
                 fontsize=9.5, color="0.35", va="top", ha="left",
                 fontweight="bold")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(a.labels, fontweight="bold")
    ax1.set_xlim(-0.55, 1.55)
    ax1.set_ylim(0, exp * 1.06)
    ax1.set_ylabel(f"{ELEMENT} atoms per box", fontsize=11)
    # >>> "THE PRODUCTS CHANGED" WAS A MISLEADING TITLE ON A NITROGEN BUDGET.
    # Water is the primary product of nitromethane decomposition in experiment,
    # in Han et al., and in these runs (0.46-0.50 H2O per nitromethane against
    # 0.16-0.30 N2). It carries no nitrogen, so it is absent from this panel
    # entirely. Titled "the products changed", the panel reads as a ranking of
    # products and invites the conclusion that N2 is the main one. It is not.
    # This panel answers one question only: where does the NITROGEN go.
    if a.annotate:
        ax1.set_title(f"(a)  where the {ELEMENT}itrogen goes"
                      if ELEMENT == "N" else f"(a)  where the {ELEMENT} goes",
                      loc="left", fontweight="bold", fontsize=11.5)
    ax1.grid(alpha=.18, axis="y")
    _panel_letter(ax1, "a")
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=12.5, loc="upper center", bbox_to_anchor=(0.5, -0.055),
               ncol=2, framealpha=.95, handlelength=1.4, columnspacing=1.2)

    # ---- (b) the null ----------------------------------------------------
    for r, lab, col in zip(a.runs, a.labels, ("#b03030", "#0f7d7d")):
        t, y, th, _ = surv[r]
        ax2.plot(t, y, color=col, lw=3.0, zorder=5,
                 label=f"{lab}   t½ = {th:.2f} ps")
    ax2.axhline(0.5, color="0.55", ls="--", lw=1.2, zorder=2)
    ax2.set_xlim(0, a.tmax)
    ax2.set_ylim(0, 1.06)
    ax2.set_xlabel("time from start of production (ps)", fontsize=11)
    ax2.set_ylabel("nitromethane remaining, per molecule", fontsize=11)
    if a.annotate:
        ax2.set_title("(b)  the rate did not", loc="left", fontweight="bold",
                      fontsize=11.5)
    ax2.grid(alpha=.2)
    ax2.legend(fontsize=13.5, loc="upper right", framealpha=.95)
    _panel_letter(ax2, "b")
    d = abs(surv[a.runs[1]][2] - surv[a.runs[0]][2])
    if a.annotate:
        ax2.annotate(f"{d*1000:.0f} fs apart\n= {d/surv[a.runs[0]][3]:.0f} frame",
                     xy=(0.30, 0.5), xytext=(0.42, 0.30),
                     xycoords=("axes fraction", "data"),
                     textcoords="axes fraction", fontsize=10, color="0.3",
                     fontweight="bold")
        fig.suptitle("The correction reroutes the nitrogen, not the rate",
                     fontsize=14, fontweight="bold", y=0.968)
    print(f"    half-lives differ by {d*1000:.0f} fs = "
          f"{d/surv[a.runs[0]][3]:.0f} frame(s) at {surv[a.runs[0]][3]*1000:.0f} fs")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=300)
    print(f"\n  wrote {a.out}")

    os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
    with open(a.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sink"] + [f"{l}_{s}" for l in a.labels
                               for s in ("mean", "sd")])
        for lab, _, _ in GROUPS:
            w.writerow([lab] + [f"{f(gs if s == 'mean' else sds)[lab]:.4f}"
                                for r in a.runs
                                for s, f in (("mean", lambda d, r=r: d[r]),
                                             ("sd", lambda d, r=r: d[r]))])
    print(f"  wrote {a.csv}")

    print("\n  CAPTION MUST STATE:")
    print(f"    window         {a.from_ps:.0f}-{a.to_ps:.0f} ps, "
          f"{len(frames)} frames, one trajectory per model")
    print(f"    conservation   every bar sums to {exp}, the {ELEMENT} atoms in "
          f"the box")
    print(f"    connectivity   {a.scale}x the summed covalent radii; the slate "
          f"is threshold-dependent")
    print( "    error bars     fluctuation of each sink across the window, NOT")
    print( "                   run-to-run reproducibility. One seed per model.")
    print( "    NOT A RANKING  water is the primary product overall and holds no")
    print( "                   nitrogen, so it does not appear in panel (a) at all.")
    tail = sorted(set(tails[a.runs[0]]) | set(tails[a.runs[1]]))
    print(f"    'other' holds  {', '.join(tail[:14])}"
          + (" ..." if len(tail) > 14 else ""))


if __name__ == "__main__":
    main()
