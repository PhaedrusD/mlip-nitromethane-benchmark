#!/usr/bin/env python
"""
plot_zbl_split.py -- the ZBL story as TWO figures, each with one sentence
=========================================================================

  zbl_failure.png   PROCEDURE section: what goes wrong, and how we detect it
  zbl_results.png   RESULTS section:   what ZBL does, and what it must not do

>>> WHY THE PREVIOUS PANELS WERE REBUILT
    The first version showed (b) as a probability-density histogram and (c) as
    21 grouped bars. Both are technically correct and both failed the only test
    that matters: the presenter could not say what they meant in one sentence.

    (b) A density histogram asks the viewer to compare three overlapping curves
        and read a y-axis in units nobody thinks in. The actual claim is about
        ONE NUMBER per arm -- how close did atoms ever get -- so plot that
        number, once per arm, against the distance where runs die.

    (c) Grouped bars ask the viewer to compare 21 heights and notice that 14 of
        them match. The claim is "did the chemistry change?", which is a
        RATIO question: population divided by the no-ZBL control. On that axis
        1.0 means unchanged, and the eye only has to check whether points sit
        on a line.

    Same data, same conclusions, one sentence each.

SENTENCES THESE FIGURES SUPPORT
  failure: "The run sat at a normal 0.82 A for 65 ps, then one atom pair fell
            to 0.096 A in a single frame and the integrator died."
  results: "Both ZBL variants keep atoms further apart -- but only the MACE one
            leaves the product chemistry alone."

USAGE
    python plot_zbl_split.py --root rerun3 --outdir MLIP_paper
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# >>> POSTER TYPE. Every fontsize= below is a RELATIVE number; poster_style
# multiplies all of them by one factor and forces bold, so the size hierarchy
# this script encodes survives and only the overall scale changes.
# POSTER_SCALE=1.0 POSTER_GROW=1.0 restores the original look exactly.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import poster_style
poster_style.apply()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "files"))

# no-ZBL blue, MACE red, standard teal: panel (b) shows only the two ZBL arms,
# and red vs teal are distinguishable where blue vs teal were not
ARMS = [("zbl-none",     "#1f4fd8", "no ZBL\n(control)"),
        ("zbl-mace",     "#b03030", "MACE ZBL"),
        ("zbl-textbook", "#0f7d7d", "standard ZBL")]
THRESH = 0.4
T0 = 20.0          # steady state starts here


def arm_dir(root, tag):
    h = [d for d in glob.glob(os.path.join(root, "*")) if d.endswith(tag)]
    return h[0] if h else None


def load(d):
    from ase.io.trajectory import Trajectory
    t = sorted(glob.glob(os.path.join(d, "*.traj")))
    return list(Trajectory(t[0])) if t else None


def min_dists(traj, stride):
    out = []
    for i in range(0, len(traj), stride):
        a = traj[i]
        m = a.get_all_distances(mic=True)
        np.fill_diagonal(m, np.inf)
        out.append(float(m.min()))
    return np.array(out)


# ---------------------------------------------------------------- FIGURE 1
def fig_failure(mindist_csv, out):
    """One panel. One statement: stable for 65 ps, then gone in one frame."""
    d = np.genfromtxt(mindist_csv, delimiter=",", names=True)
    t, md = d["time_ps"], d["min_dist_A"]
    pre = md[t < t[-1] - 1.0]
    mu, sd = float(pre.mean()), float(pre.std())
    nsig = (mu - md.min()) / sd

    fig, ax = plt.subplots(figsize=poster_style.figsize(9.6, 5.4))
    # margins reserved up front -- see the note in fig_results()
    fig.subplots_adjust(top=0.83, bottom=0.25, left=0.115, right=0.985)
    ax.plot(t, md, color="#333", lw=0.8, zorder=3)
    ax.axhspan(mu - 3 * sd, mu + 3 * sd, color="#1f4fd8", alpha=.12, zorder=0)
    ax.text(2, mu + 3 * sd + .015,
            f"normal range: {mu:.2f} ± {sd:.2f} Å", fontsize=10, color="#1f4fd8",
            fontweight="bold", va="bottom")
    ax.axhspan(0, THRESH, color="#c0392b", alpha=.10, zorder=0)
    ax.text(2, THRESH - .04, "atoms this close ⇒ the run dies",
            fontsize=10, color="#c0392b", fontweight="bold", va="top")

    i = int(np.argmin(md))
    ax.annotate(f"one 25 fs frame:\n0.86 Å → {md.min():.3f} Å\n"
                f"a {nsig:.0f}σ jump",
                # pushed right and up: at poster type this callout sat on top
                # of the "atoms this close" label above the red band
                xy=(t[i], md.min()), xytext=(t[-1] * .66, 0.235),
                fontsize=11, color="#c0392b", fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2.0))
    ax.set_xlabel("time (ps)", fontsize=11)
    ax.set_ylabel("closest distance between\nany two atoms (Å)", fontsize=11)
    ax.set_ylim(0, 1.05)
    # two lines: at poster type this sentence is wider than the canvas
    ax.set_title("MACE-Osaka24 crystal at 3000 K\n"
                 "stable for 65 ps, then gone in one frame",
                 fontsize=12.5, fontweight="bold")
    ax.grid(alpha=.2)
    fig.text(0.5, 0.015,
             "The temperature was flat right up to the event (+66 K over 65 ps),\n"
             "so this is not the model slowly overheating. It is a rare "
             "fluctuation that carries two atoms\ninside the range the training "
             "data covered, where nothing pushes back.",
             ha="center", va="bottom", fontsize=9, style="italic", color="#555")
    fig.savefig(out, dpi=300)
    print(f"  wrote {out}")
    print(f"    normal {mu:.3f} ± {sd:.3f} Å   min {md.min():.3f} Å   {nsig:.1f} sd")


# ---------------------------------------------------------------- FIGURE 2

def _panel_letter(ax, letter):
    """Bold panel letter just outside the top-left corner, journal style.
    Outside the axes so it can never land on data, a legend or a colourbar."""
    ax.text(0.0, 1.015, f"({letter})", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=13, fontweight="bold")

def fig_results(root, outdir, stride, sstride, roll, annotate=False):
    from species_analysis import species_populations, SPECIES_ORDER
    floors, pcts, pops_all = {}, {}, {}
    for tag, col, lab in ARMS:
        d = arm_dir(root, tag)
        traj = load(d) if d else None
        if traj is None:
            print(f"  [skip] {tag}")
            continue
        md = min_dists(traj, stride)
        floors[tag] = md.min()
        pcts[tag] = float(np.percentile(md, 0.1))
        tp, p = species_populations(traj[::sstride], dt_fs=0.1,
                                    log_stride=250 * sstride, roll_ps=roll)
        pops_all[tag] = (np.asarray(tp), p)
        print(f"  {tag:<14} floor {md.min():.3f} Å   P0.1 {pcts[tag]:.3f}")

    tags = [t for t, _, _ in ARMS if t in floors]
    cols = dict((t, c) for t, c, _ in ARMS)
    labs = dict((t, l) for t, _, l in ARMS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=poster_style.figsize(13.4, 5.6),
                                   gridspec_kw=dict(width_ratios=[1.0, 1.25]))
    # >>> RESERVE THE MARGINS, DO NOT LET bbox_inches RECOVER THEM.
    # suptitle, both panel titles and the footer live in FIGURE coordinates
    # while the axes filled the canvas. At poster type they landed on top of
    # each other; cropping afterwards cannot undo an overlap.
    # >>> PUBLICATION LAYOUT. No suptitle, no panel titles, no footer, no
    # explanatory boxes: the caption carries all of it. Margins shrink to suit.
    # --annotate restores the slide version.
    fig.subplots_adjust(top=0.92, bottom=0.16, left=0.085, right=0.985,
                        wspace=0.28)

    # ---- (a) one number per arm: how close did atoms ever get? -------------
    x = np.arange(len(tags))
    ax1.bar(x, [floors[t] for t in tags], 0.55,
            color=[cols[t] for t in tags], edgecolor="white", lw=1.0, zorder=3)
    for i, t in enumerate(tags):
        ax1.text(i, floors[t] + .012, f"{floors[t]:.3f} Å", ha="center",
                 fontsize=12, fontweight="bold", color=cols[t])
    # >>> THE "RUN DIES BELOW HERE" LINE IS OFF BY DEFAULT.
    # That threshold came from the crystal collapse, not from any run drawn
    # here; the ungated control bottoms out at 0.575 A and never approached it.
    # Drawn, it implies the control was close to failing. It was not.
    if annotate:
        ax1.axhspan(0, THRESH, color="#c0392b", alpha=.10, zorder=0)
        ax1.axhline(THRESH, color="#c0392b", ls="--", lw=1.4)
        ax1.text(len(tags) - 0.5, THRESH - .02, "the run dies below here",
                 fontsize=9.5, color="#c0392b", ha="right", va="top",
                 fontweight="bold")
    base = floors[tags[0]]
    if annotate:   # the +0.039 / +0.053 deltas are in the caption
        for i, t in enumerate(tags[1:], start=1):
            ax1.annotate("", xy=(i, floors[t]), xytext=(i, base),
                         arrowprops=dict(arrowstyle="->", color="0.3", lw=1.6))
            ax1.text(i + .30, (base + floors[t]) / 2, f"+{floors[t]-base:.3f}",
                     fontsize=9.5, color="0.3", va="center")
    ax1.set_xticks(x)
    ax1.set_xticklabels([labs[t] for t in tags], fontsize=9)
    # wrapped: one line is taller than the axes once the type grows
    ax1.set_ylabel("closest approach (Å)", fontsize=11)
    ax1.set_ylim(0, max(floors.values()) * 1.55)
    # room on the right for the "+0.053" delta labels, which at poster type are
    # wider than the gap the default x-limits leave
    ax1.set_xlim(-0.62, len(tags) - 0.10)
    if annotate:
        ax1.set_title("(a)  ZBL keeps atoms apart", loc="left",
                      fontweight="bold", fontsize=12.5)
    ax1.grid(alpha=.2, axis="y")
    _panel_letter(ax1, "a")

    # ---- (b) ratio to control: did the chemistry change? -------------------
    ctrl = pops_all["zbl-none"][1]
    # >>> PRODUCTS ONLY. CH3NO2 is leftover REACTANT, not a product, and its
    # steady-state mean is small enough (0.04) that the ratio carries a +-50%
    # error bar that runs off the axis and dominates the panel. "other" is a
    # catch-all with no chemical meaning. Both are excluded, and the caption
    # says the panel is about products.
    EXCLUDE = {"other", "CH3NO2 (NM+isomers)", "CH3NOOH (+H)"}
    show = [s for s in SPECIES_ORDER
            if s not in EXCLUDE
            and np.asarray(ctrl.get(s, [0]))[pops_all["zbl-none"][0] >= T0].mean() > 0.02]

    def mean_of(tag, s):
        tp, p = pops_all[tag]
        return float(np.asarray(p[s])[tp >= T0].mean())

    def sd_of(tag, s):
        tp, p = pops_all[tag]
        return float(np.asarray(p[s])[tp >= T0].std())

    xs = np.arange(len(show))
    ax2.axhspan(0.75, 1.25, color="#7fbf9a", alpha=.16, zorder=0)
    ax2.axhline(1.0, color="0.25", lw=1.6, zorder=1)
    # pulled in from the axis edge: at poster type "unchanged" is wide enough
    # that right-aligning it at the last tick ran it off the frame
    if annotate:
        ax2.text(len(show) - 1.15, 1.02, "unchanged", fontsize=10, color="0.3",
                 ha="right", fontweight="bold")
    for j, tag in enumerate(t for t in tags if t != "zbl-none"):
        off = (j - 0.5) * 0.22
        r = [mean_of(tag, s) / mean_of("zbl-none", s) for s in show]
        e = [sd_of(tag, s) / mean_of("zbl-none", s) for s in show]
        ax2.errorbar(xs + off, r, yerr=e, fmt="o", ms=11, capsize=4,
                     color=cols[tag], label=labs[tag].replace("\n", " "),
                     lw=0, elinewidth=1.6, markeredgecolor="white",
                     markeredgewidth=1.0, zorder=4)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([s.split(" (")[0] for s in show], fontsize=10.5)
    ax2.set_ylabel("product ratio, ZBL / no ZBL", fontsize=11)
    ax2.set_ylim(0, 1.6)
    if annotate:
        ax2.set_title("(b)  …but only one keeps the chemistry",
                      loc="left", fontweight="bold", fontsize=12.5)
    ax2.legend(fontsize=13, loc="lower left", framealpha=.95)
    ax2.grid(alpha=.2, axis="y")
    _panel_letter(ax2, "b")
    ratios = [mean_of("zbl-textbook", s) / mean_of("zbl-none", s) for s in show]
    worst = min(ratios)
    # >>> PARKED IN THE EMPTY LOWER-LEFT, NOT THE MIDDLE.
    # Centred, the box covered the two textbook-ZBL points it is describing --
    # the reader was asked to take the claim on faith with the evidence hidden
    # underneath it. Below 0.6 on the left there is nothing plotted.
    if annotate:
        ax2.text(0.015, 0.03, f"textbook ZBL wipes out\n"
                            f"up to {1/worst:.0f}× of the products:\n"
                            f"not a floor, a different\npotential",
                 transform=ax2.transAxes, fontsize=9.5, color="#0f7d7d",
                 fontweight="bold", ha="left", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#0f7d7d",
                           lw=1.2, alpha=.95))
        fig.suptitle("Remediating MACE-Osaka24: a repulsive floor works, "
                     "but only at the right strength",
                     fontsize=14.5, fontweight="bold", y=0.975)
        fig.text(0.5, 0.015,
                 "Liquid nitromethane, 1.97 g/cc, 3000 K, 150 ps, one run per arm.\n"
                 "(b) shows PRODUCTS only; error bars are the fluctuation of each "
                 "product about its steady-state mean.\n"
                 "ZBL is exactly zero beyond ~1 Å, so a point off the line in (b) "
                 "means the correction is doing more than acting as a floor.",
                 ha="center", va="bottom", fontsize=9, style="italic", color="#555")
    # the caption now owes these numbers; print them so it can be written from them
    print("  CAPTION MUST STATE: liquid, 1.97 g/cc, 3000 K, 150 ps, one run per arm;")
    print("    (a) floors:", "  ".join(f"{labs[t].splitlines()[0]} {floors[t]:.3f} A" for t in tags))
    print("    (b) products only, ratio to the ungated run; error bars = fluctuation")
    print("        of each product about its steady-state mean, NOT run-to-run spread.")
    print(f"    worst textbook ratio {worst:.2f} ({1/worst:.1f}x suppression)")
    out = os.path.join(outdir, "zbl_results.png")
    fig.savefig(out, dpi=300)
    print(f"  wrote {out}")
    for s in show:
        print(f"    {s:<22}" + "".join(
            f"{mean_of(t, s)/mean_of('zbl-none', s):>8.2f}" for t in tags))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="rerun3")
    ap.add_argument("--mindist", default="files/osaka_crystal_mindist.csv")
    ap.add_argument("--outdir", default="MLIP_paper")
    ap.add_argument("--annotate", action="store_true",
                    help="slide version: titles, footer, explanatory box, and "
                         "the crystal-derived 'run dies below here' line")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--species-stride", type=int, default=20)
    ap.add_argument("--roll-ps", type=float, default=2.0)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    fig_failure(a.mindist, os.path.join(a.outdir, "zbl_failure.png"))
    fig_results(a.root, a.outdir, a.stride, a.species_stride, a.roll_ps,
                annotate=a.annotate)


if __name__ == "__main__":
    main()
