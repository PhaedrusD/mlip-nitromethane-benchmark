#!/usr/bin/env python
"""
plot_ablation_scans.py -- the rise, and the rise switched off, for both models
===============================================================================

Two panels, one per OMol25-derived model: the C-N dissociation curve intact,
then with the label channel scaled down. The step at the 6 A cutoff flattens.

>>> WHY THIS AND NOT THE LADDER PLOT
    A seam-against-lambda plot is the more complete summary and it is the
    weaker figure. The reader of this paper has already met the spurious rise
    as a feature of a potential energy curve, so showing the same curve with
    the feature disappearing is a direct before-and-after in the reader's
    existing frame. A ladder plot asks them to accept a scalar called "seam"
    and trust that it stands for the thing they saw earlier.

    The dose-response evidence does not disappear; it moves to the text, where
    the non-monotonicity of the composition channel can be stated in a sentence
    (see SS28). Keep ablation_*_ladder_summary.csv for the SI.

>>> WHICH CHANNEL EACH PANEL SHOWS, AND WHY THEY DIFFER
    UMA-S-1P1   spin embedding alone. Falls monotonically, 1.068 -> 0.067 eV.
    MACE-OMOL-0 charge and spin lookups together. Spin alone never clears the
                seam in this model; the two are concatenated and then
                projected, so a one-sided change creates an imbalance the
                projection was never trained on.

    Do NOT substitute UMA's composition embedding here, even though it clears
    the seam at a lighter dose and scores far better against the sham. Its
    dose-response reverses: the seam returns at lambda 0.25 and 0. A channel
    that restores the defect when you remove more of it is not behaving like a
    cause, and a scan figure cannot show that, which is exactly why it must not
    be the channel the scan figure advertises.

>>> ENERGY IS PLOTTED RELATIVE TO EACH CURVE'S OWN MINIMUM
    Scaling an embedding shifts the whole surface, so absolute energies are not
    comparable between doses and the comparison being made here is about SHAPE.
    E_rel_eV in the source CSVs is already referenced this way.

USAGE
    python plot_ablation_scans.py --out MLIP_paper/fig_ablation_scans.png
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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

KCAL = 23.0605
CUTOFF = 6.0

PANELS = [
    dict(label="UMA-S-1P1", channel="spin embedding",
         csv="MLIP_paper/ablation_uma_ladder_curves.csv",
         summary="MLIP_paper/ablation_uma_ladder_summary.csv",
         doses=["0.5", "0.25", "0"], shades=["#d7c4ea", "#a97fd0", "#7a3fb0"],
         chan_label="spin embedding"),
    dict(label="MACE-OMOL-0", channel="both lookups",
         csv="MLIP_paper/ablation_omol_ladder_curves.csv",
         summary="MLIP_paper/ablation_omol_ladder_summary.csv",
         doses=["0.9", "0.8", "0.75", "0.7"], shades=["#c9d9f7", "#8fb0ea", "#4f7fdf", "#1f4fd8"],
         chan_label="charge + spin lookups"),
]


def load_curves(path):
    out = collections.defaultdict(lambda: ([], []))
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                x = float(r["bond_A"])
                y = float(r["E_rel_eV"])
            except (KeyError, ValueError):
                continue
            k = (r["target"], r["lambda"])
            out[k][0].append(x)
            out[k][1].append(y)
    return {k: (np.array(v[0]), np.array(v[1])) for k, v in out.items()}


def seam_of(path, target, lam):
    """Remaining seam in eV, read from the summary rather than recomputed."""
    for r in csv.DictReader(open(path)):
        if r["target"] == target and float(r["lambda"]) == float(lam):
            return float(r["seam_eV"])
    return None



def _panel_letter(ax, letter):
    """Bold panel letter just outside the top-left corner, journal style.
    Outside the axes so it can never land on data, a legend or a colourbar."""
    ax.text(0.0, 1.015, f"({letter})", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=13, fontweight="bold")

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--units", choices=["eV", "kcal"], default="kcal")
    ap.add_argument("--xmax", type=float, default=8.0)
    ap.add_argument("--out", default="MLIP_paper/fig_ablation_scans.png")
    ap.add_argument("--annotate", action="store_true",
                    help="slide version: suptitle and two-line panel titles")
    a = ap.parse_args()

    f = KCAL if a.units == "kcal" else 1.0
    ylab = ("energy rel. to own minimum "
            + ("(kcal mol$^{-1}$)" if a.units == "kcal" else "(eV)"))

    fig, axes = plt.subplots(1, 2, figsize=poster_style.figsize(13.2, 5.6),
                             sharey=True)
    # left margin holds the y-label. With --annotate the top also holds a
    # suptitle AND two-line panel titles, which poster_style renders taller
    # than matplotlib does; without it only the panel letters sit up there.
    fig.subplots_adjust(top=0.74 if a.annotate else 0.93, bottom=0.145,
                        left=0.105, right=0.985, wspace=0.07)

    for ax, p, letter in zip(axes, PANELS, "ab"):
        cur = load_curves(p["csv"])
        intact = next((v for k, v in cur.items() if k[0] == "intact"), None)
        if intact is None:
            sys.exit(f"  no intact curve in {p['csv']}")

        # >>> THE CUTOFF REGION IS SHADED, NOT ANNOTATED WITH AN ARROW.
        # The claim is that the step sits AT the cutoff, so the cutoff has to
        # be on the figure as a position, not as a label pointing at one.
        ax.axvspan(CUTOFF, a.xmax, color="0.88", alpha=.55, zorder=0)
        ax.axvline(CUTOFF, color="0.45", ls="--", lw=1.4, zorder=2)

        m = intact[0] <= a.xmax
        ax.plot(intact[0][m], intact[1][m] * f, color="0.1", lw=3.4, zorder=6,
                label="intact")

        s0 = seam_of(p["summary"], p["channel"], p["doses"][0])
        for lam, col in zip(p["doses"], p["shades"]):
            k = (p["channel"], lam)
            if k not in cur:
                print(f"  !! {p['label']}: no curve for lambda={lam}")
                continue
            x, y = cur[k]
            mm = x <= a.xmax
            sm = seam_of(p["summary"], p["channel"], lam)
            ax.plot(x[mm], y[mm] * f, color=col, lw=2.4, zorder=5,
                    label=f"λ = {float(lam):g}"
                          + (f"   (rise {sm*f:.1f})" if sm is not None else ""))

        si = seam_of(p["summary"], p["channel"], "1.0")
        intact_seam = None
        for r in csv.DictReader(open(p["summary"])):
            intact_seam = float(r["seam_eV"]) - float(r["seam_change_eV"])
            break
        if a.annotate:
            ax.set_title(f"{p['label']}\nscaling the {p['chan_label']}",
                         fontsize=12, fontweight="bold", pad=10)
        else:
            # model name as an in-axes label, not a title; the channel being
            # scaled goes in the caption
            ax.text(0.03, 0.97, p["label"], transform=ax.transAxes,
                    ha="left", va="top", fontsize=15, fontweight="bold")
        _panel_letter(ax, letter)
        ax.set_xlim(1.2, a.xmax)
        ax.set_xlabel("C–N distance (Å)", fontsize=11)
        ax.grid(alpha=.2)
        ax.legend(fontsize=17, loc="lower right", framealpha=.95,
                  title=f"rise intact: {intact_seam*f:.1f}",
                  title_fontsize=17, handlelength=2.4, borderpad=0.6,
                  labelspacing=0.45)
        print(f"  [{p['label']:<12}] intact seam {intact_seam*f:.2f} "
              f"{'kcal/mol' if a.units=='kcal' else 'eV'}")
        for lam in p["doses"]:
            sm = seam_of(p["summary"], p["channel"], lam)
            if sm is not None:
                print(f"      λ={float(lam):<5g} seam {sm*f:6.2f}  "
                      f"({100*(1-sm/intact_seam):5.1f}% removed)")

    axes[0].set_ylabel(ylab, fontsize=11)
    if a.annotate:
        fig.suptitle("Scaling the global label channel removes the rise at "
                     "the cutoff", fontsize=13.5, fontweight="bold", y=0.978)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=300)
    print(f"\n  wrote {a.out}")

    print("\n  CAPTION MUST STATE:")
    print(f"    shaded band    beyond the {CUTOFF:.1f} Å interaction cutoff")
    print( "    energies       relative to each curve's own minimum, because")
    print( "                   scaling an embedding shifts the whole surface")
    print( "    channels       UMA spin embedding alone; MACE-OMOL-0 needs the")
    print( "                   charge and spin lookups scaled together")
    print( "    NOT SHOWN      the dose-response ladder and the sham controls;")
    print( "                   those belong in the text and the SI")


if __name__ == "__main__":
    main()
