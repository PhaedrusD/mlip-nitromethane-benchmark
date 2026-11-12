#!/usr/bin/env python
"""
plot_mrr_remediation.py -- what the fine-tuning bought, and what it cost
=========================================================================

ONE figure for the ANI-1xnr-Mrr subsection, replacing the 26-row arrow plot.

>>> WHY THE ARROW PLOT GOES
    Its content is already in the prose, with numbers: radical-pair MAE 28.3 ->
    4.8, all nine improved, CH3NO2 C-N -50.1 -> -0.6, atom-producing 14.3 ->
    14.6. A figure that restates the text is the first one to cut when space is
    short. What is NOT in the prose is the cost: one bond (N2) is quoted, and
    the 25-of-26 collapse and its dependence on whether the error improved
    appear nowhere.

>>> WHAT EACH PANEL SAYS

    (a) "The correction pulled the radical-pair bonds onto zero and left the
         atom-producing bonds scattered."
        Signed error, four strips: each class before and after. SIGNED, not
        absolute -- the direction is the mechanism. ANI-1xnr under-binds every
        radical pair (all negative), which is exactly the shape a Morse tail
        correction can fix, and why it does nothing for the rest.

        Four strips instead of 26 labelled rows costs the per-bond identity and
        buys back two thirds of the vertical space. The identity lives in
        mrr_bde_table.csv and in the text.

    (b) "The members stopped disagreeing -- hardest where the error did not
         improve."
        Ensemble spread before against after, log-log. Filled marks are bonds
        whose |error| improved, open marks are bonds where it did not. The open
        marks sitting FURTHER below the diagonal is the whole argument: the
        ensemble lost its spread most completely on the bonds the fine-tuning
        failed to fix.

>>> THE ENCODING IS CONSISTENT ACROSS BOTH PANELS
    Colour = bond class. Fill = whether |error| improved. A reader learns the
    key once. Do not add a third channel.

>>> SPREAD CONVENTION
    --ddof 0 (default) is the POPULATION standard deviation over the eight
    members, which is what the draft's "+/-55.7 -> +/-1.4" for N2 is. The eight
    members are the entire ensemble, not a sample drawn from one, so ddof=0 is
    the defensible choice -- but the summary CSV stores ddof=1, so this script
    converts. State the convention in the caption either way.

USAGE
    python plot_mrr_remediation.py \\
        --csv MLIP_paper/mrr_ensemble_spread_summary.csv \\
        --out MLIP_paper/fig_mrr_remediation.png
"""

import argparse
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

N_MEMBERS = 8
C_RAD = "#1f7a4d"        # radical-pair: the class the correction targeted
C_ATOM = "#b5651d"       # atom-producing: the class it did not


def load(path, ddof):
    """Rows, with spread converted to the requested ddof.

    The CSV stores ddof = 1. Converting is exact for a fixed n:
        std(ddof=0) = std(ddof=1) * sqrt((n-1)/n)
    Recomputing from the per-member file would also work but risks the two
    figures in this section disagreeing over a rounding choice.
    """
    k = 1.0 if ddof == 1 else np.sqrt((N_MEMBERS - 1) / N_MEMBERS)
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append(dict(
                    species=r["species"], bond=r["bond"],
                    rad=(r["cls"] == "radical-pair"),
                    err_base=float(r["err_base"]), err_mrr=float(r["err_mrr"]),
                    std_base=float(r["std_base"]) * k,
                    std_mrr=float(r["std_mrr"]) * k,
                    improved=abs(float(r["err_mrr"])) < abs(float(r["err_base"]))))
            except (KeyError, ValueError):
                continue
    return rows


def strip(ax, y, vals, color, rng):
    """One horizontal strip of points, jittered so overlaps are visible."""
    j = rng.uniform(-0.13, 0.13, size=len(vals))
    ax.scatter(vals, np.full(len(vals), y) + j, s=58, color=color,
               edgecolor="white", lw=1.0, zorder=4, alpha=0.95)
    # mean marker: the number the text quotes as an MAE lives here
    ax.plot([np.mean(np.abs(vals))], [y], "|", ms=0)     # keeps API symmetric



def _panel_letter(ax, letter):
    """Bold panel letter just outside the top-left corner, journal style.
    Outside the axes so it can never land on data, a legend or a colourbar."""
    ax.text(0.0, 1.015, f"({letter})", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=13, fontweight="bold")

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="MLIP_paper/mrr_ensemble_spread_summary.csv")
    ap.add_argument("--annotate", action="store_true",
                    help="slide version: panel titles")
    ap.add_argument("--out", default="MLIP_paper/fig_mrr_remediation.png")
    ap.add_argument("--ddof", type=int, default=0, choices=[0, 1])
    ap.add_argument("--seed", type=int, default=0, help="jitter reproducibility")
    a = ap.parse_args()

    rows = load(a.csv, a.ddof)
    if not rows:
        sys.exit(f"  no usable rows in {a.csv}")
    rng = np.random.default_rng(a.seed)

    rad = [r for r in rows if r["rad"]]
    atom = [r for r in rows if not r["rad"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=poster_style.figsize(12.4, 5.0),
                                   gridspec_kw=dict(width_ratios=[1.25, 1.0]))
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.145, top=0.855,
                        wspace=0.30)

    # ------------------------------------------------------ (a) signed error
    # >>> ORDER: after ABOVE before within each class, so the eye reads the
    # move upward as "corrected". Classes separated by a gap, not a line.
    strips = [(3.0, rad,  "err_base", C_RAD,  "ANI-1xnr"),
              (2.4, rad,  "err_mrr",  C_RAD,  "ANI-1xnr-Mrr"),
              (1.1, atom, "err_base", C_ATOM, "ANI-1xnr"),
              (0.5, atom, "err_mrr",  C_ATOM, "ANI-1xnr-Mrr")]
    ax1.axvspan(-5, 5, color="0.93", zorder=0)
    ax1.axvline(0, color="0.3", lw=1.4, zorder=2)
    for y, grp, key, col, _ in strips:
        strip(ax1, y, np.array([g[key] for g in grp]), col, rng)
        m = np.mean(np.abs([g[key] for g in grp]))
        ax1.text(ax1.get_xlim()[1], y, "", va="center")   # placeholder, see below

    ax1.set_yticks([s[0] for s in strips])
    ax1.set_yticklabels([s[4] for s in strips], fontsize=9.5)
    ax1.set_ylim(0.05, 3.55)
    ax1.set_xlabel("signed error vs CCSD(T)  (kcal mol$^{-1}$)", fontsize=10.5)
    # >>> NO PANEL TITLES BY DEFAULT: the caption carries them. --annotate for slides.
    if a.annotate:
        ax1.set_title("(a)  radical pairs pulled onto zero", loc="left",
                      fontweight="bold", fontsize=11.5)
    ax1.grid(axis="x", alpha=.2)
    _panel_letter(ax1, "a")
    ax1.set_axisbelow(True)

    # >>> MAKE ROOM FOR THE MAE COLUMN BEFORE PLACING IT.
    # First version right-aligned the MAE inside the data limits and printed
    # "MAE 14.3" straight through the atom-producing points at +15 to +25.
    # Widening the axis is the fix; moving the text is not, because there is
    # nowhere inside the data that is reliably empty.
    x0, x1 = ax1.get_xlim()
    ax1.set_xlim(x0, x1 + 0.30 * (x1 - x0))
    xr = ax1.get_xlim()
    for y0, grp, col, lab in ((3.0, rad, C_RAD, f"radical-pair  (n={len(rad)})"),
                              (1.1, atom, C_ATOM,
                               f"atom-producing  (n={len(atom)})")):
        ax1.text(xr[0] + 0.03 * (xr[1] - xr[0]), y0 + 0.30, lab, fontsize=9.5,
                 color=col, fontweight="bold", va="bottom")
    for y, grp, key, col, _ in strips:
        m = float(np.mean(np.abs([g[key] for g in grp])))
        ax1.text(xr[1] - 0.01 * (xr[1] - xr[0]), y, f"MAE {m:.1f}",
                 fontsize=9, color=col, ha="right", va="center",
                 fontweight="bold")

    # ------------------------------------------------------ (b) spread
    sb = np.array([r["std_base"] for r in rows])
    sm = np.array([r["std_mrr"] for r in rows])
    lo, hi = max(min(sb.min(), sm.min()) * 0.55, 1e-3), max(sb.max(), sm.max()) * 2.0
    ax2.plot([lo, hi], [lo, hi], color="0.45", ls="--", lw=1.4, zorder=2)
    for f in (10, 100):
        ax2.plot([lo, hi], [lo / f, hi / f], color="0.82", ls=":", lw=1.1,
                 zorder=1)
        if a.annotate:   # the caption names the dotted lines
            ax2.text(hi * 0.9, hi / f * 0.9, f"{f}×", fontsize=9, color="0.55",
                     ha="right", va="top")

    # >>> FILL CARRIES THE ARGUMENT. Open marks are bonds the fine-tuning did
    # NOT fix; they sit further below the diagonal than the filled ones.
    for r in rows:
        col = C_RAD if r["rad"] else C_ATOM
        ax2.scatter(r["std_base"], r["std_mrr"], s=72, zorder=5, lw=1.6,
                    color=col if r["improved"] else "white",
                    edgecolor="white" if r["improved"] else col)
    # per-species offsets so NO and CN, which sit side by side, do not collide
    OFF = {"NO": (-4, 12), "CN": (2, -20), "N2": (8, -4), "N2H4": (8, -4)}
    for r in rows:
        if r["std_base"] / max(r["std_mrr"], 1e-9) >= 10:
            ax2.annotate(r["species"], (r["std_base"], r["std_mrr"]),
                         textcoords="offset points",
                         xytext=OFF.get(r["species"], (8, -4)),
                         fontsize=8, color="0.25")

    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi)
    lbl = "population s.d." if a.ddof == 0 else "sample s.d."
    # >>> SHORT LABEL. At poster type the full sentence overran the canvas;
    # the convention belongs in the caption, where it can be said properly.
    ax2.set_xlabel("spread, ANI-1xnr (kcal mol$^{-1}$)", fontsize=10.5)
    ax2.set_ylabel("spread, ANI-1xnr-Mrr (kcal mol$^{-1}$)", fontsize=10.5)
    if a.annotate:
        ax2.set_title("(b)  members stop disagreeing", loc="left",
                      fontweight="bold", fontsize=11.5)
    ax2.grid(alpha=.2, which="both")
    _panel_letter(ax2, "b")

    h = [plt.Line2D([], [], ls="", marker="o", ms=8, color=C_RAD,
                    mec="white", label="radical-pair"),
         plt.Line2D([], [], ls="", marker="o", ms=8, color=C_ATOM,
                    mec="white", label="atom-producing"),
         plt.Line2D([], [], ls="", marker="o", ms=8, color="white",
                    mec="0.35", mew=1.6, label="|error| NOT improved")]
    ax2.legend(handles=h, fontsize=13, loc="upper left", framealpha=.95)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=300)
    print(f"  wrote {a.out}")

    imp = np.array([r["improved"] for r in rows])
    ratio = sm / sb
    print("\n  CAPTION MUST STATE:")
    print(f"    (a) signed error, {len(rad)} radical-pair and {len(atom)} "
          f"atom-producing bonds; shaded band ±5 kcal/mol")
    print(f"    (b) spread = {lbl} over {N_MEMBERS} members; dashed y = x; "
          f"dotted 10× and 100×")
    print(f"    open marks = |error| did not improve ({(~imp).sum()} of "
          f"{len(rows)} bonds)")
    print(f"\n  NUMBERS FOR THE TEXT")
    print(f"    spread narrowed on {(sm < sb).sum()}/{len(rows)}, "
          f"median factor {1/np.median(ratio):.1f}x")
    print(f"    median ratio, |error| improved     {np.median(ratio[imp]):.3f}")
    print(f"    median ratio, |error| NOT improved {np.median(ratio[~imp]):.3f}")
    for grp, nm in ((rad, "radical-pair"), (atom, "atom-producing")):
        b = np.mean([abs(g["err_base"]) for g in grp])
        m = np.mean([abs(g["err_mrr"]) for g in grp])
        w = sum(1 for g in grp if not g["improved"])
        print(f"    {nm:<15} MAE {b:5.1f} -> {m:5.1f}   "
              f"{len(grp)-w}/{len(grp)} improved")
    worse = [g for g in atom if not g["improved"]]
    if worse:
        d = max(abs(g["err_mrr"]) - abs(g["err_base"]) for g in worse)
        print(f"    largest single atom-producing degradation: +{d:.1f} kcal/mol")


if __name__ == "__main__":
    main()
