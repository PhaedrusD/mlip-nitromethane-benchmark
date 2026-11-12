#!/usr/bin/env python
"""
make_comparison_figure.py -- presentation figure from an existing scan JSON
===========================================================================

Reads a bde_scan.py results file and plots a chosen SUBSET of models. Nothing is
recomputed, so this is safe to re-run and re-style as often as you like.

WHY A SEPARATE SCRIPT
    plot_graph() inside bde_scan.py runs at the end of a scan and plots whatever
    is in `results`. For a talk you want a curated subset, a reference line, and
    a zoom on the region that carries the argument. Doing that by editing
    bde_scan.py means re-running scans to restyle a figure.

USAGE
    python make_comparison_figure.py                       # the curated set
    python make_comparison_figure.py --with-singlet        # adds NSE S=1
    python make_comparison_figure.py --models "ANI-1xnr,MACE-OMOL"

WHY THE TRIPLET IS THE DEFAULT
    C-N scission gives CH3(doublet) + NO2(doublet). Coupled high-spin that is a
    TRIPLET, and the triplet asymptote is degenerate with the broken-symmetry
    singlet at large r -- so S=3 is the state that corresponds to the UKS
    reference the other models are judged against. Forcing S=1 asks AIMNet2-NSE
    for a closed-shell description of two separating radicals, which correctly
    climbs toward the RESTRICTED surface (it lands on RKS, +163.6 vs 155.1).
    That is a real result but it is not the like-for-like comparison.
"""
import argparse, json, os
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

# --- references, rigid C-N at this geometry (from the status deck) ------------
# >>> VERIFY these correspond to THIS scan's geometry before presenting. <<<
UKS_TRUTH = 75.9     # broken-symmetry DFT: the physically correct asymptote
RKS_REF   = 155.1    # restricted DFT: what a closed-shell description gives

STYLE = {
    "ANI-1xnr":          dict(color="#0f7d7d", lw=2.0, ls="-"),
    "MACE-Osaka24":      dict(color="#2e8b30", lw=2.0, ls="-"),
    "MACE-OMOL":         dict(color="#1f4fd8", lw=2.0, ls="-"),
    "UMA-S-1P1":         dict(color="#a24bd6", lw=2.0, ls="-"),
    "AIMNet2-NSE (S=3)": dict(color="#c62828", lw=2.0, ls="-"),
    "AIMNet2-NSE (S=1)": dict(color="#c62828", lw=1.8, ls="--"),
    "MACE-OFF23":        dict(color="#e8860c", lw=2.0, ls="-"),
    "AIMNet2-rxn":       dict(color="#7b4a12", lw=2.0, ls="-"),
}

# >>> THE FIVE MODELS THE PAPER BENCHMARKS, AND ONLY THOSE.
# This list previously carried AIMNet2-NSE (S=3) in place of MACE-OFF23, which
# is in the json but was not in the curated set. Figure 1 must show the same
# five models as Table 1 and the rest of the paper; AIMNet2 is a separate probe
# and belongs in the SI if anywhere. Pass --models to override.
CURATED = ["ANI-1xnr", "MACE-OFF23", "MACE-Osaka24", "MACE-OMOL", "UMA-S-1P1"]


def load_dft(path, method="uks_bs"):
    """The reference dissociation CURVE, not just its asymptote.

    >>> WHY A CURVE AND NOT A HORIZONTAL LINE
        The figure previously drew the UKS asymptote as a dashed horizontal
        line at 75.9 kcal/mol. That is the reference BOND ENERGY, not the
        reference potential, and it cannot show where a model departs from the
        reference -- only where it ends up. A reader comparing shapes, which is
        what this paper's argument rests on, had nothing to compare against.

    Energies are referenced to the UKS minimum so the curve is on the same
    footing as every model curve, each of which is referenced to its own.
    """
    import csv as _csv
    import collections as _c
    by = _c.defaultdict(dict)
    with open(path) as fh:
        for row in _csv.DictReader(fh):
            if row.get("converged") not in (None, "", "1"):
                continue
            by[float(row["bond_A"])][row["method"]] = float(row["E_eV"])
    r = sorted(k for k in by if method in by[k])
    if not r:
        return None, None
    e = np.array([by[k][method] for k in r])
    # reference to the UKS minimum, matching the models' own-minimum convention
    return np.array(r), (e - e.min()) * 23.0605


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json",
                    default="MLIP-Benchmark-BDE-Heatmap/json_results/"
                            "scan_results_Nitromethane_C-N_2_rigid.json")
    ap.add_argument("--out", default="MLIP_paper/nitromethane_CN_comparison.png")
    ap.add_argument("--models", default=None,
                    help="comma-separated; default is the curated set")
    ap.add_argument("--dft", default="files/dft/dft_full/dft_energies_full.csv",
                    help="DFT reference scan; set empty to omit the curve")
    ap.add_argument("--with-rks", action="store_true",
                    help="also draw the restricted (RKS) curve, which runs to "
                         "~157 kcal/mol and compresses everything else")
    ap.add_argument("--with-singlet", action="store_true",
                    help="also draw AIMNet2-NSE (S=1), the restricted-like curve")
    # >>> DEFAULTS ARE NOW PUBLICATION DEFAULTS, NOT SLIDE DEFAULTS.
    # Running this with no flags produces the figure as it goes into the
    # manuscript: no title, no explanatory block. Both are opt-in for talks.
    ap.add_argument("--title", default="",
                    help="on-figure title; empty for publication. The caption "
                         "carries this in the manuscript.")
    ap.add_argument("--annotate", action="store_true",
                    help="restore the grey block explaining the shading and "
                         "the rise interval; for slides, not for the paper")
    a = ap.parse_args()

    d = json.load(open(a.json))
    want = ([m.strip() for m in a.models.split(",")] if a.models else list(CURATED))
    if a.with_singlet and "AIMNet2-NSE (S=1)" not in want:
        i = (want.index("AIMNet2-NSE (S=3)") + 1
             if "AIMNet2-NSE (S=3)" in want else len(want))
        want.insert(i, "AIMNet2-NSE (S=1)")
    want = [m for m in want if m in d]

    # The RKS line exists to explain restricted-like curves. With none shown it
    # just stretches the y-axis by 60 kcal/mol and flattens everything else.
    show_rks = any(m in want for m in ("AIMNet2-NSE (S=1)", "AIMNet2-rxn"))

    # >>> WIDER CANVAS, AXES ON THE LEFT TWO THIRDS.
    # The key block used to sit INSIDE the axes at top-left, on top of the
    # MACE-OFF23 rise. Moving it out is the only way to stop it covering a
    # curve, and that needs room made for it rather than taken from the plot.
    fig, ax = plt.subplots(figsize=poster_style.figsize(11.4, 5.2))
    fig.subplots_adjust(left=0.080, right=0.570, bottom=0.125, top=0.905)

    # >>> THE REFERENCE IS A CURVE. See load_dft.
    # Falls back to the horizontal asymptote only if the scan file is missing,
    # and says so, because a silent fallback would reintroduce exactly the
    # problem this change fixes.
    dft_r = dft_e = None
    if a.dft and os.path.exists(a.dft):
        dft_r, dft_e = load_dft(a.dft, "uks_bs")
    if dft_r is None:
        print(f"  !! no DFT scan at {a.dft!r}; drawing the asymptote as a "
              f"horizontal line instead")
        ax.axhline(UKS_TRUTH, color="0.25", ls="--", lw=1.3, zorder=1)
    else:
        # >>> THE REFERENCE IS DRAWN ON TOP, AND DASHED SO THE MODELS SHOW
        # THROUGH. MACE-OMOL and UMA track it to within a line width from 1.5
        # to 5 A -- which is the paper's point -- so at a lower zorder the
        # reference vanishes under them exactly where the reader needs to see
        # that it is being tracked.
        ax.plot(dft_r, dft_e, color="0.10", lw=2.0, zorder=7,
                dashes=(5, 3), solid_capstyle="butt")
        print(f"  DFT-UKS curve: {len(dft_r)} points, {dft_r[0]:.2f}-"
              f"{dft_r[-1]:.2f} Å, asymptote {dft_e[-1]:.1f} kcal/mol")
    if show_rks and a.with_rks and a.dft and os.path.exists(a.dft):
        rr, re_ = load_dft(a.dft, "rks")
        if rr is not None:
            ax.plot(rr, re_, color="0.70", ls=":", lw=1.8, zorder=1)

    # shade the region beyond the 6.0 A cutoff; unlabelled, see the caption
    ax.axvspan(6.0, 7.05, color="0.92", zorder=0)

    rows = []
    for m in want:
        r = np.asarray(d[m]["r"]); e = np.asarray(d[m]["E"])
        ax.plot(r, e, marker=".", ms=4, label=m, zorder=3, **STYLE.get(m, {}))
        g = lambda x: float(e[np.abs(r - x).argmin()])
        rows.append((m, g(7.0), g(7.0) - g(4.5)))

    ax.set_xlabel("r(C–N) / Å")
    ax.set_ylabel("ΔE / kcal mol$^{-1}$")
    # >>> NO ON-FIGURE TITLE. Journals set the figure number and title in the
    # caption; a title baked into the image duplicates it and cannot be edited
    # at proof stage. --title restores one for slides and posters.
    if a.title:
        ax.set_title(a.title, fontsize=11)
    ax.grid(color="lightgrey", ls="-", lw=0.4, zorder=0)
    ax.set_xlim(1.1, 7.05)

    # ---------------------------------------------------------------- the key
    # >>> HAND-BUILT, NOT ax.legend(), AND NOT MONOSPACE.
    # This block has to do two jobs at once: identify each curve by colour AND
    # carry the two numbers that make the argument. ax.legend() can only do the
    # first, and the old version did the second in a separate monospace box --
    # monospace because that is the only way to align columns inside a single
    # text string.
    #
    # Aligning columns by PLACING them at fixed x instead lets the whole block
    # use the same face as the rest of the figure, which is what a reader
    # notices. Everything here is in FIGURE coordinates, to the right of the
    # axes, so no curve is ever covered no matter how the data changes.
    # >>> THE NAME COLUMN NEEDS ~0.25 OF THE FIGURE WIDTH.
    # At 0.20 the longest label, "MACE-Osaka24", ran into its own right-aligned
    # asymptote and printed as "MACE-Osaka2465.1". Column collisions in a
    # hand-placed block are silent: nothing errors, the text just merges.
    X_SWATCH, X_NAME, X_ASYM, X_RISE = 0.600, 0.655, 0.905, 0.990
    y = 0.845
    DY = 0.072

    fig.text(X_NAME, y + 0.055, "model", fontsize=9.5, fontweight="bold",
             color="0.25", ha="left", va="baseline")
    fig.text(X_ASYM, y + 0.055, "D (7.0 Å)", fontsize=9.5, fontweight="bold",
             color="0.25", ha="right", va="baseline")
    fig.text(X_RISE, y + 0.055, "rise", fontsize=9.5, fontweight="bold",
             color="0.25", ha="right", va="baseline")
    fig.add_artist(plt.Line2D([X_SWATCH, X_RISE], [y + 0.038] * 2,
                              color="0.75", lw=1.0))

    for m, v, sp in rows:
        st = STYLE.get(m, {})
        fig.add_artist(plt.Line2D([X_SWATCH, X_SWATCH + 0.042], [y + 0.011] * 2,
                                  color=st.get("color", "0.4"),
                                  lw=st.get("lw", 2.0),
                                  ls=st.get("ls", "-")))
        fig.text(X_NAME, y, m, fontsize=9.5, ha="left", va="baseline")
        fig.text(X_ASYM, y, f"{v:.1f}", fontsize=10, ha="right", va="baseline")
        fig.text(X_RISE, y, f"{sp:+.1f}", fontsize=10, ha="right", va="baseline",
                 # the rise IS the result; colour it where it is not ~zero
                 color="#c0392b" if abs(sp) > 5 else "0.25",
                 fontweight="bold" if abs(sp) > 5 else "normal")
        y -= DY

    # the references get rows too, so the dashed line needs no on-axes label
    y -= 0.018
    fig.add_artist(plt.Line2D([X_SWATCH, X_SWATCH + 0.042], [y + 0.011] * 2,
                              color="0.10", lw=2.0, dashes=(5, 3)))
    fig.text(X_NAME, y, "DFT-UKS", fontsize=9.5, ha="left", va="baseline")
    fig.text(X_ASYM, y, f"{UKS_TRUTH:.1f}", fontsize=10, ha="right",
             va="baseline")
    y -= DY
    if show_rks:
        fig.add_artist(plt.Line2D([X_SWATCH, X_SWATCH + 0.042], [y + 0.011] * 2,
                                  color="0.75", ls=":", lw=1.2))
        fig.text(X_NAME, y, "DFT-RKS", fontsize=9.5, ha="left", va="baseline")
        fig.text(X_ASYM, y, f"{RKS_REF:.1f}", fontsize=10, ha="right",
                 va="baseline")
        y -= DY

    # >>> THE GREY EXPLANATORY BLOCK MOVED TO THE CAPTION.
    # It said what the shading means and over what interval the rise is
    # measured. Both belong in the caption, where they can be edited without
    # rerunning this script and typeset like the rest of the paper. The numbers
    # are printed to stdout instead so the caption can be written from them.
    if a.annotate:
        fig.text(X_SWATCH, y, "shaded: beyond the 6.0 Å cutoff\n"
                              "rise measured 4.5 → 7.0 Å",
                 fontsize=9, color="0.45", ha="left", va="top")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=300)   # margins chosen above
    print(f"  wrote {a.out}   ({len(want)} models: {', '.join(want)})")
    for m, v, s in rows:
        print(f"    {m:<20s} E(7.0) = {v:+7.1f}   rise 4.5->7.0 = {s:+6.1f}")


if __name__ == "__main__":
    main()
