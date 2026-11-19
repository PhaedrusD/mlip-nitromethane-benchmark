#!/usr/bin/env python
"""
plot_toc.py -- Table of Contents / abstract graphic
===================================================

One small image, 3.25 x 1.75 in (8.25 x 4.45 cm), the size ACS specifies for
TOC graphics. Check the current "Guidelines for Table of Contents/Abstract
Graphics" before submitting.

Left:  the nitromethane molecule (MLIP_paper/nitromethane_render.png, made by
       render_molecule.py from coordinates).
Right: the rigid C-N dissociation curve. The broken-symmetry DFT reference
       flattens; the two OMol25-trained models follow it and then rise by
       ~25 kcal/mol as the fragments pass the 6 A interaction cutoff.

Data (all in the repository, nothing is generated here):
    files/dft/dft_full/dft_energies_full.csv      DFT, method uks_bs
    seam_scan/pull_uma.csv, seam_scan/pull_omol.csv   fixed-geometry scans,
        the same coordinates as the DFT scan (SI Section S3, Table S5)

No caption accompanies a TOC graphic, so the few words needed are on the image.

USAGE
    python plot_toc.py                 # -> MLIP_paper/toc_graphic.png (+ .tif)
"""
import argparse, csv, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

KCAL = 23.0605
HARTREE = 627.5095
CUTOFF = 6.0
C_DFT, C_UMA, C_OMOL = "0.10", "#8e6bbf", "#1f4fd8"


def dft(path):
    r, e = [], []
    for row in csv.DictReader(open(path)):
        if row["method"] == "uks_bs":
            r.append(float(row["bond_A"])); e.append(float(row["E_hartree"]))
    r, e = np.array(r), np.array(e)
    o = np.argsort(r)
    return r[o], (e[o] - e.min()) * HARTREE


def model(path):
    rows = [x for x in csv.DictReader(l for l in open(path) if not l.startswith("#"))]
    r = np.array([float(x["bond_A"]) for x in rows])
    e = np.array([float(x["E_eV"]) for x in rows])
    return r, (e - np.nanmin(e)) * KCAL


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="MLIP_paper/toc_graphic.png")
    ap.add_argument("--molecule", default="MLIP_paper/nitromethane_render.png")
    ap.add_argument("--no-molecule", action="store_true")
    a = ap.parse_args()

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 7, "axes.linewidth": 0.8})
    fig = plt.figure(figsize=(3.25, 1.75))

    have_mol = (not a.no_molecule) and os.path.exists(a.molecule)
    if have_mol:
        axm = fig.add_axes([0.00, 0.14, 0.27, 0.74])
        axm.imshow(mpimg.imread(a.molecule)); axm.axis("off")
        fig.text(0.135, 0.07, "CH$_3$–NO$_2$", ha="center", va="center",
                 fontsize=7.5, fontweight="bold")
        ax = fig.add_axes([0.385, 0.215, 0.595, 0.745])
    else:
        ax = fig.add_axes([0.14, 0.215, 0.84, 0.745])

    rd, ed = dft("files/dft/dft_full/dft_energies_full.csv")
    ru, eu = model("seam_scan/pull_uma.csv")
    ro, eo = model("seam_scan/pull_omol.csv")

    ax.axvspan(CUTOFF, 8.0, color="0.86", zorder=0, lw=0)
    ax.axvline(CUTOFF, color="0.45", ls="--", lw=0.8, zorder=1)
    ax.plot(ro, eo, color=C_OMOL, lw=1.6, zorder=3)
    ax.plot(ru, eu, color=C_UMA, lw=1.6, zorder=4)
    ax.plot(rd, ed, color=C_DFT, lw=1.6, zorder=5)

    ax.set_xlim(1.2, 8.0); ax.set_ylim(0, 118)
    ax.set_xticks([2, 4, 6, 8]); ax.set_yticks([0, 50, 100])
    ax.tick_params(length=2.5, width=0.8, pad=2)
    ax.set_xlabel("C–N distance (Å)", labelpad=1.5, fontsize=7.5)
    ax.set_ylabel("energy (kcal mol$^{-1}$)", labelpad=1.5, fontsize=7.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # the few words a caption would otherwise carry
    ax.text(4.2, 68, "DFT reference", color=C_DFT, fontsize=7, ha="center", va="top")
    ax.text(3.3, 97, "OMol25-trained\nMLIPs", color=C_UMA, fontsize=7,
            ha="center", va="center", fontweight="bold", linespacing=0.95)
    ax.text(7.0, 8, "beyond\ncutoff", color="0.35", fontsize=6.5, ha="center",
            va="bottom", linespacing=0.95)
    # size of the rise, as defined in the paper: the model's own energy at
    # 7.0 A minus its energy at 4.5 A (25 kcal/mol). The arrow spans that step,
    # from the model's level before the rise up to its plateau. It is NOT the
    # gap to the DFT plateau, which is 28 kcal/mol.
    x = 7.45
    y0, y1 = float(np.interp(4.5, ru, eu)), float(np.interp(7.0, ru, eu))
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="<->", color="#b03030", lw=1.0,
                                shrinkA=0, shrinkB=0))
    ax.text(x - 0.12, 0.5 * (y0 + y1), f"+{y1 - y0:.0f}", color="#b03030",
            fontsize=7.5, fontweight="bold", ha="right", va="center")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=600)
    fig.savefig(os.path.splitext(a.out)[0] + ".tif", dpi=600)
    print(f"  wrote {a.out} and .tif  (3.25 x 1.75 in at 600 dpi)")
    print(f"  UMA-S-1P1 at 4.5 A {y0:.1f}, at 7.0 A {y1:.1f}, rise {y1 - y0:.1f} kcal/mol")


if __name__ == "__main__":
    main()
