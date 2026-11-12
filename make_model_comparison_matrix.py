#!/usr/bin/env python
"""
make_model_comparison_matrix.py -- all five models on ONE axis, not five grids
===============================================================================

>>> THE PROBLEM WITH FIVE BUBBLE MATRICES SIDE BY SIDE
    Each bubble matrix is a 7 x 6 grid with 17 cells filled, so 60% of every
    panel is blank. Five of them is 210 empty cells of paper. That is the
    cosmetic complaint; the real one is that the figure's CLAIM is comparative
    -- "each model fails in a different pattern" -- and a reader can only test
    that claim by looking at one panel, remembering a bubble AREA, moving to
    the next panel, and comparing from memory.

    People compare POSITION ALONG A COMMON SCALE far better than they compare
    area, and far better than they compare anything across a saccade. So the
    comparison should happen inside one set of axes, with the models adjacent.

    A second problem is specific to the bubble encoding: size and colour carry
    the SAME variable, and size is clipped at 45 kcal/mol by
    `np.minimum(np.abs(values), 45)`. MACE-OMOL's +56.2 and ANI-1xnr's -60.5
    therefore draw at exactly the same area as a 45. Nothing on the figure says
    so. Neither layout here clips anything.

>>> TWO LAYOUTS, SAME NUMBERS

    --layout dots   (default, recommended)
        One row per cell, error on the x-axis, five coloured dots per row,
        zero marked. Reads as: how far from the reference, in which direction,
        for each model, on a single shared ruler. Systematic bias shows as a
        colour sitting consistently to one side; an isolated failure shows as
        one dot thrown off a tight cluster. Both are the paper's argument.

    --layout grid
        17 rows x 5 columns, colour = signed error, number in every cell. This
        is the bubble matrix folded so that the comparison runs left to right
        along a row. Use it if the section needs something that reads as a
        table. It is also the honest answer to "should it be five figures or a
        table?" -- it is both.

>>> ROW ORDER IS AN ARGUMENT
    --order spread (default) puts the cells where the models DISAGREE most at
    the top and the cells where they agree at the bottom. The eye then starts
    where the information is, and the flat bottom of the figure is itself a
    result: on ordinary C-H bonds the five models are interchangeable, and
    everything that separates them is concentrated in nitro, peroxide and
    C-C chemistry. --order bond groups by bond type instead, which is the
    right choice if the surrounding text walks through bonds in order.

USAGE
    python make_model_comparison_matrix.py
    python make_model_comparison_matrix.py --layout grid
    python make_model_comparison_matrix.py --order bond --out fig.png
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import poster_style
poster_style.apply()

from bde_matrix_data import MODEL_ORDER, COLORS, cells


def order_rows(rows, how):
    if how == "spread":
        # descending disagreement; ties broken by the largest single error so
        # a big shared failure outranks a small one
        return sorted(rows, key=lambda r: (r["spread"], r["maxabs"]),
                      reverse=True)
    from bde_matrix_data import BONDS
    return sorted(rows, key=lambda r: BONDS.index(r["bond"]))


def draw_dots(rows, models, a):
    n = len(rows)
    # >>> THE MARGINS ARE THE FIGURE.
    # Row labels are two words of body type and the key is five model names
    # plus a number, so both need real room reserved BEFORE anything is drawn.
    # At left=0.255 the first character of every label printed off the canvas,
    # and bbox_inches="tight" cannot recover text that was clipped by the axes.
    fig, ax = plt.subplots(figsize=poster_style.figsize(11.2, 0.42 * n + 2.2))
    fig.subplots_adjust(left=0.300, right=0.735, bottom=0.095, top=0.925)

    ylab = []
    for k, r in enumerate(rows):
        y = n - 1 - k
        ylab.append(r["label"])
        # a guide line from zero to the furthest model, so the row reads as a
        # deviation rather than as five unrelated points
        v = r["values"]
        ax.plot([min(0, v.min()), max(0, v.max())], [y, y],
                color="0.88", lw=1.4, zorder=1, solid_capstyle="round")
        for m, val in zip(models, v):
            ax.plot(val, y, "o", ms=8, color=COLORS.get(m, "0.4"),
                    mec="white", mew=1.0, zorder=3)

    ax.axvline(0, color="0.25", lw=1.4, zorder=2)
    ax.set_yticks(range(n))
    ax.set_yticklabels(ylab[::-1], fontsize=9.5)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("BDE error / kcal mol$^{-1}$   (positive = bond too strong)")
    ax.grid(axis="x", color="lightgrey", ls="-", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # >>> A SHADED "USABLE" BAND, NOT A DECORATION.
    # Chemical accuracy is 1 kcal/mol and no MLIP here is close to it on bond
    # breaking, so a 1 kcal band would be a joke. +/-10 is the band inside
    # which a model's error is small next to the 60 kcal/mol spread the figure
    # is about; state the number in the caption rather than implying a standard.
    ax.axvspan(-a.band, a.band, color="0.945", zorder=0)

    # >>> KEY AND SUMMARY ARE ONE BLOCK, NOT TWO.
    # A separate mean-|error| block repeats all five model names, and at poster
    # type the repeated names collided with their own right-aligned numbers
    # ("MACE-Osaka2414.7"). Hanging the number off the key line instead removes
    # the second name column entirely and makes the block half as tall.
    mae = [float(np.mean([abs(r["values"][i]) for r in rows]))
           for i in range(len(models))]
    X_DOT, X_NAME, X_NUM = 0.755, 0.775, 0.995
    y0 = 0.845
    DY = 0.036

    fig.text(X_NAME, y0 + 0.042, "model", fontsize=9.5, fontweight="bold",
             color="0.25", va="baseline")
    fig.text(X_NUM, y0 + 0.042, "mean |err|", fontsize=9.5, fontweight="bold",
             color="0.25", ha="right", va="baseline")
    fig.add_artist(plt.Line2D([X_DOT, X_NUM], [y0 + 0.030] * 2, color="0.75",
                              lw=1.0))
    for i, m in enumerate(models):
        yy = y0 - i * DY
        fig.add_artist(plt.Line2D([X_DOT], [yy + 0.008], marker="o", ms=8,
                                  color=COLORS.get(m, "0.4"), mec="white",
                                  mew=1.0))
        fig.text(X_NAME, yy, m, fontsize=9.5, va="baseline")
        fig.text(X_NUM, yy, f"{mae[i]:.1f}", fontsize=9.5, ha="right",
                 va="baseline",
                 fontweight="bold" if mae[i] == min(mae) else "normal")

    # >>> KEEP EVERY LINE SHORT ENOUGH TO FIT THE 0.245-WIDE COLUMN.
    # figure text does not wrap and does not error when it runs past the
    # canvas; it is simply cut off at the edge, and "rows sorted by
    # across-mode" is how that looks in a printed draft.
    fig.text(X_DOT, y0 - len(models) * DY - 0.020,
             f"shaded band\n±{a.band:.0f} kcal mol⁻¹\n\n"
             f"rows sorted by\nacross-model spread",
             fontsize=9, color="0.45", va="top", linespacing=1.35)

    # left-aligned over the axes, not centred: a centred title on a figure with
    # a key at the right runs into the key.
    ax.set_title(a.title, fontsize=11, loc="left")
    return fig


def draw_grid(rows, models, a):
    n = len(rows)
    M = np.array([r["values"] for r in rows])
    vlim = float(np.ceil(np.abs(M).max() / 5.0) * 5.0)   # nothing is clipped

    # Rotated column labels hang BELOW and LEFT of their tick, so the bottom
    # margin has to hold a rotated "MACE-Osaka24" at poster type, not a line of
    # text. At 0.075 the first two names were cut off by the canvas edge.
    fig, ax = plt.subplots(figsize=poster_style.figsize(9.4, 0.40 * n + 3.0))
    fig.subplots_adjust(left=0.310, right=0.850, bottom=0.135,
                        top=0.930 if a.title else 0.985)

    im = ax.imshow(M, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")

    for i in range(n):
        for j in range(len(models)):
            v = M[i, j]
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=9.5,
                    color="white" if abs(v) > 0.62 * vlim else "0.1")

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9.5, rotation=30, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=9.5)
    ax.set_xticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=0)

    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("BDE error / kcal mol$^{-1}$", size=10.5, weight="bold",
                 rotation=270, labelpad=26)
    cb.ax.tick_params(labelsize=9)

    ax.set_title(a.title, fontsize=11, loc="left")
    return fig


def draw_grid_wide(rows, models, a):
    """The grid turned 90 degrees for a 16:9 slide: models down the side,
    the 17 bond/environment cells across the top. Same numbers, same colour
    scale, slide-sized type. Written to a separate file; the paper figure is
    untouched. poster_style is reset here because its 1.75x type scaling is
    for the paper figures and makes 17 columns of labels collide."""
    # poster_style patches Text.set_size to multiply every size by SCALE,
    # and rcdefaults() does not undo a monkeypatch. Divide it back out so the
    # sizes below are the sizes that land on the slide.
    matplotlib.rcdefaults()
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["savefig.dpi"] = 300
    f = a.font / poster_style.SCALE
    n = len(rows)
    M = np.array([r["values"] for r in rows]).T          # (models, cells)
    vlim = float(np.ceil(np.abs(M).max() / 5.0) * 5.0)

    SHORT = {"Nitro Compounds": "nitro", "Alkanes": "alkane",
             "Alcohols": "alcohol", "Amines": "amine",
             "Peroxides": "peroxide", "Carbonyls": "carbonyl"}

    fig, ax = plt.subplots(figsize=(13.33, 5.8))
    fig.subplots_adjust(left=0.150, right=0.895, bottom=0.235, top=0.975)

    im = ax.imshow(M, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
    for i in range(len(models)):
        for j in range(n):
            v = M[i, j]
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                    fontsize=f, fontweight="bold",
                    color="white" if abs(v) > 0.62 * vlim else "0.1")

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=f + 1, fontweight="bold")
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"{r['bond'].replace('-', chr(8211))}\n{SHORT[r['env']]}"
                        for r in rows],
                       fontsize=f, rotation=90, va="top", linespacing=1.1)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", lw=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=0, pad=6)
    for sp in ax.spines.values():
        sp.set_visible(False)

    cax = fig.add_axes([0.912, 0.235, 0.018, 0.74])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("BDE error / kcal mol$^{-1}$", size=f, weight="bold",
                 rotation=270, labelpad=18)
    cb.ax.tick_params(labelsize=f - 1)
    if a.title:
        ax.set_title(a.title, fontsize=f + 3, loc="left", pad=10)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", default="dots", choices=["dots", "grid", "grid_wide"])
    ap.add_argument("--order", default="spread", choices=["spread", "bond"])
    ap.add_argument("--models", default=None, help="comma separated subset")
    ap.add_argument("--band", type=float, default=10.0,
                    help="shaded reference band, dots layout only")
    # >>> SHORT ON PURPOSE. poster_style scales type by 1.75 but the canvas by
    # only 1.25, so a title that fits at draft size overruns the canvas at
    # poster size. The full sentence belongs in the caption, where it can wrap.
    ap.add_argument("--title",
                    default="BDE error by bond and environment")
    ap.add_argument("--font", type=float, default=14,
                    help="base font size, grid_wide only (slides: 14-16)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    models = ([m.strip() for m in a.models.split(",")] if a.models
              else list(MODEL_ORDER))
    rows = order_rows(cells(models), a.order)

    fig = {"dots": draw_dots, "grid": draw_grid,
           "grid_wide": draw_grid_wide}[a.layout](rows, models, a)

    out = a.out or os.path.join(_HERE, "thermodynamic_bde_results",
                                f"bde_model_comparison_{a.layout}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"  wrote {out}   ({len(rows)} cells x {len(models)} models)")

    print(f"\n  {'cell':<26}" + "".join(f"{m:>15}" for m in models) + "  spread")
    for r in rows:
        print(f"  {r['label']:<26}"
              + "".join(f"{v:>15.1f}" for v in r["values"])
              + f"{r['spread']:>8.1f}")
    print(f"\n  {'mean |error|':<26}"
          + "".join(f"{np.mean([abs(r['values'][i]) for r in rows]):>15.1f}"
                    for i in range(len(models))))


if __name__ == "__main__":
    main()
