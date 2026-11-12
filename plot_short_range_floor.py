#!/usr/bin/env python
"""
plot_short_range_floor.py -- the O...H compression test, publication form
=========================================================================

Reads short_range_floor/floor_*.csv (from short_range_floor.py, one model per
process) and draws two panels on the same five curves:

  (a) linear, 0.45 to 0.83 A: the window thermal motion reaches. All five
      models agree to within ~25% down to 0.65 A; MACE-OFF23 leaves the group
      there, at the thermal floor (closest approach in any 3000 K run,
      0.56-0.59 A, shaded).
  (b) log scale, 0.70 to 0.05 A (all values positive there): ANI, OMOL and UMA stay bounded
      (ANI and OMOL turn over and DECLINE); MACE-Osaka24 leaves the group near
      0.32 A and both MACE models rise to 10^8-10^10 kcal/mol.

Also writes short_range_floor_table.csv / .md: dE at fixed separations, the
peak of each bounded wall, and where each cliff begins (first point at which
the model exceeds three times the median of the bounded group).

No on-figure titles or notes; the caption carries them. --annotate restores
them for slides.

USAGE
    python plot_short_range_floor.py
"""
import argparse, csv, glob, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
try:
    import poster_style; poster_style.apply()
except ModuleNotFoundError:
    pass

LABEL = {"off23": "MACE-OFF23", "osaka24": "MACE-Osaka24", "omol": "MACE-OMOL-0",
         "uma": "UMA-S-1P1", "ani": "ANI-1xnr"}
COLOR = {"ani": "#b03030", "off23": "#e8860c", "osaka24": "#2e8b30",
         "omol": "#1f4fd8", "uma": "#8e6bbf"}
ORDER = ["ani", "omol", "uma", "osaka24", "off23"]
BOUNDED = ["ani", "omol", "uma"]
THERMAL_FLOOR = (0.56, 0.59)      # closest approach reached in any 3000 K run
KT_3000 = 5.96                    # kcal/mol


def _panel_letter(ax, letter):
    ax.text(0.0, 1.015, f"({letter})", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=13, fontweight="bold")


def load(indir):
    D = {}
    for f in glob.glob(os.path.join(indir, "floor_*.csv")):
        rows = list(csv.DictReader(open(f)))
        m = rows[0]["model"]
        r = np.array([float(x["r_A"]) for x in rows])
        e = np.array([float(x["dE_kcal"]) for x in rows])
        o = np.argsort(r)
        D[m] = (r[o], e[o], float(rows[0]["r0"]))
    missing = [m for m in ORDER if m not in D]
    if missing:
        sys.exit(f"  missing: {missing}")
    return D


def cliff_onset(D, m):
    """Largest r (going inward) at which the energy more than doubles over a
    single 0.02 A step. A bounded wall never does; NaN then."""
    r, e, _ = D[m]
    # arrays are sorted ascending in r; step inward = index i+1 -> i
    ratio = e[:-1] / np.maximum(e[1:], 1e-9)
    bad = np.where((ratio > 2.0) & (e[1:] > 50))[0]
    return float(r[bad.max() + 1]) if len(bad) else float("nan")


def table(D, out_csv, out_md):
    pts = [0.80, 0.70, 0.60, 0.50, 0.30, 0.10]
    rows = []
    for m in ORDER:
        r, e, _ = D[m]
        row = {"model": LABEL[m]}
        for p in pts:
            row[f"dE_{p:.2f}A"] = float(np.interp(p, r, e))
        if m in BOUNDED:
            k = int(np.argmax(e))
            row["peak_kcal"] = float(e[k]); row["peak_r_A"] = float(r[k])
            row["cliff_r_A"] = float("nan")
        else:
            row["peak_kcal"] = float("nan"); row["peak_r_A"] = float("nan")
            row["cliff_r_A"] = cliff_onset(D, m)
        row["dE_0.05A"] = float(e[0])
        rows.append(row)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def fmt(v, big=False):
        if v != v:
            return "–"
        return f"{v:.1e}" if abs(v) >= 1e5 else f"{v:,.0f}"
    hdr = ["model"] + [f"{p:.2f} Å" for p in pts] + ["0.05 Å", "wall peak", "cliff onset"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for row in rows:
        cells = [row["model"]] + [fmt(row[f"dE_{p:.2f}A"]) for p in pts] + [fmt(row["dE_0.05A"])]
        cells.append("–" if row["peak_kcal"] != row["peak_kcal"]
                     else f"{row['peak_kcal']:.0f} at {row['peak_r_A']:.2f} Å")
        cells.append("–" if row["cliff_r_A"] != row["cliff_r_A"] else f"{row['cliff_r_A']:.2f} Å")
        lines.append("| " + " | ".join(cells) + " |")
    md = ("Energy added by compressing one O···H contact in a 3000 K liquid frame "
          "from 0.834 Å, kcal/mol relative to the unperturbed frame, bare model, "
          "all other atoms fixed. Wall peak: maximum for the three bounded models. "
          "Cliff onset: first separation at which the energy more than doubles "
          "over a single 0.02 Å step.\n\n" + "\n".join(lines) + "\n")
    open(out_md, "w").write(md)
    print(md)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default="short_range_floor")
    ap.add_argument("--out", default="MLIP_paper/fig_short_range_floor.png")
    ap.add_argument("--annotate", action="store_true")
    a = ap.parse_args()

    D = load(a.indir)
    rows = table(D, "MLIP_paper/short_range_floor_table.csv",
                 "MLIP_paper/short_range_floor_table.md")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.6))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.14, top=0.92, wspace=0.24)

    for ax in (ax1, ax2):
        ax.axvspan(*THERMAL_FLOOR, color="0.85", alpha=0.7, zorder=0)
        for m in ORDER:
            r, e, _ = D[m]
            ax.plot(r, e, color=COLOR[m], lw=2.4 if m in ("off23", "osaka24") else 2.0,
                    label=LABEL[m], zorder=3)
        ax.grid(alpha=.2)
        ax.set_xlabel("O–H separation (Å)")
        ax.invert_xaxis()

    # (a) the thermal window, linear
    ax1.set_xlim(0.84, 0.45)
    ax1.set_ylim(0, 1200)
    ax1.axhline(KT_3000, color="0.4", ls=":", lw=1.2, zorder=2)
    ax1.set_ylabel("ΔE (kcal mol$^{-1}$)")
    ax1.legend(fontsize=11, loc="upper left", framealpha=.95)
    _panel_letter(ax1, "a")
    if a.annotate:
        ax1.text(0.575, 1150, "thermal floor\n0.56–0.59 Å", ha="center", va="top",
                 fontsize=9, color="0.35")
        ax1.text(0.83, KT_3000 * 1.6, "kT at 3000 K", fontsize=9, color="0.4")
        ax1.set_title("the window thermal motion reaches", loc="left", fontsize=11)

    # (b) the whole cut, symlog
    # plain log axis: every value from 0.70 A inward is positive (the smallest
    # is ANI's 77 kcal/mol near 0.27 A), so no symlog and no linthresh to explain
    ax2.set_xlim(0.70, 0.0)
    ax2.set_yscale("log")
    ax2.set_ylim(30, 3e10)
    ax2.set_ylabel("ΔE (kcal mol$^{-1}$, log scale)")
    _panel_letter(ax2, "b")
    if a.annotate:
        for m in ("off23", "osaka24"):
            c = cliff_onset(D, m)
            ax2.axvline(c, color=COLOR[m], ls="--", lw=1.0)
            ax2.text(c, 2e9, f"{c:.2f} Å", color=COLOR[m], ha="right", fontsize=9)
        ax2.set_title("the full cut to 0.05 Å", loc="left", fontsize=11)
        fig.suptitle("Compressing one O···H contact: who has a floor", fontsize=13,
                     fontweight="bold")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=300)
    print(f"  wrote {a.out}")
    for m in ("off23", "osaka24"):
        print(f"  {LABEL[m]} cliff onset: {cliff_onset(D, m):.3f} Å")


if __name__ == "__main__":
    main()
