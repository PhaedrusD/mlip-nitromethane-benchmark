#!/usr/bin/env python
"""
plot_short_range_floor_hh.py -- the H...H compression test
==========================================================

Companion to plot_short_range_floor.py (Figure 5, O-H). Same frame, same
protocol, but the closest H...H contact is compressed (1.02 -> 0.05 A).
Reads MLIP_paper/short_range_floor_hh/floor_*.csv written by
    short_range_floor.py --pair HH --model <m>

  (a) linear, 1.02 to 0.38 A: the window thermal motion reaches
      (closest H-H approach in any 3000 K run, 0.56-0.59 A, shaded).
  (b) log scale of |dE|, 0.70 to 0.05 A. A dashed segment marks where dE is
      NEGATIVE (the configuration is lower in energy than the starting frame).

Also writes MLIP_paper/short_range_floor_hh_table.{csv,md}.
"""
import argparse, csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_short_range_floor as base      # applies poster_style if present
from plot_short_range_floor import LABEL, COLOR, ORDER, THERMAL_FLOOR, KT_3000, load, _panel_letter


def turnover(r, e):
    """First local maximum going inward (r is ascending): (r_max, e_max), or
    None if the curve rises monotonically to the innermost point."""
    rr, ee = r[::-1], e[::-1]                # inward
    for i in range(1, len(ee) - 1):
        if ee[i] > ee[i - 1] and ee[i] > ee[i + 1]:
            return float(rr[i]), float(ee[i])
    return None


def table(D, out_csv, out_md):
    pts = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.10, 0.05]
    rows = []
    for m in ORDER:
        r, e, _ = D[m]
        row = {"model": LABEL[m]}
        for p in pts:
            row[f"dE_{p:.2f}A"] = float(np.interp(p, r, e))
        t = turnover(r, e)
        row["first_max_r_A"] = t[0] if t else float("nan")
        row["first_max_kcal"] = t[1] if t else float("nan")
        row["min_kcal"] = float(e.min()); row["min_r_A"] = float(r[int(np.argmin(e))])
        rows.append(row)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def fmt(v):
        if v != v:
            return "none"
        s = f"{v:.1e}" if abs(v) >= 1e5 else f"{v:,.0f}"
        return s.replace("-", "−")
    hdr = ["Model"] + [f"{p:.2f} Å" for p in pts] + ["First maximum"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for row in rows:
        cells = [row["model"]] + [fmt(row[f"dE_{p:.2f}A"]) for p in pts]
        cells.append("none" if row["first_max_kcal"] != row["first_max_kcal"]
                     else f"{row['first_max_kcal']:.0f} at {row['first_max_r_A']:.2f} Å")
        lines.append("| " + " | ".join(cells) + " |")
    md = "\n".join(lines) + "\n"
    open(out_md, "w").write(md)
    print(md)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", default="MLIP_paper/short_range_floor_hh")
    ap.add_argument("--out", default="MLIP_paper/fig_short_range_floor_hh.png")
    a = ap.parse_args()

    D = load(a.indir)
    table(D, "MLIP_paper/short_range_floor_hh_table.csv",
          "MLIP_paper/short_range_floor_hh_table.md")
    r0 = max(v[2] for v in D.values())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.6))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.14, top=0.92, wspace=0.24)

    for ax in (ax1, ax2):
        ax.axvspan(*THERMAL_FLOOR, color="0.85", alpha=0.7, zorder=0)
        ax.grid(alpha=.2)
        ax.set_xlabel("H–H separation (Å)")

    # (a) the thermal window, linear
    for m in ORDER:
        r, e, _ = D[m]
        ax1.plot(r, e, color=COLOR[m], lw=2.4 if m in ("off23", "osaka24") else 2.0,
                 label=LABEL[m], zorder=3)
    ax1.set_xlim(r0 + 0.01, 0.38)
    ax1.set_ylim(-80, 280)
    ax1.axhline(0, color="0.3", lw=1.0, zorder=2)
    ax1.axhline(KT_3000, color="0.4", ls=":", lw=1.2, zorder=2)
    ax1.set_ylabel("ΔE (kcal mol$^{-1}$)")
    ax1.legend(fontsize=11, loc="upper left", framealpha=.95)
    _panel_letter(ax1, "a")

    # (b) the whole cut: |dE| on a log axis, negative stretches dashed
    for m in ORDER:
        r, e, _ = D[m]
        lw = 2.4 if m in ("off23", "osaka24") else 2.0
        pos = np.where(e > 0, e, np.nan)
        neg = np.where(e < -5.0, -e, np.nan)   # ignore sub-kcal float32 noise at the start
        ax2.plot(r, pos, color=COLOR[m], lw=lw, zorder=3)
        if np.isfinite(neg).any():
            ax2.plot(r, neg, color=COLOR[m], lw=lw, ls="--", zorder=3,
                     label=f"{LABEL[m]}, ΔE < 0 (dashed)")
    ax2.set_xlim(0.70, 0.0)
    ax2.set_yscale("log")
    ax2.set_ylim(20, 1e9)
    ax2.set_ylabel("|ΔE| (kcal mol$^{-1}$, log scale)")
    ax2.legend(fontsize=11, loc="upper left", framealpha=.95)
    _panel_letter(ax2, "b")

    fig.savefig(a.out, dpi=300)
    print(f"  wrote {a.out}")
    for m in ORDER:
        r, e, _ = D[m]
        t = turnover(r, e)
        print(f"  {LABEL[m]:<13} first max {t}   min {e.min():.1f} at {r[int(np.argmin(e))]:.3f} A"
              f"   crosses zero inward at "
              f"{[round(float(r[i]),3) for i in range(len(r)-1) if e[i] < 0 <= e[i+1]]}")


if __name__ == "__main__":
    main()
