#!/usr/bin/env python
"""
plot_tasks.py
=============

Nitromethane C-N through ALL FIVE of UMA's task heads.

UMA is one set of weights with a mixture-of-linear-experts router selecting a
different expert mix per task. Each task was trained against a DIFFERENT DFT
reference, so absolute energies are not comparable -- every curve here is
referenced to its own minimum.

⚠️ THE CAVEAT THAT MUST TRAVEL WITH THIS FIGURE
-----------------------------------------------
Only `omol` ramps. It is tempting to conclude "the defect is specific to the
OMol25 expert." That is NOT what the data support, because the other four tasks
were ALREADY FLAT by 5.5 A -- there was no interaction energy left for the
envelope to switch off, so the test could not have detected a ramp even if one
existed. They are UNINFORMATIVE NULLS, not clean negatives. Same trap as
CH4-CH4 in the falsification test.

Panel (b) makes that visible instead of leaving it in a footnote.
"""
import argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd

# Publication form by default: no suptitle, no panel titles, no on-figure
# explanatory text. --annotate restores the slide version. The caption carries
# the 23 kcal/mol, the cutoff, and the 2 kcal/mol detectability line.
_ap = argparse.ArgumentParser()
_ap.add_argument("--annotate", action="store_true")
ANN = _ap.parse_args().annotate


def _panel_letter(ax, letter):
    """Bold panel letter just outside the top-left corner, journal style."""
    ax.text(0.0, 1.015, f"({letter})", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=13, fontweight="bold")


EV = 23.0605
TASKS = [("omol", "#D55E00", "ωB97M-V/def2-TZVPD"),
         ("oc20", "#0072B2", "RPBE"),
         ("omat", "#009E73", "PBE/PAW"),
         ("odac", "#CC79A7", "PBE-D3"),
         ("omc",  "#E69F00", "PBE-D3")]

fig, ax = plt.subplots(1, 2, figsize=(13.4, 5.5),
                       gridspec_kw={"width_ratios": [1.35, 1]})
rows = []
for t, c, ref in TASKS:
    d = pd.read_csv(f"seam_scan/pull_umaTASK_{t}.csv", comment="#") \
          .dropna(subset=["E_eV"]).sort_values("bond_A")
    r, E = d.bond_A.values, (d.E_eV.values - d.E_eV.min()) * EV
    lw = 2.9 if t == "omol" else 1.9
    ax[0].plot(r, E, "-", color=c, lw=lw, zorder=4 if t == "omol" else 3,
               label=f"{t}   ({ref})")
    at = lambda q: float(np.interp(q, r, E))
    retained = E[-1] - at(5.5)
    far = d[d.bond_A > 5.0]
    rows.append((t, c, retained, float(far.F_analytic_eV_A.abs().max())))

a = ax[0]
a.axvline(6.0, color="#888", ls="--", lw=1.3)
a.axvspan(6.0, 8.05, color="#888888", alpha=.08)
if ANN:
    a.text(6.08, 6, "r_max = 6.0 Å\n(all five tasks)", fontsize=9, color="#555")
    a.annotate("", xy=(7.5, 103.5), xytext=(7.5, 80.4),
               arrowprops=dict(arrowstyle="<->", color="#D55E00", lw=1.8))
    a.text(7.4, 92, "23 kcal/mol ", ha="right", va="center", fontsize=10,
           fontweight="bold", color="#D55E00")
    a.set_title("(a)  One model, five task heads", loc="left", fontweight="bold")
a.set_xlim(1.2, 8.05); a.set_xlabel("C–N distance (Å)", fontsize=12, fontweight="bold")
a.set_ylabel("energy rel. to own minimum (kcal/mol)", fontsize=12, fontweight="bold")
_panel_letter(a, "a")
a.legend(fontsize=14, loc="lower right", framealpha=.95, handlelength=2.6,
         borderpad=0.7, labelspacing=0.5)
a.grid(alpha=.2)

# ---- (b) does the head read the spin label? --------------------------------
# Same geometries, label flipped singlet -> triplet; mean |dE| over the bonded
# region (r <= 2.0 A), from task_label_test.csv. Replaces the old "interaction
# energy left at 5.5 A" panel, which was circular: for omol that quantity IS
# the rise. This is the quantity the Results quote (78-93 vs < 2 kcal/mol).
import collections
lab = collections.defaultdict(list)
for row in pd.read_csv("MLIP_paper/task_label_test.csv").itertuples():
    lab[row.task].append((row.r_A, row.E_c0s1_eV, row.E_c0s3_eV))
b = ax[1]
x = np.arange(len(TASKS))
vals = []
for t, c, ref in TASKS:
    pts = [(s1, s3) for r, s1, s3 in lab[t] if r <= 2.0]
    vals.append(np.mean([abs(s3 - s1) for s1, s3 in pts]) * EV)
b.bar(x, vals, color=[c for _, c, _ in TASKS], edgecolor="black", lw=.6, width=.62)
for i, v in enumerate(vals):
    b.text(i, v * 1.3, f"{v:.1f}", ha="center", fontweight="bold", fontsize=10)
b.set_yscale("log"); b.set_ylim(0.05, 300)
b.set_xticks(x); b.set_xticklabels([t for t, _, _ in TASKS], fontsize=11, fontweight="bold")
b.set_ylabel("|E(triplet) − E(singlet)| (kcal/mol)", fontsize=12, fontweight="bold")
_panel_letter(b, "b")
if ANN:
    b.set_title("(b)  Only omol reads the label", loc="left", fontweight="bold")
b.grid(alpha=.2, axis="y", which="both")

if ANN:
    fig.suptitle("All five UMA task heads, and why four of them prove nothing",
                 fontsize=15, fontweight="bold")
    fig.text(0.5, 0.015, "oc20/omat/odac/omc were already flat by 5.5 Å, so the "
             "envelope had nothing left to switch off. These are UNINFORMATIVE "
             "NULLS, not clean negatives, the same trap as CH₄–CH₄ in the "
             "falsification test.",
             ha="center", fontsize=8.6, color="#666", style="italic")
    fig.tight_layout(rect=[0, .035, 1, .945])
else:
    fig.tight_layout(rect=[0, 0, 1, 0.97], w_pad=3.0)
fig.savefig("MLIP_paper/fig_uma_tasks.png", dpi=200)
fig.savefig("MLIP_paper/fig_uma_tasks.pdf")
for t, c, ret, f in rows:
    print(f"  {t:<6} retained {abs(ret):6.2f} kcal/mol   peak|F|>5Å {f:.3f} eV/Å")
print("wrote MLIP_paper/fig_uma_tasks.png")
