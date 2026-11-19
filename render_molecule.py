#!/usr/bin/env python
"""
render_molecule.py -- a shaded ball-and-stick PNG with a transparent background
================================================================================

No POV-Ray needed. Spheres are drawn as radial-gradient discs with a specular
highlight, painter-sorted by depth, bonds as rounded capsules behind them.
Default is the nitromethane geometry the paper uses (orca_dlpno/CH3NO2_opt.xyz),
rotated so the C-N axis runs left to right and the nitro group is edge-on.

USAGE
    python render_molecule.py                                  # -> MLIP_paper/nitromethane_render.png
    python render_molecule.py --xyz some.xyz --out x.png --rot 20,-30,0
"""
import argparse, os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import to_rgb

# CPK-like, slightly desaturated so it sits well on a dark slide
COLOR = {"C": "#4A4F58", "N": "#2F5BD8", "O": "#D8402A", "H": "#E6E8EC"}
RADIUS = {"C": 0.46, "N": 0.44, "O": 0.42, "H": 0.28}      # drawn radius, A (ball and stick)
BOND_CUT = {("C", "H"): 1.25, ("C", "N"): 1.65, ("N", "O"): 1.45}


def rot(v, ax, deg):
    t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    R = {"x": [[1, 0, 0], [0, c, -s], [0, s, c]],
         "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
         "z": [[c, -s, 0], [s, c, 0], [0, 0, 1]]}[ax]
    return v @ np.array(R).T


def shaded_sphere(ax, x, y, r, color, z, light=(-0.45, 0.6)):
    base = np.array(to_rgb(color))
    n = 28
    for k in range(n):
        f = 1 - k / n                       # 1 at rim .. small at the core
        rr = r * f
        cx, cy = x + light[0] * r * (1 - f) * 0.55, y + light[1] * r * (1 - f) * 0.55
        shade = base * (0.45 + 0.75 * (1 - f) ** 0.9)
        shade = np.clip(shade + 0.25 * (1 - f) ** 3, 0, 1)
        ax.add_patch(Circle((cx, cy), rr, color=shade, zorder=z, lw=0))
    # specular
    ax.add_patch(Circle((x + light[0] * r * 0.45, y + light[1] * r * 0.45), r * 0.18,
                        color=np.clip(base * 0.3 + 0.75, 0, 1), alpha=0.85, zorder=z + 0.01, lw=0))
    ax.add_patch(Circle((x, y), r, fill=False, ec=base * 0.55, lw=1.2, zorder=z + 0.02))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz", default="orca_dlpno/CH3NO2_opt.xyz")
    ap.add_argument("--out", default="MLIP_paper/nitromethane_render.png")
    ap.add_argument("--rot", default="z:-90,x:50,y:-30", help="sequence of axis:deg, applied in order")
    ap.add_argument("--size", type=float, default=6.0, help="figure size, inches (square)")
    a = ap.parse_args()

    lines = open(a.xyz).read().splitlines()[2:]
    sym = [l.split()[0] for l in lines if l.strip()]
    pos = np.array([[float(v) for v in l.split()[1:4]] for l in lines if l.strip()])
    pos -= pos.mean(axis=0)
    for step in a.rot.split(","):
        axn, d = step.split(":")
        pos = rot(pos, axn, float(d))

    # bonds
    bonds = []
    for i in range(len(sym)):
        for j in range(i + 1, len(sym)):
            key = tuple(sorted((sym[i], sym[j])))
            cut = BOND_CUT.get(key)
            if cut and np.linalg.norm(pos[i] - pos[j]) < cut:
                bonds.append((i, j))

    fig = plt.figure(figsize=(a.size, a.size), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_aspect("equal"); ax.axis("off")
    fig.patch.set_alpha(0); ax.patch.set_alpha(0)

    # perspective-free orthographic view down z; depth = z for painter's order
    order = np.argsort(pos[:, 2])
    zrank = {int(i): k for k, i in enumerate(order)}
    for i, j in bonds:
        p, q = pos[i], pos[j]
        zb = (p[2] + q[2]) / 2
        z = 10 + 10 * (zb - pos[:, 2].min()) / (np.ptp(pos[:, 2]) + 1e-9)
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#8E94A3", lw=11, solid_capstyle="round", zorder=5)
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#C9CED8", lw=7, solid_capstyle="round", zorder=5.1)
    for i in order:
        z = 10 + 10 * (pos[i, 2] - pos[:, 2].min()) / (np.ptp(pos[:, 2]) + 1e-9)
        shaded_sphere(ax, pos[i, 0], pos[i, 1], RADIUS[sym[i]], COLOR[sym[i]], z)

    lim = np.abs(pos[:, :2]).max() + 0.7
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=300, transparent=True)
    print(f"  wrote {a.out}  ({len(sym)} atoms, {len(bonds)} bonds)")


if __name__ == "__main__":
    main()
