#!/usr/bin/env python
"""
nve_timestep_test.py -- is the -9.9 meV/atom jump a PES step or integrator error?
=================================================================================

THE ONE QUESTION
    The UMA NVE run shows a discrete -9.9 meV/atom drop in total energy at
    9.40 ps, 11x the 0.90 meV/atom noise floor, with negligible drift on either
    side (+0.09 and +0.03 meV/atom/ps). It is real. What it MEANS is unsettled,
    and the slide currently implies the wrong thing.

    Two candidates, and they respond differently to the timestep:

        INTEGRATOR ERROR   velocity-Verlet energy error scales as dt^2. Halve
                           dt, the step falls to a quarter. This is what a fast
                           rearrangement (H transfer, bond snapping) does to a
                           0.5 fs timestep at 2300 K.

        PES DISCONTINUITY  a genuine step in the potential is a property of the
                           model, not the integrator. It does NOT shrink with
                           dt. If this is the answer, the next question is where
                           it comes from -- and 5.5-5.6 A, where the fragments
                           sit at 9.40 ps, is close to the 5.76-5.89 A seam
                           measured in the rigid scans.

    Nothing else distinguishes them from a single trajectory, which is why the
    figure cannot be captioned until this runs.

>>> WHY THE RESTART IS EXACT
    uma_nve_traj.xyz stores momenta as well as positions, so we resume from the
    real dynamical state -- not from positions plus a fresh Maxwell-Boltzmann
    draw, which would be a different trajectory and would prove nothing.

>>> THE HONEST CAVEAT, STATED UP FRONT
    MD is chaotic. Different timesteps visit slightly different geometries and
    the trajectories separate exponentially. Over the 0.4 ps window here, with
    7 atoms, they should stay close enough to compare -- but if they visibly
    diverge, the question to ask is whether the step EXISTS at each dt, not
    whether its size matches to two decimals. The script prints the trajectory
    separation so you can judge rather than assume.

USAGE (cluster, uma env, via sbatch -- never the login node)
    conda activate uma
    python nve_timestep_test.py --traj uma_nve_traj.xyz --restart-ps 9.20 \\
        --window-ps 0.40 --timesteps 0.5,0.25,0.1,0.05
"""

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", default="uma_nve_traj.xyz")
    ap.add_argument("--csv", default="nve_log.csv",
                    help="used only to map frame index -> time")
    ap.add_argument("--restart-ps", type=float, default=9.20)
    ap.add_argument("--window-ps", type=float, default=0.40)
    ap.add_argument("--timesteps", default="0.5,0.25,0.1,0.05",
                    help="fs, comma separated. The first should be the ORIGINAL "
                         "timestep so it reproduces the known result.")
    ap.add_argument("--log-fs", type=float, default=1.0,
                    help="log at fixed TIME intervals so every dt is compared "
                         "on the same grid")
    ap.add_argument("--uma-ckpt", default="uma-s-1p1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="MLIP_paper/nve_timestep_test.csv")
    a = ap.parse_args()

    from ase.io import read
    from ase.md.verlet import VelocityVerlet
    from ase import units
    from fairchem.core import pretrained_mlip, FAIRChemCalculator

    d = np.genfromtxt(a.csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    ph = np.array([str(x) for x in d["phase"]])
    t_csv = np.asarray(d["time_ps"], float)[ph == "prod"]

    frames = read(a.traj, index=":")
    off = len(frames) - len(t_csv)
    k = int(np.argmin(np.abs(t_csv - a.restart_ps)))
    src = frames[off + k]
    if src.get_momenta() is None or not np.any(src.get_momenta()):
        raise SystemExit("  the trajectory carries no momenta -- an exact "
                         "restart is impossible and this test is meaningless")
    print(f"  restarting from frame {off+k} (t = {t_csv[k]:.3f} ps), "
          f"{len(src)} atoms")
    print(f"  window {a.window_ps} ps  ->  covers the 9.40 ps step\n")

    pred = pretrained_mlip.get_predict_unit(a.uma_ckpt, device=a.device)
    nat = len(src)

    rows, results, traces = [], [], {}
    for dt_fs in [float(x) for x in a.timesteps.split(",")]:
        at = src.copy()
        at.info = {"charge": 0, "spin": 1}
        at.calc = FAIRChemCalculator(pred, task_name="omol")
        nsteps = int(round(a.window_ps * 1000.0 / dt_fs))
        every = max(1, int(round(a.log_fs / dt_fs)))
        log = []

        def rec(at=at, log=log):
            log.append((at.get_potential_energy() + at.get_kinetic_energy(),
                        at.get_positions().copy()))

        dyn = VelocityVerlet(at, timestep=dt_fs * units.fs)
        dyn.attach(rec, interval=every)
        rec()
        dyn.run(nsteps)

        E = np.array([x[0] for x in log])
        pos = np.array([x[1] for x in log])
        e = (E - E[0]) * 1000.0 / nat                 # meV/atom
        tt = a.restart_ps + np.arange(len(e)) * every * dt_fs / 1000.0
        j = np.diff(e)
        i = int(np.argmax(np.abs(j)))
        step, t_step = float(j[i]), float(tt[i + 1])
        # drift measured AWAY from the step, so the step does not pollute it
        pre = e[:max(2, i - 2)]
        post = e[min(len(e) - 2, i + 3):]
        dpre = float(np.polyfit(tt[:len(pre)], pre, 1)[0]) if len(pre) > 3 else np.nan
        dpost = float(np.polyfit(tt[-len(post):], post, 1)[0]) if len(post) > 3 else np.nan
        noise = float(np.std(np.diff(e[:max(3, i - 2)]))) if i > 5 else np.nan

        results.append((dt_fs, step, t_step, dpre, dpost, noise))
        traces[dt_fs] = (tt, e, pos)
        rows.append(f"{dt_fs},{step:.4f},{t_step:.4f},{dpre:.4f},{dpost:.4f},"
                    f"{noise:.4f},{len(e)}")
        print(f"  dt = {dt_fs:5.3f} fs   largest step {step:+8.3f} meV/atom "
              f"at {t_step:.3f} ps   drift {dpre:+.3f} / {dpost:+.3f} "
              f"meV/atom/ps   noise {noise:.3f}")

    # ---- did the trajectories stay together? -------------------------------
    print(f"\n  trajectory separation from the dt={results[0][0]} fs run "
          f"(max atom displacement at the end of the window):")
    t0, e0, p0 = traces[results[0][0]]
    for dt_fs, *_ in results[1:]:
        tt, e, pos = traces[dt_fs]
        n = min(len(p0), len(pos))
        sep = float(np.max(np.linalg.norm(p0[n - 1] - pos[n - 1], axis=1)))
        print(f"    dt {dt_fs:5.3f} fs   {sep:6.3f} A"
              + ("   <-- diverged; compare step PRESENCE, not size"
                 if sep > 0.5 else ""))

    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(
        "dt_fs,step_meV_atom,t_step_ps,drift_pre,drift_post,noise,n_points\n"
        + "\n".join(rows) + "\n")

    # ---- the verdict --------------------------------------------------------
    print(f"\n{'='*76}\nVERDICT\n{'='*76}")
    dt0, s0 = results[0][0], abs(results[0][1])
    print(f"\n  {'dt/fs':>7}{'|step|':>10}{'(dt/dt0)^2 x step0':>22}{'ratio':>9}")
    for dt_fs, step, *_ in results:
        pred_dt2 = s0 * (dt_fs / dt0) ** 2
        print(f"  {dt_fs:>7.3f}{abs(step):>10.3f}{pred_dt2:>22.3f}"
              f"{abs(step)/pred_dt2 if pred_dt2 > 1e-9 else np.inf:>9.2f}")

    smallest = abs(results[-1][1])
    frac = smallest / s0 if s0 > 1e-9 else np.nan
    dt_ratio2 = (results[-1][0] / dt0) ** 2
    print()
    if frac < 3 * dt_ratio2:
        print(f"  INTEGRATOR ERROR. The step falls roughly as dt^2 "
              f"({frac:.3f} of the\n  original at dt^2 ratio {dt_ratio2:.3f}). "
              f"The jump is a property of the\n  0.5 fs timestep during a fast "
              f"rearrangement, NOT of the potential.\n  Caption the slide as a "
              f"timestep result and drop the cutoff line.")
    elif frac > 0.5:
        print(f"  PES DISCONTINUITY. The step survives at {results[-1][0]} fs "
              f"({frac:.2f} of its\n  original size) when dt^2 alone predicts "
              f"{dt_ratio2:.3f}. This is a real feature\n  of the model. Next: "
              f"rigid-scan the CH3O + NO geometry from this frame\n  and see "
              f"whether the seam sits at the same separation as the CH3 + NO2\n"
              f"  scan (5.76-5.89 A).")
    else:
        print(f"  MIXED. The step shrinks but more slowly than dt^2 "
              f"({frac:.3f} vs {dt_ratio2:.3f}).\n  Most likely a real but "
              f"STEEP feature that the integrator resolves better at\n  small "
              f"dt. Report both columns; do not pick a label.")
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
