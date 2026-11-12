#!/usr/bin/env python
"""
zbl_patch.py -- bolt a ZBL screened-Coulomb wall onto any ASE calculator
========================================================================

WHAT THIS IS FOR
----------------
MACE-Osaka24 collapsed an O-H contact in the low-density nitromethane runs
(0.856 -> 0.096 A in a single 25 fs frame, returning -3.7e7 eV where true
nuclear repulsion is about +1200 eV). The architectural cause is that MACE's
B-features are polynomials in a learned density, and polynomials have no floor.

ZBL supplies the missing floor. It is a purely pairwise function of the two
atomic numbers and the distance, so it needs NO bond detection: below ~0.5 A
every pair repels regardless of what it is bonded to. That is what makes this
failure mode fixable, in contrast to the OMol25 global-label failure, which
needs a system decomposition the architecture cannot express.

WHY AN EXTERNAL WRAPPER RATHER THAN PATCHING MACE
-------------------------------------------------
mace-torch already ships this: `ZBLBasis` in mace/modules/radial.py, enabled by
`pair_repulsion=True` at model construction (`--pair_repulsion` when training).
Its forward() computes, per directed edge,

    v_edge = 0.5 * (14.3996 Z_u Z_v / r) * phi(r/a) * envelope(r, r_cov, p)

and scatter-sums onto the receiver node. Every pair appears as TWO directed
edges (u->v and v->u), so the 0.5 cancels and the model total is exactly

    E_ZBL = sum over UNIQUE PAIRS of (14.3996 Z_i Z_j / r) * phi * envelope

There is no message passing, no learned parameter, no dependence on any other
atom. So an external additive calculator reproduces the native term EXACTLY --
and unlike an internal patch it also works on UMA and ANI, which is what we
need for a like-for-like comparison across all five models.

Edge-list caveat, checked: MACE only builds edges within the model cutoff
(4.5-6.0 A here), but the ZBL envelope dies at r_cov_i + r_cov_j <= ~1.5 A for
our elements, so every ZBL-relevant pair is inside any model cutoff. The two
routes see the same pairs.

TWO FLAVOURS, AND WHY THE CHOICE MATTERS
----------------------------------------
    flavor="mace"      a = 0.4543 * 0.529 / (Z1^0.300 + Z2^0.300)
    flavor="textbook"  a = 0.46850        / (Z1^0.23  + Z2^0.23 )

These are NOT the same potential. For O-H the screening lengths are 0.084 A and
0.179 A -- MACE screens more than twice as fast, so its wall is far softer:

    r(O-H)      MACE ZBL    textbook ZBL    bare Z1Z2/r
    0.960 A        0.0 eV         5.4 eV        120.0 eV
    0.856 A        0.1 eV         7.7 eV        134.6 eV   <- collapse began
    0.700 A        1.0 eV        13.8 eV        164.6 eV
    0.500 A        6.9 eV        33.2 eV        230.4 eV
    0.096 A      452.5 eV       710.0 eV       1200.0 eV   <- collapse ended

kT at 3500 K is 0.30 eV. MACE's default supplies 0.1 eV at the distance where
the collapse began, so it would probably NOT have stopped it. Do not assume
"add ZBL" fixes this -- measure it. That is what the flavour switch is for.

The softness is deliberate, not a bug: textbook ZBL adds 13.8 eV at 0.7 A to an
O-H bond only 4.8 eV deep, which would wreck the chemistry. A wall stiff enough
to hold at 0.85 A is stiff enough to corrupt bonding at 0.95 A. MACE resolved
that tension toward not corrupting bonding. `ZBLBasis(trainable=True)` exists
in their code precisely because the right answer is not obvious.

DOUBLE COUNTING -- READ THIS BEFORE CLAIMING A FIX
--------------------------------------------------
Osaka24 was trained WITHOUT this term. Whatever it learned below the covalent
contact distance is still in the network; adding ZBL does not remove it. So the
result is (ZBL) + (whatever the model already does there), not a clean
replacement. This is defensible ONLY because:
  (a) the envelope is identically zero at and beyond r_cov_i + r_cov_j, so no
      geometry that real chemistry visits is touched -- every equilibrium
      property must come back numerically UNCHANGED, and that is the control
      that makes the experiment interpretable; and
  (b) inside that radius the model is wrong by seven orders of magnitude, so
      there is no accuracy left to preserve.
The clean version is delta-learning (subtract ZBL from the labels, retrain on
the residual, add it back), which needs a retrain we are not doing. Frame the
result as an intervention experiment, not as a better model.

USAGE
-----
    from zbl_patch import ZBLCalculator
    calc = ZBLCalculator(base_calc, flavor="mace")      # or "textbook"

    # standalone:
    E, F = zbl_energy_forces(atoms, flavor="textbook")

    # self-test (analytic forces vs finite difference, and the pairwise
    # equivalence proof above, evaluated numerically):
    python zbl_patch.py --selftest
"""
import numpy as np
from ase.data import covalent_radii
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list

E2 = 14.3996          # e^2/(4 pi eps0), eV * Angstrom -- MACE's value
C = np.array([0.18175, 0.50986, 0.28022, 0.02817])
B = np.array([3.19980, 0.94229, 0.40290, 0.20162])
# MACE hard-codes slightly rounded coefficients; use theirs for flavor="mace"
C_MACE = np.array([0.1818, 0.5099, 0.2802, 0.02817])
B_MACE = np.array([3.2, 0.9423, 0.4028, 0.2016])

FLAVORS = {
    #             prefactor            exponent   c        b
    "mace":     (0.4543 * 0.529,       0.300,     C_MACE,  B_MACE),
    "textbook": (0.46850,              0.23,      C,       B),
}


def _screen(x, c, b):
    """phi(x) and dphi/dx."""
    e = np.exp(-b[None, :] * x[:, None])
    return (c * e).sum(1), (-c * b * e).sum(1)


def _envelope(q, p):
    """
    MACE's polynomial cutoff and its derivative, on q = r / r_cov.

    p(0)=1 and p(1)=p'(1)=p''(1)=0 -- the same Gasteiger envelope that produces
    the long-range seam in Finding 20. Here it is doing useful work: it is what
    guarantees the correction vanishes identically at the covalent contact
    distance, so nothing above that radius can be perturbed.
    """
    c1 = (p + 1.0) * (p + 2.0) / 2.0
    c2 = p * (p + 2.0)
    c3 = p * (p + 1.0) / 2.0
    env = 1.0 - c1 * q ** p + c2 * q ** (p + 1) - c3 * q ** (p + 2)
    d = (-c1 * p * q ** (p - 1) + c2 * (p + 1) * q ** p
         - c3 * (p + 2) * q ** (p + 1))
    inside = q < 1.0
    return env * inside, d * inside


def zbl_pair(r, z1, z2, flavor="mace", p=6, scale=1.0):
    """
    V(r) and dV/dr for one pair type, eV and eV/A. Arrays in, arrays out.

    V = scale * (E2 Z1 Z2 / r) * phi(r/a) * env(r / (rcov1 + rcov2))
    """
    r = np.atleast_1d(np.asarray(r, float))
    z1 = np.atleast_1d(np.asarray(z1, int))
    z2 = np.atleast_1d(np.asarray(z2, int))
    pref, expo, c, b = FLAVORS[flavor]
    a = pref / (z1.astype(float) ** expo + z2.astype(float) ** expo)
    rcov = covalent_radii[z1] + covalent_radii[z2]

    x = r / a
    q = r / rcov
    phi, dphi = _screen(x, c, b)
    env, denv = _envelope(q, p)

    coul = E2 * z1 * z2 / r
    V = scale * coul * phi * env
    dV = scale * (-coul / r * phi * env
                  + coul * dphi / a * env
                  + coul * phi * denv / rcov)
    return V, dV


def zbl_energy_forces(atoms, flavor="mace", p=6, scale=1.0):
    """
    Total ZBL energy (eV) and forces (eV/A) for an Atoms object.

    PBC-aware via ase.neighborlist. The global cutoff is the largest possible
    covalent-radius sum over the species actually present, so no relevant pair
    is missed and no irrelevant pair is built.
    """
    zs = atoms.get_atomic_numbers()
    if len(zs) < 2:
        return 0.0, np.zeros((len(zs), 3))
    rcut = float(2.0 * covalent_radii[np.unique(zs)].max())
    i, j, D, d = neighbor_list("ijDd", atoms, rcut)
    if len(i) == 0:
        return 0.0, np.zeros((len(zs), 3))

    V, dV = zbl_pair(d, zs[i], zs[j], flavor=flavor, p=p, scale=scale)
    # neighbor_list returns EACH pair twice (i->j and j->i), so halve the energy
    E = 0.5 * float(V.sum())

    # >>> NO 0.5 HERE, AND THIS IS EASY TO GET WRONG <<<
    # The 0.5 above is correct for the ENERGY because neighbor_list yields each
    # pair as two directed edges. But the force loop accumulates only onto the
    # FIRST index, so atom a picks up its contribution solely from edge (a,b)
    # and atom b solely from edge (b,a). Each atom therefore sees its pair once,
    # not twice, and halving here would make every force a factor of 2 too
    # small. (Caught by the finite-difference check in _selftest, which the
    # first version failed by exactly 2x.)
    #
    # D = r_j - r_i, and F_i = -dE/dr_i = +V'(d) * D/d. V' < 0 for a repulsive
    # pair, so F_i points along -D, i.e. away from j. Correct.
    F = np.zeros((len(zs), 3))
    np.add.at(F, i, (dV / d)[:, None] * D)
    return E, F


class ZBLCalculator(Calculator):
    """
    Additive ZBL wall on top of any ASE calculator.

    The base calculator is left completely untouched -- no monkeypatching, no
    forward hooks, no weight changes. Above r_cov_i + r_cov_j this class adds
    EXACTLY ZERO, so any property computed at or near equilibrium is bit-for-bit
    the base model's answer. That is the control.
    """
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, base, flavor="mace", p=6, scale=1.0, **kw):
        super().__init__(**kw)
        if flavor not in FLAVORS:
            raise ValueError(f"flavor must be one of {sorted(FLAVORS)}")
        self.base = base
        self.flavor = flavor
        self.p = p
        self.scale = scale

    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        a = atoms.copy()
        a.calc = self.base
        e = a.get_potential_energy()
        f = a.get_forces()
        dE, dF = zbl_energy_forces(atoms, self.flavor, self.p, self.scale)
        self.results = {"energy": e + dE, "free_energy": e + dE,
                        "forces": f + dF,
                        "energy_base": e, "energy_zbl": dE}

    def __repr__(self):
        return (f"ZBLCalculator(base={self.base.__class__.__name__}, "
                f"flavor={self.flavor!r}, p={self.p}, scale={self.scale})")


# ----------------------------------------------------------------------

def _selftest():
    """Analytic forces vs finite difference, plus the pairwise-equivalence check."""
    from ase import Atoms
    from ase.calculators.calculator import Calculator as _C

    print("1. analytic dV/dr vs finite difference (pair level)")
    ok = True
    for flavor in FLAVORS:
        for (z1, z2) in [(8, 1), (1, 1), (6, 8), (7, 8)]:
            r = np.linspace(0.15, 1.6, 60)
            _, dV = zbl_pair(r, z1, z2, flavor)
            h = 1e-6
            V1, _ = zbl_pair(r + h, z1, z2, flavor)
            V0, _ = zbl_pair(r - h, z1, z2, flavor)
            num = (V1 - V0) / (2 * h)
            # skip the envelope kink at q = 1 where the 3rd derivative jumps
            m = np.abs(num) > 1e-8
            err = np.max(np.abs(dV[m] - num[m]) / np.abs(num[m]))
            flag = "ok " if err < 1e-5 else "FAIL"
            ok &= err < 1e-5
            print(f"   {flag} {flavor:<9} Z{z1}-Z{z2}  max rel err {err:.2e}")

    print("\n2. analytic forces vs finite difference (3D, with PBC)")
    rng = np.random.default_rng(0)
    for pbc in (False, True):
        # 1.4 A box, not 2.2: the earlier version drew positions so sparse that
        # NO pair fell inside a covalent contact, so E was 0 and the test passed
        # vacuously while the forces were in fact wrong by 2x. Squeeze them.
        at = Atoms("OHHOHH", positions=rng.uniform(0, 1.4, (6, 3)),
                   cell=[6, 6, 6], pbc=pbc)
        E, F = zbl_energy_forces(at, "textbook")
        assert E > 0.5, "test geometry has no close contacts -- vacuous"
        num = np.zeros_like(F)
        h = 1e-5
        for k in range(len(at)):
            for d in range(3):
                p0 = at.get_positions()
                p0[k, d] += h; at.set_positions(p0)
                Ep, _ = zbl_energy_forces(at, "textbook")
                p0[k, d] -= 2 * h; at.set_positions(p0)
                Em, _ = zbl_energy_forces(at, "textbook")
                p0[k, d] += h; at.set_positions(p0)
                num[k, d] = -(Ep - Em) / (2 * h)
        err = np.max(np.abs(F - num))
        flag = "ok " if err < 1e-4 else "FAIL"
        ok &= err < 1e-4
        print(f"   {flag} pbc={str(pbc):<5}  E = {E:9.3f} eV   "
              f"max |F_analytic - F_numeric| = {err:.2e} eV/A")

    print("\n3. equivalence to MACE's scatter-sum formulation")
    at = Atoms("OHHOHH", positions=rng.uniform(0, 2.0, (6, 3)))
    E, _ = zbl_energy_forces(at, "mace")
    # replicate ZBLBasis.forward: per DIRECTED edge, 0.5 * v, summed over nodes
    zs = at.get_atomic_numbers()
    i, j, d = neighbor_list("ijd", at, 4.0)
    V, _ = zbl_pair(d, zs[i], zs[j], "mace")
    E_native = float((0.5 * V).sum())
    print(f"   {'ok ' if abs(E - E_native) < 1e-9 else 'FAIL'} "
          f"wrapper {E:.10f} eV   MACE-style {E_native:.10f} eV   "
          f"diff {abs(E - E_native):.2e}")
    ok &= abs(E - E_native) < 1e-9

    print("\n4. the correction is EXACTLY zero above the covalent contact")
    w = Atoms("OH", positions=[[0, 0, 0], [0, 0, 0.9572]])
    for r in [0.96, 0.97, 0.98, 1.10, 1.50]:
        w.set_positions([[0, 0, 0], [0, 0, r]])
        E, F = zbl_energy_forces(w, "mace")
        want = "must be 0" if r >= 0.97 else "may be > 0 (inside cutoff)"
        got = "zero" if (E == 0.0 and not F.any()) else "nonzero"
        print(f"   r = {r:.2f} A (r_cov sum = 0.97)   E = {E:.3e} eV   "
              f"{got:<7} [{want}]")
        if r >= 0.97:
            ok &= (E == 0.0)

    print("\n5. base calculator is untouched above the cutoff")

    class Const(_C):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",),
                      system_changes=all_changes):
            _C.calculate(self, atoms, properties, system_changes)
            self.results = {"energy": -12.3456789,
                            "forces": np.ones((len(atoms), 3))}

    w.set_positions([[0, 0, 0], [0, 0, 1.10]])
    w.calc = ZBLCalculator(Const(), flavor="mace")
    e = w.get_potential_energy(); f = w.get_forces()
    same = (e == -12.3456789) and np.all(f == 1.0)
    print(f"   {'ok ' if same else 'FAIL'} wrapped energy {e!r} "
          f"(base -12.3456789), forces unchanged: {np.all(f == 1.0)}")
    ok &= same

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


def _table():
    print(f"{'r(O-H)':>8}{'MACE ZBL':>12}{'textbook':>12}{'bare Z1Z2/r':>14}")
    for r in [0.096, 0.2, 0.3, 0.5, 0.7, 0.8, 0.856, 0.9, 0.96, 0.97, 1.0]:
        m = zbl_pair(r, 8, 1, "mace")[0][0]
        t = zbl_pair(r, 8, 1, "textbook")[0][0]
        print(f"{r:>8.3f}{m:>12.1f}{t:>12.1f}{E2 * 8 / r:>14.1f}")


if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    if a.table:
        _table()
    sys.exit(_selftest() if a.selftest or not a.table else 0)
