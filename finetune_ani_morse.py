#!/usr/bin/env python
"""
finetune_ani_morse.py -- transfer-learn ANI-1xnr on energy-only PCDA-Morse data
===============================================================================

THE RECIPE (dos Santos JCIM 2025 limitations section, implemented literally)
----------------------------------------------------------------------------
The Morse correction is pairwise, so the augmented frames carry trustworthy
ENERGIES but no meaningful intra-fragment FORCES (the fragments are rigid at
in-molecule geometries; their true forces are nonzero and the pairwise form
cannot supply them). Therefore:

    * Morse frames  : energy-only loss (force weight identically zero)
    * anchor frames : energy loss against labels from the FROZEN original model

ANCHORS ARE WHAT MAKE THIS SAFE. ~660 energy-only frames alone would cause
catastrophic forgetting -- the network drifts everywhere the new data is silent.
The rehearsal set pins it: perturbed near-equilibrium geometries of every
species in the study (parents, fragments, atoms), labelled by the frozen
pre-finetune model. Those labels are free (ANI evaluating itself) and define
"do not change" everywhere except the dissociation tails being corrected.

A further free consistency: the Morse targets were built as
E_ANI(r_e) + V_Morse(r), i.e. ON THE MODEL'S OWN ENERGY SCALE. There is no
reference-offset problem; the network learns a shape correction only.

THE ITERATION LOOP
------------------
    1. --selftest on your machine (2 min, mock network -- proves the loop)
    2. train member 0 only (--members 0), defaults
    3. --eval: reruns the 26-bond BDE table via ani_bde_replicate  (~minutes)
    4. read the verdict block and adjust ONE knob:
         radical-radical bonds unmoved      -> raise --w-morse (2 -> 5 -> 10)
         atom-producing bonds drifted       -> raise --w-anchor, or lower --lr
         both fine                          -> train all 8: --members 0 1 2 3 4 5 6 7
    5. only after the table looks right: rerun one MD temperature.

WHAT TO EXPECT: the BDE moves toward (BDE_original + correction), not exactly to
the Morse asymptote -- the network interpolates between competing data. Success
for CH3NO2 C-N is landing anywhere near 55-65 from 17.6, with CH4 C-H still
within ~2 of 116.8.

UNITS: torchani works in Hartree internally; the extxyz energies and anchor
labels are eV (ASE convention). Conversion happens in exactly one place
(EV2HA). If a fine-tuned model returns garbage-scale BDEs, look there first.

USAGE
    python finetune_ani_morse.py --selftest
    python finetune_ani_morse.py --data nm_morse.xyz --members 0
    python finetune_ani_morse.py --eval finetuned_ani/member0.pt
"""
import argparse, json, os, sys, time
import numpy as np

EV2HA = 1.0 / 27.211386245988
EV_KCAL = 23.060548

# ----------------------------------------------------------------------
# dataset assembly (no torch needed -- testable anywhere)
# ----------------------------------------------------------------------

def load_morse_frames(path):
    """
    extxyz written by morse_augment.py.

    >>> ASE QUIRK (caught by the sandbox test, would have bitten for real) <<<
    'energy' is a SPECIAL key in the extxyz format: morse_augment writes it via
    atoms.info, but on read ASE routes it into a SinglePointCalculator instead
    of putting it back in info. So info.get('energy') comes back None for a file
    we wrote ourselves. Check the calculator first, info second.
    """
    from ase.io import read as ase_read
    frames = ase_read(path, index=":")
    out = []
    for k, a in enumerate(frames):
        e = None
        if a.calc is not None:
            try:
                e = float(a.get_potential_energy())
            except Exception:
                e = None
        if e is None:
            e = a.info.get("energy")
        if e is None:
            raise SystemExit(f"frame {k} of {path} has no energy (checked "
                             f"calculator and info) -- was this written by "
                             f"morse_augment.py?")
        out.append((a.get_chemical_symbols(), a.get_positions().copy(),
                    float(e), "morse", a.info.get("species", "?")))
    return out


def make_anchor_frames(calc, n_per=8, sigma=0.03, seed=0, float_species=None):
    """
    Rehearsal set: every species in the BDE study (parents, fragments, atoms),
    rattled n_per times at sigma Angstrom, labelled by the FROZEN model.

    sigma=0.03 is thermal-ish displacement; it covers the region where the
    original model is trusted without straying into the tails being corrected.
    The equilibrium (unrattled) frame is included too -- it is the single most
    important geometry to preserve.
    """
    from ani_bde_replicate import REF, FRAGS, ATOMS, build
    rng = np.random.default_rng(seed)
    species = sorted(set(list(REF)) | {f for p in FRAGS.values() for f in p}
                     | set(ATOMS))
    out = []
    for nm in species:
        base = build(nm)
        for k in range(n_per + 1):
            a = base.copy()
            if k > 0 and len(a) > 1:
                a.set_positions(a.get_positions()
                                + rng.normal(0, sigma, a.positions.shape))
            a.calc = calc
            try:
                e = float(a.get_potential_energy())
                # >>> FORCES ON ANCHORS -- added after the first real fine-tune
                # FAILED (radical-radical mean|err| 20.9 -> 37.2, with untouched
                # species like N2 drifting -29). Energy-only training matched
                # the data but lowered the surface OFF the data; BFGS in the
                # BDE eval then relaxed into those spurious low spots. This is
                # dos Santos' "well-known disadvantages of not including force
                # data", on schedule. Morse frames CANNOT carry forces (the
                # pairwise form defines none for rigid fragments) -- but anchor
                # labels come from the frozen model, whose forces are exact,
                # free, and pin the curvature so relaxations cannot wander.
                f = a.get_forces().copy() if len(a) > 1 else None
            except Exception:
                continue
            # >>> FLOATED SPECIES GET FORCE-ONLY ANCHORS <<<
            # ANI's cutoff is 5.2 A, so a Morse frame at r = 6 A IS two
            # isolated fragments to the network -- the tail frames and the
            # isolated-fragment anchors constrain the SAME outputs. The tails
            # demand fragment energies rise by (De - D_old); an energy anchor
            # demands they stay put. That contradiction produced the lumpy
            # CH3ONO curve, the short asymptote, and the -29 drift on N2. And
            # the tails are RIGHT: the diagnosis is that ANI over-stabilises
            # these fragments (~9 kcal/mol each); energy-pinning them protected
            # the disease. Forces are derivatives, blind to a constant shift,
            # so force-only anchors pin geometry and curvature while leaving
            # the absolute offset free -- exactly the broken degree of freedom.
            #
            # WHICH species float is derived from the DATA, not from
            # radical-ness: float exactly the fragments of augmented bonds
            # (the Morse scans are equations relating parents to fragment
            # pairs; energy-anchored parents and atoms are the boundary
            # conditions; what the equations must re-label is left free). The
            # first version used electron-count parity instead -- equivalent
            # for homolysis of closed-shell parents (even always splits
            # odd+odd) but wrong in general: triplet O2 is even-electron and
            # open-shell, and parity would also float radicals no data asks to
            # move, leaving them unconstrained for no reason.
            if float_species is None:                    # parity fallback
                floats = int(sum(a.get_atomic_numbers())) % 2 == 1
            else:
                floats = nm in float_species
            kind = "anchor_f" if (floats and len(a) > 1) else "anchor"
            out.append((a.get_chemical_symbols(), a.get_positions().copy(),
                        e, kind, nm, f))
    return out


def split(data, frac=0.1, seed=1):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data))
    n = max(int(len(data) * frac), 1)
    return [data[i] for i in idx[n:]], [data[i] for i in idx[:n]]


# ----------------------------------------------------------------------
# torch side
# ----------------------------------------------------------------------

def get_member(info, member):
    """
    One ensemble member as a trainable torch module + its species order.

    Fine-tune members INDIVIDUALLY (same data, different starting weights) so
    the ensemble keeps its spread -- that spread is the uncertainty estimate
    dos Santos lean on, and training the averaged ensemble would collapse it.
    """
    import torchani.neurochem as nc
    model = nc.load_model_from_info_file(info, model_index=member)
    # species order: the neurochem Sequential knows its constants; fall back
    # to the ASE calculator's species attribute if the API differs
    for attr in ("species",):
        if hasattr(model, attr):
            return model, list(getattr(model, attr))
    return model, ["H", "C", "N", "O"]      # ANI-1xnr element set


def energies_batch(model, species_order, batch, device="cpu"):
    """
    Model energies (Hartree) for a list of (symbols, positions, ...) rows.

    >>> SPECIES ENCODING DEPENDS ON THE MODEL (fixed 2026-08-11) <<<
    torchani models take species as EITHER 0-based internal indices OR atomic
    numbers, controlled by model.periodic_table_index. The first run fed
    internal indices to a model built with periodic_table_index=True and died
    with "Supported elements are: tensor([1, 6, 7, 8])" -- that tensor is H/C/
    N/O as Z values, i.e. the model saying "give me atomic numbers". Detect the
    flag instead of assuming either convention; mock models without the
    attribute get the old index behaviour, which keeps the selftest valid.
    """
    import torch
    from ase.data import atomic_numbers
    if bool(getattr(model, "periodic_table_index", False)):
        s2i = {s: atomic_numbers[s] for s in species_order}
    else:
        s2i = {s: i for i, s in enumerate(species_order)}
    es = []
    for row in batch:
        syms, pos = row[0], row[1]
        sp = torch.tensor([[s2i[s] for s in syms]], dtype=torch.long,
                          device=device)
        xyz = torch.tensor(pos[None], dtype=torch.float32, device=device,
                           requires_grad=False)
        es.append(model((sp, xyz)).energies)
    return torch.cat(es)


def train(model, species_order, tr, va, lr=1e-4, epochs=300, w_morse=2.0,
          w_anchor=1.0, w_force=0.3, patience=30, device="cpu", log=print,
          shape_only=frozenset()):
    """
    shape_only : species whose MORSE frames are compared offset-free.

    >>> WHY (fixed 2026-08-18) <<<
    A species can be BOTH a floated fragment and an augmented parent. NO2 is:
    as a fragment of CH3NO2 its energy must RISE by ~50 kcal/mol (CH3 is already
    pinned by CH4's equation, so nearly the whole correction has to come from
    NO2), but morse_curve() builds NO2's own frames as
        E = E_min(NO2) + V_Morse(r)
    with E_min taken from the ORIGINAL model -- so those frames energy-anchor NO2
    at exactly the value it has to move away from. The two demands are
    contradictory, and the measured result was NO2's own frame RMSE getting
    WORSE (15.3 -> 21.9 kcal/mol, the only species to degrade) while its BDE
    fell 45 kcal/mol.

    The fix is the same one that repaired the rehearsal anchors, applied one
    level down: for these species compare the SHAPE of the curve and not its
    absolute position, by centring predictions and targets on their own means
    before taking the residual. The Morse curve then teaches how the energy
    varies with bond length -- which is what it is good for -- and says nothing
    about the offset, which the other equations determine.
    """
    """
    Energy-only fine-tune.

    loss = mean over frames of  w(frame) * ((E_pred - E_target)/sqrt(N))^2

    * /sqrt(N) is the ANI convention -- keeps big and small molecules on the
      same footing so the atoms/diatomics in the anchor set are not swamped.
    * per-frame weights implement the Morse-vs-anchor balance. w_morse is THE
      knob of the iteration loop; start at 2, raise if the targeted BDEs
      refuse to move, lower (or raise w_anchor) if the control bonds drift.
    * energies are Hartree inside the loss: eV errors / 27.2 look deceptively
      small, so report validation RMSE back in kcal/mol where it means something.
    """
    import torch
    model = model.to(device).train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=lr)
    # >>> min_lr IS LOAD-BEARING (found by running the selftest, 2026-08-11) <<<
    # Without a floor, every 10-epoch noise plateau halves the lr, and by
    # mid-training it has collapsed to ~1e-9: the model froze mid-fit at
    # 30 kcal/mol RMSE with the repulsive wall completely unfit, and stage A of
    # the selftest failed at 4.2 eV. With patience 40 and a floor at lr/20 the
    # same run converges to 0.17 eV. If a real ANI fine-tune ever stalls with
    # val flat for hundreds of epochs, suspect this mechanism first.
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5,
                                                       patience=40,
                                                       min_lr=lr * 0.05)

    from ase.data import atomic_numbers
    use_z = bool(getattr(model, "periodic_table_index", False))
    s2i = ({s: atomic_numbers[s] for s in species_order} if use_z
           else {s: i for i, s in enumerate(species_order)})

    def batch_loss(rows, with_forces=True):
        e_pred = energies_batch(model, species_order, rows, device)
        e_tgt = torch.tensor([r[2] * EV2HA for r in rows],
                             dtype=e_pred.dtype, device=device)
        n = torch.tensor([len(r[0]) for r in rows], dtype=e_pred.dtype,
                         device=device)
        # anchor_f rows (odd-electron fragments) contribute NO energy term --
        # their absolute energy is exactly what the Morse tails must move
        w = torch.tensor([w_morse if r[3] == "morse"
                          else 0.0 if r[3] == "anchor_f" else w_anchor
                          for r in rows], dtype=e_pred.dtype, device=device)
        # shape-only species: centre both prediction and target on their own
        # means within this batch so the residual carries no offset information
        if shape_only:
            idx = [k for k, r in enumerate(rows)
                   if r[3] == "morse" and r[4] in shape_only]
            if len(idx) >= 2:
                sel = torch.tensor(idx, device=device)
                ep, et = e_pred[sel], e_tgt[sel]
                e_pred = e_pred.clone(); e_tgt = e_tgt.clone()
                e_pred[sel] = ep - ep.mean()
                e_tgt[sel] = et - et.mean()
        loss = (w * ((e_pred - e_tgt) / n.sqrt()) ** 2).mean()

        # force term -- ANCHOR rows only, and only those carrying labels.
        # F_pred = -dE/dxyz via autograd; reference forces are eV/A -> Ha/A.
        if with_forces and w_force > 0:
            fl, nf = None, 0
            for r in rows:
                if len(r) < 6 or r[5] is None:
                    continue
                sp = torch.tensor([[s2i[s] for s in r[0]]], dtype=torch.long,
                                  device=device)
                xyz = torch.tensor(r[1][None], dtype=torch.float32,
                                   device=device, requires_grad=True)
                e = model((sp, xyz)).energies.sum()
                g, = torch.autograd.grad(e, xyz, create_graph=True)
                f_ref = torch.tensor(r[5][None] * EV2HA, dtype=g.dtype,
                                     device=device)
                term = ((-g - f_ref) ** 2).mean()
                fl = term if fl is None else fl + term
                nf += 1
            if nf:
                loss = loss + w_force * fl / nf
        return loss, ((e_pred - e_tgt).abs() / EV2HA * EV_KCAL)

    best, best_state, since = np.inf, None, 0
    hist = []
    rng = np.random.default_rng(2)
    for ep in range(epochs):
        idx = rng.permutation(len(tr))
        tot = 0.0
        for k in range(0, len(idx), 32):
            rows = [tr[i] for i in idx[k:k + 32]]
            opt.zero_grad()
            L, _ = batch_loss(rows)
            L.backward()
            opt.step()
            tot += float(L.detach()) * len(rows)
        with torch.no_grad():
            # energy-only for validation: autograd.grad cannot run under
            # no_grad, and the va energy RMSE is the early-stop signal anyway
            Lv, err_kcal = batch_loss(va, with_forces=False)
            v = float(Lv)
            v_rmse = float((err_kcal ** 2).mean().sqrt())
        sched.step(v)
        hist.append(dict(epoch=ep, train=tot / len(tr), val=v,
                         val_rmse_kcal=v_rmse))
        if v < best - 1e-12:
            best, since = v, 0
            best_state = {k: p.detach().clone()
                          for k, p in model.state_dict().items()}
        else:
            since += 1
        if ep % 10 == 0 or since == 0:
            log(f"  ep {ep:4d}  train {tot/len(tr):.3e}  val {v:.3e}  "
                f"val RMSE {v_rmse:6.2f} kcal/mol  (best {'*' if since==0 else since})")
        if since >= patience:
            log(f"  early stop at epoch {ep} (no val improvement in {patience})")
            break
    if best_state:
        model.load_state_dict(best_state)
    return model, hist


# ----------------------------------------------------------------------
# evaluation: rerun the BDE table with the fine-tuned model
# ----------------------------------------------------------------------

def evaluate(calc, label):
    """26-bond BDE errors via ani_bde_replicate's own machinery."""
    from ani_bde_replicate import REF, FRAGS, relaxed_energy
    rows = {}
    for nm, (bond, de, d0, note) in REF.items():
        fa, fb = FRAGS[nm]
        try:
            bde = (relaxed_energy(fa, calc, cache={})
                   + relaxed_energy(fb, calc, cache={})
                   - relaxed_energy(nm, calc, cache={})) * EV_KCAL
        except Exception as exc:
            print(f"  {nm}: FAILED {type(exc).__name__}")
            continue
        rows[nm] = bde - de
    return rows


def rigid_scan_overlay(calc_before, calc_after, species, out_png,
                       rmax=6.0, n=60, meta_json="nm_morse_meta.json"):
    """
    Rigid scan: original vs fine-tuned vs the Morse target, one panel per bond.

    The BDE table checks ENDPOINTS; this checks the PATH -- specifically
      * is the spurious peak gone (ANI-1xnr put a 28.8 kcal/mol hump at 2.7 A
        on nitromethane C-N that decayed back to the plateau),
      * is the handoff region (~2 A, where Morse frames take over from the
        original data) smooth rather than kinked, and
      * does the asymptote flatten near the Morse De rather than overshooting.
    A model can have a perfect BDE and a lumpy path, and MD runs on the path.
    """
    import json as _json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ani_bde_replicate import build, find_scission

    meta = {}
    if os.path.exists(meta_json):
        meta = {m["species"]: m for m in _json.load(open(meta_json))}

    fig, axes = plt.subplots(1, len(species), figsize=(5.6 * len(species), 4.4))
    if len(species) == 1:
        axes = [axes]
    for ax, nm in zip(axes, species):
        ref = build(nm)
        sc = find_scission(nm, ref)
        if sc is None:
            ax.set_title(f"{nm}: no scission"); continue
        i, j, side = sc
        p0 = ref.get_positions()
        u = p0[j] - p0[i]
        r0 = float(np.linalg.norm(u)); u = u / r0
        rs = np.linspace(r0, rmax, n)
        curves = {}
        for lbl, calc in (("original", calc_before), ("fine-tuned", calc_after)):
            E = []
            a = ref.copy(); a.calc = calc
            for r in rs:
                p = p0.copy(); p[side] += u * (r - r0)
                a.set_positions(p)
                try:
                    E.append(a.get_potential_energy())
                except Exception:
                    E.append(np.nan)
            E = np.array(E, float)
            curves[lbl] = (E - np.nanmin(E)) * EV_KCAL
            ax.plot(rs, curves[lbl], "o-" if lbl == "original" else "s-",
                    ms=3, lw=1.6, label=lbl)
        m = meta.get(nm)
        if m:
            a_par = m["a_inv_A"]; De = m["De_kcal"]; re_m = m["re"]
            ax.plot(rs, De * (1 - np.exp(-a_par * (rs - re_m))) ** 2, "--",
                    lw=1.4, color="#d62728", label=f"Morse target ($D_e$={De:.0f})")
        ax.axhline(0, color="k", lw=.6)
        ax.set_xlabel("r / A"); ax.set_ylabel(r"$\Delta E$ / kcal mol$^{-1}$")
        ax.set_title(nm, fontsize=11); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle("Rigid dissociation scans: before vs after PCDA fine-tune",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"\n  scan figure: {out_png}")


def vertical_bde(calc):
    """
    BDEs at FIXED reference geometries -- no relaxation anywhere.

    The discriminator for the first failure: if these track the Morse targets
    while the relaxed table is wrecked, the surface is right ON the data and
    wrong OFF it -- BFGS walked into spurious minima created by energy-only
    training. If the vertical table is wrecked too, the surface itself is
    corrupted and the anchors were too weak.
    """
    from ani_bde_replicate import REF, FRAGS, build
    out = {}
    for nm, (bond, de, d0, note) in REF.items():
        fa, fb = FRAGS[nm]
        try:
            es = []
            for s in (fa, fb, nm):
                a = build(s)
                a.calc = calc
                es.append(float(a.get_potential_energy()))
            out[nm] = (es[0] + es[1] - es[2]) * EV_KCAL - de
        except Exception:
            pass
    return out


def verdict(before, after):
    """The decision block for the iteration loop."""
    from ani_bde_replicate import FRAGS, build
    rad = [n for n in after if all(len(build(f)) > 1 for f in FRAGS[n])]
    atom = [n for n in after if n not in rad]
    print(f"\n  {'species':<12}{'before':>9}{'after':>9}{'moved':>9}   class")
    for n in sorted(after, key=lambda x: (x in atom, x)):
        b, a = before.get(n, np.nan), after[n]
        cls = "radical-radical" if n in rad else "atom-producing"
        print(f"  {n:<12}{b:>+9.1f}{a:>+9.1f}{a-b:>+9.1f}   {cls}")
    r_b = np.mean([abs(before[n]) for n in rad if n in before])
    r_a = np.mean([abs(after[n]) for n in rad])
    a_b = np.mean([abs(before[n]) for n in atom if n in before])
    a_a = np.mean([abs(after[n]) for n in atom])
    # SIGNED mean movement -- the first fine-tune moved 23/26 bonds DOWN,
    # including species not in the training set (N2: -29). |err| comparisons
    # alone cannot see direction, and the original verdict gave wrong advice.
    drift = np.mean([after[n] - before.get(n, after[n]) for n in after])
    print(f"\n  radical-radical mean|err|: {r_b:5.1f} -> {r_a:5.1f}")
    print(f"  atom-producing  mean|err|: {a_b:5.1f} -> {a_a:5.1f}")
    print(f"  signed mean movement (all bonds): {drift:+.1f} kcal/mol")
    if abs(drift) > 5:
        print(f"  VERDICT: SYSTEMATIC {'DOWNWARD' if drift < 0 else 'UPWARD'} "
              f"DRIFT -- one mechanism, not weight balance.")
        print("     Do not touch --w-morse. Compare the VERTICAL (fixed-"
              "geometry) table below:")
        print("     vertical fine, relaxed wrecked -> spurious minima from "
              "energy-only training;")
        print("       the fix is force labels on anchors (--w-force, now on "
              "by default).")
        print("     vertical wrecked too -> the surface itself moved; raise "
              "--w-anchor / more anchors.")
    elif r_a < r_b - 3 and a_a < a_b + 3:
        print("  VERDICT: working. Extend to all 8 members.")
    elif r_a >= r_b - 3:
        print("  VERDICT: targeted bonds unmoved -> raise --w-morse (2->5->10).")
    else:
        print("  VERDICT: control bonds drifted -> raise --w-anchor or lower --lr.")


# ----------------------------------------------------------------------
# self-test: the whole loop on a mock model. RUN THIS FIRST.
# ----------------------------------------------------------------------

def selftest():
    """
    The whole loop on a mock model, staged so a failure names its own cause.

    >>> REWRITTEN 2026-08-11 after the first version FAILED on the user's
    machine -- and the failure was in the TEST, in two ways worth recording:

      1. The check function read the model in Hartree but compared against eV
         thresholds: `float(...) / EV2HA * EV2HA` is a no-op (the conversion
         written and immediately undone). Its reported "tail 0.06 eV" was
         actually 0.06 Ha = 1.6 eV, and its one "ok" -- equilibrium drift
         "+0.050" -- was 1.4 eV of drift, a FALSE PASS. Unit bugs in the test
         are worse than unit bugs in the code, because the test is what you
         trust.
      2. The scenario was unfair: the mock's pretraining curve disagreed with
         the target curve at EVERY radius, so anchors and Morse frames fought
         everywhere. The real ANI case is the opposite -- right near
         equilibrium, wrong only in the tail. The test now pretrains on a curve
         that EQUALS the truth below the saturation point and is wrong only
         beyond it, which is the situation the method is designed for.

    Stages:
      A  fit from random init to the bad curve; verifies plain fitting works
         (if A fails, the training loop itself is broken -- lr, batching, units)
      B  fine-tune with Morse tail frames + near-equilibrium anchors; verifies
         the tail moves AND the equilibrium stays pinned (the actual method)
    All reporting is in eV, converted from the model's Hartree in ONE place.
    """
    import torch

    class Mock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(2, 48), torch.nn.Tanh(),
                torch.nn.Linear(48, 48), torch.nn.Tanh(),
                torch.nn.Linear(48, 1))

        def forward(self, inp):
            sp, xyz = inp
            r = (xyz[:, 0] - xyz[:, 1]).norm(dim=-1, keepdim=True)
            n = torch.full_like(r, float(sp.shape[1]))
            e = self.net(torch.cat([r, n], dim=1)).squeeze(-1)

            class R:            # mimic torchani's .energies
                pass
            out = R(); out.energies = e
            return out

    def good(r):                       # truth: deep Morse, De = 2.7 eV
        return 2.7 * (1 - np.exp(-1.9 * (r - 1.5))) ** 2

    def bad(r):
        # "ANI": correct everywhere except the DISSOCIATION TAIL, which
        # saturates at 0.7 eV. The cap must apply only beyond r_e -- a plain
        # min() would also clip the inner repulsive wall (good(1.0) = 6.8 eV),
        # putting anchors in conflict with truth on the wall and re-creating
        # the unfair-test problem this rewrite exists to fix.
        r = np.asarray(r, float)
        return np.where(r > 1.5, np.minimum(good(r), 0.7), good(r))

    def model_eV(model, r):            # THE one Hartree->eV conversion
        with torch.no_grad():
            sp = torch.zeros(1, 2, dtype=torch.long)
            xyz = torch.tensor([[[0, 0, 0], [r, 0, 0]]], dtype=torch.float32)
            return float(model((sp, xyz)).energies) / EV2HA

    def row(r, e, kind):
        return (["C", "N"], np.array([[0, 0, 0], [r, 0, 0.0]]), e, kind, "CN")

    torch.manual_seed(0)
    rs = np.linspace(1.0, 6.0, 120)

    # ---- stage A: plain fit ------------------------------------------------
    model = Mock()
    pre = [row(r, bad(r), "anchor") for r in rs]
    model, _ = train(model, ["C", "N"], pre, pre[::10], lr=3e-3, epochs=2000,
                     patience=300, log=lambda *a: None)
    fit_err = max(abs(model_eV(model, r) - bad(r)) for r in rs)
    # 0.25 eV: the residual concentrates on the steepest wall point (6.8 eV at
    # r=1.0), where 0.17 eV is a 2.5% miss. This gate certifies the LOOP, not
    # spectroscopic accuracy of a 48-neuron mock.
    ok_A = fit_err < 0.25
    print(f"  A  pretrain fit: max |model - target| = {fit_err:.3f} eV  "
          f"{'ok' if ok_A else 'FAIL -- training loop itself is broken'}")
    if not ok_A:
        print("\n  SELF-TEST FAILED at stage A -- do not train ANI yet")
        return 1

    # ---- stage B: the method ----------------------------------------------
    # bad == good below r = 1.874 (where good crosses 0.7), so anchors at
    # r <= 1.85 are CONSISTENT with the Morse frames at r >= 1.9 -- as in the
    # real ANI case, the two datasets only disagree where the model is wrong.
    morse = [row(r, good(r), "morse") for r in rs if r >= 1.9]
    # anchors now CARRY FORCE LABELS, as in the real pipeline: F = -dE/dr along
    # the bond axis (central difference on bad()), zero transverse. This
    # exercises the autograd force branch end-to-end before it touches ANI.
    def frow(r):
        h = 1e-4
        dEdr = float((bad(r + h) - bad(r - h)) / (2 * h))
        f = np.zeros((2, 3))
        f[0, 0], f[1, 0] = dEdr, -dEdr        # -dE/dxyz with u = +x
        return (["C", "N"], np.array([[0, 0, 0], [r, 0, 0.0]]),
                float(bad(r)), "anchor", "CN", f)
    anch = [frow(r) for r in rs if r <= 1.85]
    tr, va = split(morse + anch, 0.15)
    model, _ = train(model, ["C", "N"], tr, va, lr=1e-3, epochs=2000,
                     w_morse=2.0, w_force=0.3, patience=300,
                     log=lambda *a: None)

    tail = model_eV(model, 6.0) - model_eV(model, 1.5)
    eq = model_eV(model, 1.5) - good(1.5)
    ok_tail = abs(tail - 2.7) < 0.4
    ok_eq = abs(eq) < 0.15
    print(f"  B  dissociation tail: {tail:.2f} eV  (target 2.70, was 0.70)  "
          f"{'ok' if ok_tail else 'FAIL'}")
    print(f"  B  equilibrium drift: {eq:+.3f} eV  (anchors should pin it)  "
          f"{'ok' if ok_eq else 'FAIL'}")
    ok = ok_tail and ok_eq
    print(f"\n  {'SELF-TEST PASSED -- safe to train ANI' if ok else 'SELF-TEST FAILED -- do not train ANI yet'}")
    return 0 if ok else 1


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--info", default="ani-1xnr/model/ani-1xnr.info")
    ap.add_argument("--data", default="nm_morse.xyz")
    ap.add_argument("--members", type=int, nargs="+", default=[0],
                    help="ensemble members to fine-tune (individually)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--w-morse", type=float, default=2.0,
                    help="THE knob. Raise if targeted BDEs refuse to move.")
    ap.add_argument("--w-anchor", type=float, default=1.0)
    ap.add_argument("--w-force", type=float, default=0.3,
                    help="force-matching weight on ANCHOR frames (frozen-model "
                         "force labels). 0 disables -- which is what the first "
                         "failed fine-tune effectively ran.")
    ap.add_argument("--anchors-per", type=int, default=8)
    ap.add_argument("--no-shape-only", action="store_true",
                    help="disable offset-free treatment of Morse frames for "
                         "species that are both floated fragments and augmented "
                         "parents (NO2). Only for reproducing the old failure.")
    ap.add_argument("--outdir", default="finetuned_ani")
    ap.add_argument("--eval", default=None, metavar="CKPT",
                    help="skip training; evaluate this checkpoint's BDE table")
    ap.add_argument("--scan", nargs="+", default=["CH3NO2", "CH3ONO"],
                    metavar="SPECIES",
                    help="rigid-scan overlay species for the --eval figure")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    import torch
    os.makedirs(args.outdir, exist_ok=True)

    from ani_bde_replicate import load_ani
    frozen_calc = load_ani(args.info)            # ensemble, for labels+baseline

    if args.eval:
        member = int(os.path.basename(args.eval).split("member")[1].split(".")[0])
        model, order = get_member(args.info, member)
        model.load_state_dict(torch.load(args.eval, map_location=args.device))
        base_model, _ = get_member(args.info, member)
        print(f"=== BDE table: member {member}, before vs after fine-tune ===")
        before = evaluate(base_model.ase(), "before")
        after = evaluate(model.ase(), "after")
        verdict(before, after)

        print("\n=== VERTICAL BDEs (fixed reference geometries, no relaxation) ===")
        vb = vertical_bde(base_model.ase())
        va_ = vertical_bde(model.ase())
        print(f"  {'species':<12}{'before':>9}{'after':>9}{'moved':>9}")
        for nm in sorted(va_):
            if nm in vb:
                print(f"  {nm:<12}{vb[nm]:>+9.1f}{va_[nm]:>+9.1f}"
                      f"{va_[nm]-vb[nm]:>+9.1f}")
        print("  (compare against the relaxed table: divergence between the "
              "two localises\n   the failure to the relaxation, i.e. to "
              "off-data spurious minima)")
        if args.scan:
            rigid_scan_overlay(base_model.ase(), model.ase(), args.scan,
                               os.path.join(os.path.dirname(args.eval) or ".",
                                            f"scan_member{member}.png"))
        return

    print("loading Morse frames + building anchor set (frozen-model labels)...")
    morse = load_morse_frames(args.data)
    # float set = fragments of the bonds actually present in the Morse data
    from ani_bde_replicate import FRAGS, ATOMS
    aug_parents = {r[4] for r in morse}
    float_species = {f for p in aug_parents for f in FRAGS.get(p, ())} - ATOMS
    print(f"  augmented parents: {sorted(aug_parents)}")
    print(f"  floated (force-only) fragments: {sorted(float_species)}")
    # species that are BOTH a floated fragment AND an augmented parent: their
    # own Morse frames must not pin an offset the other equations have to move
    shape_only = float_species & aug_parents
    if shape_only:
        print(f"  shape-only Morse frames (floated AND augmented parents): "
              f"{sorted(shape_only)}")
    anchors = make_anchor_frames(frozen_calc, n_per=args.anchors_per,
                                 float_species=float_species)
    n_f = sum(1 for r in anchors if r[3] == "anchor_f")
    print(f"  {len(morse)} morse frames, {len(anchors)} anchor frames "
          f"({n_f} force-only)")
    tr, va = split(morse + anchors)

    for m in args.members:
        print(f"\n=== fine-tuning ensemble member {m} ===")
        model, order = get_member(args.info, m)
        t0 = time.time()
        model, hist = train(model, order, tr, va, lr=args.lr,
                            epochs=args.epochs, w_morse=args.w_morse,
                            w_anchor=args.w_anchor, w_force=args.w_force,
                            device=args.device,
                            shape_only=frozenset() if args.no_shape_only
                            else shape_only)
        ck = os.path.join(args.outdir, f"member{m}.pt")
        torch.save(model.state_dict(), ck)
        json.dump(hist, open(ck.replace(".pt", "_history.json"), "w"))
        print(f"  saved {ck}  ({time.time()-t0:.0f}s)")

    print(f"\nNEXT: python {sys.argv[0]} --eval {args.outdir}/member{args.members[0]}.pt")


if __name__ == "__main__":
    main()
