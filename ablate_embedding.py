#!/usr/bin/env python
"""
ablate_embedding.py
===================

Test D: does the global charge/spin EMBEDDING cause the seam?

THE IDEA IN ONE LINE
--------------------
Every atom is handed the same memo -- "the system you belong to is a singlet".
When the molecule splits, both fragments keep reading that memo, and it is now
wrong for both. This script removes the memo and asks whether the seam goes with
it. No retraining: the released weights are used exactly as shipped, and only the
forward pass is altered (that is what "at inference" means).

WHY THIS IS THE DECISIVE EXPERIMENT
-----------------------------------
Every model that shows the seam has a global charge/spin embedding; every model
that does not, does not have one. But those models also differ in TRAINING DATA,
so embedding and data are perfectly confounded across the four available models.
Ablation breaks the confound from the inside: same weights, same data, same
architecture -- only the embedding is disabled.

    seam vanishes  -> the embedding causes it. Causal, on the real model.
    seam survives  -> the embedding is NOT the mechanism. Also a real result,
                      and it sends us back to the training-data hypothesis.

⚠️ THE CONTROL THAT MAKES IT INTERPRETABLE
------------------------------------------
Zeroing an input the network was always trained to receive can push it far out of
distribution, in which case it produces nonsense everywhere and the comparison is
meaningless. So this script ALWAYS reports a CONTROL REGION as well: everything
from 1.2 A up to just short of the step, i.e. all the chemistry the model is
still supposed to get right.

    control preserved + seam gone      -> clean, decisive
    control destroyed                  -> INCONCLUSIVE, model is now OOD
    control preserved + seam unchanged -> embedding is not the cause

(The control used to stop at 3.0 A. That was too narrow -- see step_stats().)

Usage
-----
    python ablate_embedding.py --inspect                  # find the module first
    python ablate_embedding.py --ablate --module <name>   # then ablate
    python ablate_embedding.py --ablate --module <name> --out MLIP_paper/ablation.csv
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from ase import Atoms

sys.path.insert(0, str(Path(__file__).parent / "files"))
from bond_pull_scan import build_nm                      # noqa: E402

# names that plausibly hold the charge/spin embedding
HINTS = ("charge", "spin", "embed", "global", "total", "condition", "context")


def _walk_for_modules(obj, depth=0, seen=None, path="pred", out=None):
    """Find every nn.Module reachable from obj, with its attribute path.

    Walks __dict__ / dicts / lists rather than dir(), because dir() executes
    properties (which can raise) and misses anything held in a container.
    fairchem stores the network inside a predict-unit whose attribute name has
    changed between releases, so search rather than hard-code.
    """
    if out is None:
        out = []
    if seen is None:
        seen = set()
    if id(obj) in seen or depth > 6:
        return out
    seen.add(id(obj))

    if isinstance(obj, torch.nn.Module):
        out.append((path, obj, sum(p.numel() for p in obj.parameters())))
        return out                      # do not descend: named_modules covers it

    if isinstance(obj, dict):
        items = list(obj.items())
    elif isinstance(obj, (list, tuple)):
        items = list(enumerate(obj))
    else:
        items = list(getattr(obj, "__dict__", {}).items())

    for k, v in items:
        if v is None or isinstance(v, (str, int, float, bool, bytes)):
            continue
        _walk_for_modules(v, depth + 1, seen, f"{path}.{k}", out)
    return out


def probe(pred, calc):
    """Print what is actually inside the fairchem objects, so we can stop guessing."""
    print(f"\n{'='*78}\nPROBE -- what is inside the predict unit\n{'='*78}")
    for label, obj in (("predict_unit", pred), ("calculator", calc)):
        print(f"\n{label}: {type(obj).__module__}.{type(obj).__name__}")
        d = getattr(obj, "__dict__", {})
        for k, v in list(d.items())[:40]:
            t = type(v).__name__
            extra = ""
            if isinstance(v, torch.nn.Module):
                extra = f"  <-- nn.Module, {sum(p.numel() for p in v.parameters())/1e6:.1f}M params"
            elif isinstance(v, (list, tuple, dict)):
                extra = f"  (len {len(v)})"
            print(f"    {k:<34}{t:<28}{extra}")

    found = _walk_for_modules(pred, path="pred") + _walk_for_modules(calc, path="calc")
    print(f"\n{'-'*78}\nnn.Modules reachable (largest first):")
    for path, mod, n in sorted(found, key=lambda x: -x[2])[:12]:
        print(f"   {n/1e6:9.2f} M   {path:<44}{type(mod).__name__}")
    if not found:
        print("   NONE FOUND -- paste this output and we will go deeper.")


def load_uma(device="cpu", dtype="float32", ckpt="uma-s-1p1", task="omol"):
    """UMA. Its label embedding feeds BOTH node features and MoLE routing --
    two exits, separately hookable. MACE-OMOL only has the first."""
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    pred = pretrained_mlip.get_predict_unit(ckpt, device=device)
    calc = FAIRChemCalculator(pred, task_name=task)
    found = _walk_for_modules(pred, path="pred") + _walk_for_modules(calc, path="calc")
    if not found:
        probe(pred, calc)
        raise SystemExit("\nno nn.Module found -- paste the PROBE output above.")
    path, mod, n = max(found, key=lambda x: x[2])
    print(f"  using {path}  ({type(mod).__name__}, {n/1e6:.1f} M params)")
    return calc, mod


def load_omol(device="cpu", dtype="float32"):
    from mace.calculators import mace_omol
    calc = mace_omol(model="extra_large", device=device, default_dtype=dtype)
    # the torch module lives on the calculator; name varies by mace version
    for attr in ("models", "model"):
        m = getattr(calc, attr, None)
        if m is None:
            continue
        if isinstance(m, (list, tuple)):
            m = m[0]
        if isinstance(m, torch.nn.Module):
            return calc, m
    raise RuntimeError("could not find the torch module on the calculator; "
                       "inspect `dir(calc)` and adjust load_omol()")


def find_cutoff(obj, calc=None):
    """Hunt for the model's cutoff radius.

    >>> WHY THIS IS NOT JUST getattr(model, 'r_max') (2026-08-05)
    MACE exposes it directly -- `calc.models[0].r_max` -> tensor(6.). fairchem
    does not: the cutoff lives in a config dict or a graph-generation object,
    under one of half a dozen names that have changed between releases
    (cutoff, max_radius, cutoff_radius, radius, max_neighbors...). So search
    both the module tree AND the plain attributes, and print EVERYTHING that
    looks like a radius so we can judge rather than trust one guess.

    This number is the last brick in the ramp mechanism: the transition should
    complete AT the cutoff. MACE-OMOL r_max = 6.0 A and its ramps finish by
    ~5.9 -- match. UMA is the remaining unknown.
    """
    PAT = re.compile(r"cutoff|r_max|rmax|max_radius|radius|max_neigh", re.I)
    hits, seen = [], set()

    def walk(o, path, depth=0):
        if depth > 5 or id(o) in seen:
            return
        seen.add(id(o))
        d = getattr(o, "__dict__", None)
        items = list(o.items()) if isinstance(o, dict) else \
                (list(d.items()) if d else [])
        for k, v in items:
            p = f"{path}.{k}"
            if PAT.search(str(k)):
                if isinstance(v, (int, float)):
                    hits.append((p, float(v)))
                elif hasattr(v, "item"):
                    try:
                        hits.append((p, float(v.item())))
                    except Exception:
                        pass
                elif isinstance(v, (str, bool)) or v is None:
                    hits.append((p, v))
            if isinstance(v, (dict,)) or hasattr(v, "__dict__"):
                if not isinstance(v, (str, int, float, bool, bytes)):
                    walk(v, p, depth + 1)

    for name, o in (("model", obj), ("calc", calc)):
        if o is not None:
            walk(o, name)
    # torch modules often carry it as a buffer or plain attribute too
    if isinstance(obj, torch.nn.Module):
        for n, m in obj.named_modules():
            for attr in ("cutoff", "r_max", "max_radius", "cutoff_radius"):
                v = getattr(m, attr, None)
                if isinstance(v, (int, float)):
                    hits.append((f"{n}.{attr}" if n else attr, float(v)))
                elif hasattr(v, "item"):
                    try:
                        hits.append((f"{n}.{attr}" if n else attr, float(v.item())))
                    except Exception:
                        pass

    print(f"\n{'='*78}\nCUTOFF HUNT -- anything that looks like a radius\n{'='*78}")
    if not hits:
        print("  nothing matched. Try:  print(pred.model.module.backbone.__dict__.keys())")
        return
    uniq = sorted(set(hits), key=lambda x: str(x[0]))
    for p, v in uniq:
        star = "   <<< plausible cutoff" if isinstance(v, float) and 3.0 <= v <= 12.0 else ""
        print(f"   {p:<58}{v}{star}")
    rad = sorted({v for _, v in uniq if isinstance(v, float) and 3.0 <= v <= 12.0})
    if rad:
        print(f"\n  candidate radii: {rad}")
        print(f"  >>> compare against the measured ramp completion (r90):")
        print(f"      MACE-OMOL  ramps finish ~5.78-5.89 A,  r_max = 6.0  -> MATCH")
        print(f"      UMA        ramps finish ~5.60-5.89 A,  r_max = ?    -> check above")


def inspect(model):
    """Print every submodule whose name looks like a charge/spin embedding."""
    print(f"\n{'='*78}\nMODULE TREE -- candidates for the charge/spin embedding\n{'='*78}")
    hits = []
    for name, mod in model.named_modules():
        if not name:
            continue
        low = name.lower()
        if any(h in low for h in HINTS):
            nparam = sum(p.numel() for p in mod.parameters(recurse=False))
            shapes = [tuple(p.shape) for p in mod.parameters(recurse=False)]
            hits.append((name, type(mod).__name__, nparam, shapes))

    if hits:
        print(f"\n{'module':<52}{'type':<20}{'params':>9}")
        print("-" * 78)
        for name, kind, n, shapes in hits:
            print(f"{name:<52}{kind:<20}{n:>9}")
            for s in shapes:
                print(f"{'':<52}{'':<20}   {s}")
    else:
        print("\n  no name matched the hints. Full top-level tree instead:\n")
        for name, mod in model.named_modules():
            if name.count(".") <= 1 and name:
                print(f"  {name:<52}{type(mod).__name__}")

    print(f"\n{'='*78}\nALL nn.Embedding / small Linear layers (the usual shape of a\n"
          f"lookup-table embedding: rows = number of distinct label values)\n{'='*78}")
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Embedding):
            print(f"  EMBEDDING  {name:<46}{tuple(mod.weight.shape)}")
        elif isinstance(mod, torch.nn.Linear) and mod.in_features <= 64:
            print(f"  Linear     {name:<46}"
                  f"in={mod.in_features} out={mod.out_features}")
    print("\npick the module whose row count matches the number of charge or spin "
          "values\nOMol25 supports, then rerun with:  --ablate --module <name>\n")


def ablate(model, pattern, scale=0.0):
    """Scale the matching module(s) OUTPUT by `scale`. Returns hook handles.

    scale=0.0 removes the contribution entirely; scale=1.0 is a no-op. Anything
    in between is a dose-response probe.

    >>> WHY SCALING RATHER THAN ZEROING (2026-08-04)
    Zeroing the joint embedding sent MACE-OMOL to ~1e20 eV -- the network leans
    on that input everywhere, not just at the seam, so a hard switch-off is far
    outside anything it ever saw. Sweeping lambda lets us find the largest
    perturbation that still leaves the bonded region intact, and read the seam
    trend inside that safe range.

    >>> ScriptModules CANNOT be hooked (RuntimeError). Parts of MACE are
    TorchScript-compiled; we skip them and hook the eager parent instead.
    """
    handles, matched, skipped = [], [], []
    rx = re.compile(pattern)
    for name, mod in model.named_modules():
        if not name or not rx.search(name):
            continue
        if isinstance(mod, torch.jit.ScriptModule) or \
           type(mod).__name__ == "RecursiveScriptModule":
            skipped.append(name)
            continue
        handles.append(mod.register_forward_hook(
            lambda m, i, o, _s=scale: (o * _s) if torch.is_tensor(o) else o))
        matched.append(name)
    if not matched:
        raise SystemExit(f"no hookable module matched {pattern!r} "
                         f"(skipped scripted: {skipped})")
    tag = "zeroed" if scale == 0 else f"scaled x{scale:g}"
    print(f"  {tag}: {', '.join(matched)}"
          + (f"   [skipped scripted: {', '.join(skipped)}]" if skipped else ""))
    return handles, matched


def ablate_norm_preserved(model, pattern):
    """Replace the Embedding output with a NORM-PRESERVED label-free vector.

    >>> WHY THE PREVIOUS TWO DESIGNS BOTH FAILED ON MACE-OMOL (2026-08-06)

    1. SCALING toward zero shrinks the vector's magnitude. MACE-OMOL leans on
       this input everywhere, so below lambda ~0.75 it leaves the distribution
       entirely and returns ~1e20 eV. Only two usable rungs.

    2. MEAN ROW was meant to fix that -- "typical magnitude, no information" --
       but it does not. The mean of ~101 roughly-random 128- or 1024-dim vectors
       has norm about ||row||/sqrt(101), i.e. a TENTH of a real row. So the mean
       row is very nearly zeroing in disguise, and it blew up for the same reason.

    THE FIX is to separate the two things we were conflating: the DIRECTION of
    the embedding carries the label, the MAGNITUDE is what keeps the network in
    distribution. So take the mean direction and rescale it to the typical row
    norm:

        v = mean(rows);   v <- v * (median ||row|| / ||v||)

    Every label value now maps to the same vector, so no label information
    survives -- but that vector has the magnitude the network expects, so the
    downstream layers are never asked to process something they have never seen
    at that scale. In-distribution by construction, label-free by construction.

    This is the ablation the MACE-OMOL comparison needed and never got.
    """
    handles, matched = [], []
    rx = re.compile(pattern)
    for name, mod in model.named_modules():
        if not name or not rx.search(name):
            continue
        if not isinstance(mod, torch.nn.Embedding):
            continue
        W = mod.weight.detach()
        v = W.mean(dim=0, keepdim=True)
        typical = W.norm(dim=1).median()
        v = v * (typical / v.norm().clamp_min(1e-12))
        print(f"    {name}: mean-row norm {W.mean(0).norm():.4f} -> rescaled to "
              f"{typical:.4f} (median row norm)")
        handles.append(mod.register_forward_hook(
            lambda m, i, o, _v=v: _v.to(o.dtype).to(o.device)
                                    .expand_as(o).clone()))
        matched.append(name)
    if not matched:
        raise SystemExit(f"no nn.Embedding matched {pattern!r}")
    print(f"  label-blinded, norm preserved: {', '.join(matched)}")
    return handles, matched


def ablate_to_mean(model, pattern):
    """Replace the matching Embedding's OUTPUT with the MEAN over all its rows.

    >>> WHY THIS IS THE RIGHT LIMIT (2026-08-04)
    Scaling toward zero leaves the distribution -- MACE-OMOL explodes to ~1e20 eV
    below lambda ~0.5, so we can never reach a clean knockout that way. The mean
    row has TYPICAL MAGNITUDE (it is an average of vectors the model was trained
    on) but carries NO LABEL INFORMATION: every charge/spin value maps to the
    same vector. That is the ablation we actually want -- the memo is still
    handed out, it just no longer says anything.
    """
    handles, matched = [], []
    rx = re.compile(pattern)
    for name, mod in model.named_modules():
        if not name or not rx.search(name):
            continue
        if not isinstance(mod, torch.nn.Embedding):
            continue
        mean_row = mod.weight.detach().mean(dim=0, keepdim=True)   # (1, dim)
        handles.append(mod.register_forward_hook(
            lambda m, i, o, _mr=mean_row: _mr.to(o.dtype).to(o.device)
                                            .expand_as(o).clone()))
        matched.append(name)
    if not matched:
        raise SystemExit(f"no nn.Embedding matched {pattern!r}")
    print(f"  label-blinded (mean row): {', '.join(matched)}")
    return handles, matched


SHAM_EXCLUDE = re.compile(
    r"charge|spin|embed|global|total|condition|context|dataset|composition|routing")


def sham_candidates(model, k=3, seed=0, nparam_ref=None):
    """Pick modules that are NOT global-conditioning channels, size-matched.

    >>> WHY THIS CONTROL EXISTS (2026-08-05)
    The UMA ladder removed the seam by scaling `composition_embedding` by 0.9 --
    a 10% perturbation. A referee will immediately ask the obvious question:

        "Is the seam just a fragile feature that ANY small perturbation smears
         out? Did you try nudging something irrelevant by the same amount?"

    Without an answer, the whole ablation argument is dead. So: take modules of
    comparable parameter count that are GEOMETRY-DEPENDENT (interior Linears in
    the message-passing stack) rather than global constants, scale them by the
    same lambdas, and see whether the seam survives.

        sham keeps the seam   -> the effect is SPECIFIC to global channels. The
                                 causal claim stands.
        sham also kills it    -> we are measuring generic fragility, not a
                                 mechanism. The claim collapses, and we would
                                 need to know that BEFORE Sterling sees it.

    Selection is seeded and printed, so it is reproducible and auditable -- a
    referee can check we did not shop for a sham that happened to behave.
    """
    pool = []
    for name, mod in model.named_modules():
        if not name or SHAM_EXCLUDE.search(name.lower()):
            continue
        if not isinstance(mod, (torch.nn.Linear, torch.nn.Embedding)):
            continue
        if isinstance(mod, torch.jit.ScriptModule) or \
           type(mod).__name__ == "RecursiveScriptModule":
            continue
        n = sum(p.numel() for p in mod.parameters(recurse=False))
        if n:
            pool.append((name, n))
    if not pool:
        return []
    if nparam_ref:            # keep the closest in size, then sample from those
        pool.sort(key=lambda x: abs(np.log(x[1]) - np.log(max(nparam_ref, 1))))
        pool = pool[:max(4 * k, 12)]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
    return [pool[i] for i in sorted(idx)]


CONTROL_LO, CONTROL_PAD, STEP_FROM = 1.2, 0.4, 4.0


def step_stats(r, E):
    """The step, its location, and where the control window should end.

    >>> BOTH OF THESE WERE WRONG UNTIL 2026-08-05, and both were caught by
    plotting the curves rather than by any summary number.

    1. seam was  E(plateau) - E(5.0 A),  which assumes the curve has flattened
       by 5.0 A. True for nitromethane C-N; FALSE for ethyl hydroperoxide O-O,
       whose curve peaks at 4.7 A and descends -- the formula returned -30.4
       kcal/mol for a curve with no step at all.
    2. the control window was a fixed 1.2-3.0 A. Too narrow: ablation artifacts
       between 3 A and the step fell outside it and scored as zero damage.
       On O-O that hid 27-70 kcal/mol trenches.

    rise is zero when the curve never climbs after flattening, which is the
    correct answer for a bond that has no seam.
    """
    r, E = np.asarray(r), np.asarray(E)
    far = r >= STEP_FROM
    rise = float(E[-1] - np.nanmin(E[far]))
    j = np.diff(E)
    sel = np.where(r[:-1] >= STEP_FROM)[0]
    k = sel[int(np.nanargmax(np.abs(j[sel])))]
    step_r = float(r[k + 1])
    return rise, step_r, float(j[k]), (r >= CONTROL_LO) & (r <= step_r - CONTROL_PAD)


def scan(calc, rs, charge=0, spin=1):
    base = build_nm()
    pos = base.get_positions()
    u = np.array([0.0, 0.0, 1.0])
    no2 = [1, 2, 3]
    E = np.full(len(rs), np.nan)
    for i, r in enumerate(rs):
        a = Atoms(symbols=base.get_chemical_symbols(), positions=pos.copy())
        shift = (a.positions[0] + r * u) - a.positions[1]
        for k in no2:
            a.positions[k] += shift
        a.info = {"charge": charge, "spin": spin}
        a.calc = calc
        try:
            E[i] = a.get_potential_energy()
        except Exception as e:
            print(f"    r={r:.2f} failed: {type(e).__name__}")
    return E


def report(rs, E0, E1, out=None):
    """E0 = intact model, E1 = ablated. Both relative to their own minimum."""
    r = np.asarray(rs)
    a = E0 - np.nanmin(E0)
    b = E1 - np.nanmin(E1)
    at = lambda y, q: float(np.interp(q, r, y))

    # step = energy appearing AFTER the curve should already have flattened
    seam0, step_r0, _sz, m = step_stats(r, a)
    seam1, _sr1, _sz1, _ = step_stats(r, b)
    bonded_rms = float(np.sqrt(np.nanmean((a[m] - b[m]) ** 2)))
    bonded_max = float(np.nanmax(np.abs(a[m] - b[m])))

    print(f"\n{'='*78}\nRESULT\n{'='*78}")
    print(f"{'':<26}{'intact':>12}{'ablated':>12}{'change':>12}")
    print("-" * 78)
    print(f"{'seam (5.0 A -> plateau)':<26}{seam0:>12.3f}{seam1:>12.3f}"
          f"{seam1 - seam0:>+12.3f}   eV")
    print(f"{'plateau':<26}{a[-1]:>12.3f}{b[-1]:>12.3f}{b[-1]-a[-1]:>+12.3f}   eV")
    print(f"{'E(5.0 A)':<26}{at(a,5.0):>12.3f}{at(b,5.0):>12.3f}"
          f"{at(b,5.0)-at(a,5.0):>+12.3f}   eV")
    print(f"\nCONTROL -- everything before the step "
          f"({CONTROL_LO:.1f}-{step_r0 - CONTROL_PAD:.1f} A), intact vs ablated:")
    print(f"   RMS difference {bonded_rms:8.3f} eV      max {bonded_max:8.3f} eV")

    print(f"\n{'-'*78}\nINTERPRETATION")
    if bonded_max > 1.0:
        print("  ⚠️  INCONCLUSIVE. Ablation destroyed the bonded region too, so the")
        print("      model is now far out of distribution and the seam comparison")
        print("      means nothing. Try ablating a narrower module, or freeze the")
        print("      embedding to a constant instead of zeroing it.")
    elif abs(seam1) < 0.25 * abs(seam0):
        print("  ✅ SEAM REMOVED while the bonded region survived.")
        print("      The global charge/spin embedding CAUSES the seam. Causal,")
        print("      on the released model, no retraining.")
    elif abs(seam1 - seam0) < 0.1 * abs(seam0):
        print("  ❌ SEAM UNCHANGED while the bonded region survived.")
        print("      The embedding is NOT the mechanism. Points back at the")
        print("      training data (no disconnected-fragment configurations).")
    else:
        print(f"  ~ PARTIAL: seam went {seam0:+.3f} -> {seam1:+.3f} eV.")
        print("      Suggests the embedding contributes but is not the whole story.")

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            f.write("bond_A,E_intact_eV,E_ablated_eV,E_intact_rel,E_ablated_rel\n")
            for i, rr in enumerate(r):
                f.write(f"{rr:.3f},{E0[i]:.6f},{E1[i]:.6f},{a[i]:.6f},{b[i]:.6f}\n")
        print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inspect", action="store_true",
                    help="list candidate embedding modules and exit")
    ap.add_argument("--cutoff", action="store_true",
                    help="hunt for the model's cutoff radius and exit. This is the "
                         "last brick in the ramp mechanism: the transition should "
                         "complete AT the cutoff.")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--module", default=None,
                    help="regex matching the module(s) to zero, from --inspect")
    ap.add_argument("--model", default="omol", choices=["omol", "uma"],
                    help="omol -> run in the `mace` env; uma -> run in the `uma` env")
    ap.add_argument("--uma-ckpt", default="uma-s-1p1")
    ap.add_argument("--probe", action="store_true",
                    help="dump the fairchem object structure and exit")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--r-min", type=float, default=1.2)
    ap.add_argument("--r-max", type=float, default=8.0)
    ap.add_argument("--dr", type=float, default=0.05)
    ap.add_argument("--ladder", action="store_true",
                    help="run the full ablation ladder in one go (recommended)")
    ap.add_argument("--lambdas", default="0.9,0.75,0.5,0.25,0.0",
                    help="comma list of lambda values for the ladder. MACE-OMOL "
                         "falls off a cliff below ~0.75, so for that model use a "
                         "FINE grid inside the safe window, e.g. "
                         "--lambdas 0.98,0.95,0.92,0.9,0.87,0.85,0.82,0.8,0.77,0.75")
    ap.add_argument("--sham", type=int, default=3, metavar="K",
                    help="number of SHAM targets in the ladder (default 3, 0 to "
                         "disable). Size-matched, geometry-dependent modules that "
                         "are NOT global-conditioning channels. If the seam "
                         "survives these at the same lambda, the effect is "
                         "specific; if it does not, we are measuring fragility.")
    ap.add_argument("--sham-seed", type=int, default=0,
                    help="seed for sham selection -- fixed so it is reproducible "
                         "and cannot be shopped")
    # >>> FIXED 2026-08-05: the default was hardcoded to *_omol.csv regardless of
    # --model, so the UMA run silently OVERWROTE the MACE-OMOL mean-row CSVs.
    ap.add_argument("--out", default=None,
                    help="default: MLIP_paper/ablation_<model>.csv")
    args = ap.parse_args()
    if args.out is None:
        args.out = f"MLIP_paper/ablation_{args.model}.csv"

    if args.model == "uma":
        print(f"loading UMA ({args.uma_ckpt}, task=omol) ...")
        if args.probe:
            from fairchem.core import pretrained_mlip, FAIRChemCalculator
            _p = pretrained_mlip.get_predict_unit(args.uma_ckpt, device=args.device)
            probe(_p, FAIRChemCalculator(_p, task_name="omol"))
            return
        calc, model = load_uma(args.device, args.dtype, args.uma_ckpt)
    else:
        print("loading MACE-OMOL (extra_large) ...")
        calc, model = load_omol(args.device, args.dtype)
    print(f"  module: {type(model).__name__}, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f} M params")

    if args.cutoff:
        find_cutoff(model, calc)
        return

    if args.inspect or not args.ablate:
        inspect(model)
        return

    # >>> FIXED: --ladder supplies its own module patterns, so it must not be
    # blocked by the --module guard.
    if not args.module and not args.ladder:
        raise SystemExit("--ablate needs --module <regex>, or use --ladder. "
                         "Run --inspect first.")

    rs = np.arange(args.r_min, args.r_max + args.dr / 2, args.dr)

    print(f"\nscanning INTACT model ({len(rs)} points) ...")
    E0 = scan(calc, rs)

    if args.ladder:
        # Dose-response. For each target, sweep lambda from intact to removed and
        # report where the bonded region breaks. The seam trend INSIDE the safe
        # range is the interpretable signal.
        if args.model == "uma":
            # Module names confirmed from --inspect on uma-s-1p1. Note the
            # "module." prefix: the network is wrapped in an AveragedModel
            # (EMA/SWA container), so the real net lives at .module.
            #
            # UMA exposes THREE routes the MACE-OMOL tree does not:
            #   routing_mlp          -> MoLE expert weights (32 experts).
            #                           A second exit for the label, absent in MACE.
            #   composition_embedding-> element composition feeding the router.
            #                           This is the mechanism blamed for
            #                           non-separability in Finding 11 -- now
            #                           testable directly rather than inferred.
            #   dataset_embedding    -> the task vector (one per dataset).
            TARGETS = [
                ("spin embedding",   r"backbone\.spin_embedding$"),
                ("charge+spin",      r"backbone\.(spin|charge)_embedding$"),
                ("MoLE routing",     r"backbone\.routing_mlp$"),
                ("composition emb",  r"backbone\.composition_embedding$"),
                ("task/dataset emb", r"backbone\.dataset_embedding$"),
            ]
        else:
            TARGETS = [
                ("spin embedding",   r"joint_embedding\.embedders\.total_spin$"),
                ("both lookups",     r"joint_embedding\.embedders\.total_(spin|charge)$"),
                ("joint projection", r"joint_embedding\.project$"),
                ("direct E readout", r"^embedding_readout$"),
            ]
        # ---- SHAM CONTROLS ---------------------------------------------------
        # Appended to the same ladder, run under identical conditions, so the
        # comparison is like-for-like. Size-matched to the smallest real target
        # so nobody can argue we perturbed something trivially small.
        if args.sham:
            ref_n = min((sum(p.numel() for p in m.parameters(recurse=False))
                         for lab, rx in TARGETS
                         for n_, m in model.named_modules()
                         if n_ and re.search(rx, n_)
                         and sum(p.numel() for p in m.parameters(recurse=False))),
                        default=None)
            shams = sham_candidates(model, k=args.sham, seed=args.sham_seed,
                                    nparam_ref=ref_n)
            if shams:
                print(f"\nSHAM CONTROLS (seed {args.sham_seed}, size-matched to "
                      f"{ref_n} params):")
                for nm, n in shams:
                    print(f"   {nm:<58}{n:>9} params")
                TARGETS = TARGETS + [(f"SHAM: {nm}", f"^{re.escape(nm)}$")
                                     for nm, _ in shams]
            else:
                print("\n⚠️  no sham candidates found -- specificity is UNTESTED.")

        LAMBDAS = [float(x) for x in args.lambdas.split(",")]
        r = np.asarray(rs)
        a = E0 - np.nanmin(E0)
        at = lambda y, q: float(np.interp(q, r, y))
        seam0, step_r0, step_sz0, bm = step_stats(r, a)
        print(f"\nintact step {seam0:.4f} eV at {step_r0:.2f} A "
              f"(jump {step_sz0:+.4f});  control window "
              f"{CONTROL_LO:.1f}-{step_r0 - CONTROL_PAD:.1f} A, {bm.sum()} points")

        # >>> FIXED 2026-08-05: the ladder printed ~30 rungs x 69 points and saved
        # NONE of it. The scalar seam is not enough -- a seam can read zero
        # because the step vanished OR because a compensating error appeared, and
        # only the curve tells them apart. Every rung is now logged long-format.
        rung_rows = [f"{rr:.3f},intact,1.000,{E0[i]:.6f},{a[i]:.6f}"
                     for i, rr in enumerate(r)]
        summary_rows = []

        for label, rx in TARGETS:
            print(f"\n\n{'#'*78}\n# TARGET: {label}   ({rx})\n{'#'*78}")
            print(f"{'lambda':>8}{'seam (eV)':>12}{'vs intact':>12}"
                  f"{'bonded max':>13}{'dispro':>10}   verdict")
            print("-" * 88)
            print(f"{1.0:>8.2f}{seam0:>12.3f}{0.0:>12.3f}{0.0:>13.3f}"
                  f"{'-':>10}   intact")
            for lam in LAMBDAS:
                try:
                    handles, _ = ablate(model, rx, scale=lam)
                except SystemExit as e:
                    print(f"  {e}"); break
                E1 = scan(calc, rs)
                for h in handles:
                    h.remove()
                b = E1 - np.nanmin(E1)
                seam1, _sr, _sz, _ = step_stats(r, b)
                # damage is judged on the INTACT model's control window, so every
                # rung is scored against the same region
                bmax = float(np.nanmax(np.abs(a[bm] - b[bm])))
                if not np.isfinite(bmax) or bmax > 1.0:
                    verdict = "BROKEN (OOD)"
                elif abs(seam1) < 0.25 * abs(seam0):
                    # >>> no comma: this string is written into a CSV field
                    verdict = "*** seam removed - model still sane ***"
                elif abs(seam1 - seam0) < 0.1 * abs(seam0):
                    verdict = "seam unchanged"
                else:
                    verdict = "seam partially reduced"
                # disproportion = seam removed per eV of collateral damage.
                # This is the number that makes the result defensible: a large
                # ratio means the ablation hit the seam and left the chemistry
                # alone. Reported explicitly so it is not eyeballed off two
                # columns.
                dis = abs(seam1 - seam0) / bmax if bmax > 1e-9 else np.inf
                print(f"{lam:>8.2f}{seam1:>12.3f}{seam1-seam0:>+12.3f}"
                      f"{bmax:>13.3f}{dis:>10.1f}   {verdict}")
                summary_rows.append(
                    f"{label},{rx},{lam:g},{seam1:.4f},{seam1-seam0:.4f},"
                    f"{bmax:.4f},{dis:.1f},{verdict}")
                rung_rows += [f"{rr:.3f},{label},{lam:g},{E1[i]:.6f},{b[i]:.6f}"
                              for i, rr in enumerate(r)]
        # ---- cleanest rung: label-blinding via the mean row -------------------
        print(f"\n\n{'#'*78}\n# TARGET: NORM-PRESERVED LABEL BLINDING (the real knockout)\n"
              f"# every charge/spin value maps to the SAME vector, rescaled to the\n"
              f"# median row norm. Same magnitude the network expects, zero label\n"
              f"# information. In-distribution by construction.\n{'#'*78}")
        MEANROW = ([("spin only",  r"backbone\.spin_embedding\.rand_emb$"),
                    ("spin+charge", r"backbone\.(spin|charge)_embedding\.rand_emb$")]
                   if args.model == "uma" else
                   [("spin only",  r"joint_embedding\.embedders\.total_spin$"),
                    ("spin+charge", r"joint_embedding\.embedders\.total_(spin|charge)$")])
        for label, rx in MEANROW:
            print(f"\n--- {label} ---")
            try:
                # norm-preserved: the mean row alone is ~||row||/sqrt(N), which is
                # near-zeroing in disguise and is what blew MACE-OMOL up before.
                handles, _ = ablate_norm_preserved(model, rx)
            except SystemExit as e:
                print(f"  {e}"); continue
            E1 = scan(calc, rs)
            for h in handles:
                h.remove()
            report(rs, E0, E1,
                   args.out.replace(".csv", f"_meanrow_{label.replace(' ','_').replace('+','_')}.csv"))

        # ---- persist the ladder ----------------------------------------------
        stem = args.out.replace(".csv", "")
        Path(stem).parent.mkdir(parents=True, exist_ok=True)
        with open(f"{stem}_ladder_curves.csv", "w") as f:
            f.write("bond_A,target,lambda,E_eV,E_rel_eV\n")
            f.write("\n".join(rung_rows) + "\n")
        with open(f"{stem}_ladder_summary.csv", "w") as f:
            f.write("target,regex,lambda,seam_eV,seam_change_eV,"
                    "bonded_max_eV,disproportion,verdict\n")
            f.write("\n".join(summary_rows) + "\n")
        print(f"\nwrote {stem}_ladder_curves.csv   ({len(rung_rows)} rows)")
        print(f"wrote {stem}_ladder_summary.csv  ({len(summary_rows)} rows)")

        # ---- did the specificity control pass? -------------------------------
        # >>> FIXED 2026-08-05. The first version compared the best REAL rung to
        # the best SHAM rung ANYWHERE on the ladder -- i.e. at different doses.
        # That is not a control, it is a comparison of two different experiments,
        # and it badly understates specificity: the shams only do anything at
        # lambda=0, where every module damages the model. Two matched comparisons
        # replace it.
        if args.sham and summary_rows:
            rows = [s.split(",") for s in summary_rows]
            real = [x for x in rows if not x[0].startswith("SHAM")]
            sham = [x for x in rows if x[0].startswith("SHAM")]
            SAFE = lambda x: float(x[5]) < 1.0            # bonded region intact
            dseam, bonded, lam = (lambda x: abs(float(x[4])),
                                  lambda x: float(x[5]), lambda x: float(x[2]))

            print(f"\n{'='*78}\nSPECIFICITY CONTROL\n{'='*78}")

            # --- (1) MATCHED DOSE: same lambda, real vs the WORST-CASE sham ----
            print("\n(1) MATCHED DOSE -- same lambda, best real target vs the most"
                  "\n    effective sham (worst case for us, so the ratio is a floor)\n")
            print(f"{'lambda':>8}{'best real':>28}{'dseam':>9}"
                  f"{'max sham dseam':>16}{'ratio':>9}")
            print("-" * 78)
            ratios = []
            for L in LAMBDAS:
                R = [x for x in real if lam(x) == L and SAFE(x)]
                S = [x for x in sham if lam(x) == L and SAFE(x)]
                if not R or not S:
                    continue
                b = max(R, key=dseam)
                s = max(S, key=dseam)
                ratio = dseam(b) / dseam(s) if dseam(s) > 1e-9 else np.inf
                ratios.append((L, ratio))
                print(f"{L:>8.2f}{b[0][:26]:>28}{dseam(b):>9.3f}"
                      f"{dseam(s):>16.3f}{ratio:>9.1f}x")

            # --- (2) MATCHED COLLATERAL -- the dose-independent statement ------
            # lambda means different things for different modules, so the fairest
            # comparison is at EQUAL DAMAGE to the bonded region: for the best
            # real rung, find the sham rung with the closest bonded_max and ask
            # how much seam each removed for the same price.
            print("\n(2) MATCHED COLLATERAL -- equal damage to the bonded region."
                  "\n    Dose-independent, and the number to quote.\n")
            R = [x for x in real if SAFE(x)]
            if R and sham:
                b = max(R, key=dseam)
                s = min(sham, key=lambda x: abs(bonded(x) - bonded(b)))
                ratio2 = dseam(b) / dseam(s) if dseam(s) > 1e-9 else np.inf
                print(f"    REAL  {b[0][:38]:<40} lam={lam(b):<5g} "
                      f"bonded={bonded(b):.3f} eV  ->  seam {dseam(b):.3f} eV")
                print(f"    SHAM  {s[0][:38]:<40} lam={lam(s):<5g} "
                      f"bonded={bonded(s):.3f} eV  ->  seam {dseam(s):.3f} eV")
                print(f"\n    At the same collateral cost, the global channel removes"
                      f" {ratio2:.0f}x more seam.")
            else:
                ratio2 = 0.0

            best_ratio = max([r for _, r in ratios] + [ratio2], default=0.0)
            print()
            if best_ratio > 10:
                print("  ✅ SPECIFIC. At matched dose and matched collateral, the global")
                print("     channels are an order of magnitude or more ahead of interior")
                print("     weights. This is a mechanism, not generic fragility.")
            elif best_ratio > 3:
                print("  ~ PARTIAL. Real targets clearly lead, but shams are not inert.")
                print("     Quote the MATCHED comparisons, never the raw lambda=0 rung.")
            else:
                print("  ❌ NOT SPECIFIC. Interior weights remove the seam about as well.")
                print("     Do not present the ablation as causal until this is resolved.")
            print("\n  NOTE: shams biting at lambda->0 is EXPECTED and not a failure --")
            print("  destroying any layer degrades the whole model. Specificity lives at")
            print("  SMALL doses, which is exactly where the ratio is largest.")

        print("\n\nread the LARGEST lambda-reduction whose bonded max is still < 1 eV;"
              "\nthe seam trend across those rows is the interpretable result."
              "\nthe MEAN-ROW rung is the cleanest test: in-distribution magnitude,"
              "\nzero label information."
              "\n>>> and read the SPECIFICITY block: a real target that beats every"
              "\n    sham is the difference between a mechanism and an artifact.")
        return

    print(f"\nablating ...")
    handles, _ = ablate(model, args.module)
    print(f"scanning ABLATED model ...")
    E1 = scan(calc, rs)
    for h in handles:
        h.remove()

    report(rs, E0, E1, args.out)


if __name__ == "__main__":
    main()
