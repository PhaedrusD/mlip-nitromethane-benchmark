"""
ani_1xnr_Mrr -- fine-tuned ANI-1xnr ensemble ("M" = Morse, "rr" = radical-radical)

    import sys; sys.path.insert(0, "/path/to/ANI-1xnr-Mrr")   # <-- needed
    from ani_1xnr_Mrr import get_calculator
    calc = get_calculator(info="ani-1xnr/model/ani-1xnr.info")
    atoms.calc = calc

This module sits beside the member*.pt checkpoints and is not installed, so it is
only importable if that folder is on sys.path (or is the cwd). Once imported it
finds its own checkpoints via __file__, so the cwd is irrelevant after that.

HOW IT LOADS (this matters for speed)
-------------------------------------
The eight fine-tuned state dicts are GRAFTED into one torchani model whose
`neural_networks` is a genuine 8-member Ensemble, by rewriting each checkpoint
key from

    potentials.nnp.neural_networks.<...>
to
    potentials.nnp.neural_networks.members.<i>.<...>

The point is that the AEV computation -- which dominates the cost -- is then done
ONCE and shared by all eight networks, instead of eight times. Measured on a
243-atom box, single thread:

    one member alone      47.7 ms/step
    this Ensemble         75.0 ms/step   (1.57x one member)
    eight separate calcs  401.5 ms/step  (8.43x one member)

So the full ensemble costs only ~57% more than a single network. There is no
reason to drop members for speed, and MD with this model is practical.

UNCERTAINTY: NOT AVAILABLE FROM THE FAST PATH, AND NOT MEANINGFUL ANYWAY
-----------------------------------------------------------------------
The Ensemble averages internally, so a single forward pass does not expose
per-member energies. `get_member_calculators()` below returns the eight networks
separately if you want the spread -- but read this first:

all eight members were fine-tuned on IDENTICAL data, so their agreement is
INDUCED, not informative. Measured: the N2 BDE error is unchanged by fine-tuning
(-6.9 -> -6.9 kcal/mol) while the spread collapsed from +/-55.7 to +/-1.4. The
model is now confidently wrong on the bonds it was already wrong on. Do not use
the spread as an error bar for this model.
"""
import os, glob

_HERE = os.path.dirname(os.path.abspath(__file__))
_PREFIX = "potentials.nnp.neural_networks."


def _checkpoints(ckpt_dir=None):
    ckpt_dir = ckpt_dir or _HERE
    cks = glob.glob(os.path.join(ckpt_dir, "member*.pt"))
    if not cks:
        raise FileNotFoundError(f"no member*.pt in {ckpt_dir}")
    return sorted(cks, key=lambda p: int(os.path.basename(p)[6:].split(".")[0]))


def get_calculator(info, ckpt_dir=None, device="cpu"):
    """ASE calculator for the fine-tuned 8-member ensemble (fast, shared AEV).

    `info` must point at the ORIGINAL ANI-1xnr release's .info file -- it supplies
    the architecture, AEV parameters and self-atomic energies. The .pt files here
    are state dicts only and are useless without it.
    """
    import torch, torchani.neurochem as nc
    cks = _checkpoints(ckpt_dir)
    model = nc.load_model_from_info_file(info)
    sd = model.state_dict()
    grafted = 0
    for c in cks:
        i = int(os.path.basename(c)[6:].split(".")[0])
        for k, v in torch.load(c, map_location="cpu").items():
            if k.startswith(_PREFIX):
                sd[_PREFIX + f"members.{i}." + k[len(_PREFIX):]] = v
                grafted += 1
    if grafted == 0:
        raise RuntimeError(
            "grafted 0 tensors -- the torchani state_dict layout is not what this "
            "loader expects (built against torchani 2.8.4). Check that keys start "
            f"with {_PREFIX!r}.")
    model.load_state_dict(sd)
    if len(cks) != 8:
        print(f"ani_1xnr_Mrr: WARNING -- only {len(cks)} of 8 members present; "
              f"this is not ANI-1xnr and every prediction will differ.")
    print(f"ani_1xnr_Mrr: {len(cks)} members grafted ({grafted} tensors), shared AEV")
    calc = model.to(torch.device(device)).eval().ase()
    calc.n_members = len(cks)
    return calc


def get_member_calculators(info, ckpt_dir=None, device="cpu"):
    """The eight members as separate ASE calculators, for spread analysis ONLY.

    ~5x slower in aggregate than get_calculator(). See the module docstring on why
    the spread is not a valid error bar for this model.
    """
    import torch, torchani.neurochem as nc
    out = []
    for c in _checkpoints(ckpt_dir):
        i = int(os.path.basename(c)[6:].split(".")[0])
        m = nc.load_model_from_info_file(info, model_index=i)
        m.load_state_dict(torch.load(c, map_location="cpu"))
        out.append(m.to(torch.device(device)).eval().ase())
    return out


def energy_spread(atoms, info, ckpt_dir=None, device="cpu"):
    """(mean_eV, std_eV) across members. Slow; diagnostic use only."""
    import numpy as np
    es = []
    for c in get_member_calculators(info, ckpt_dir, device):
        a = atoms.copy(); a.calc = c
        es.append(a.get_potential_energy())
    return float(np.mean(es)), float(np.std(es))
