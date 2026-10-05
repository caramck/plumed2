#!/usr/bin/env python3
"""Export a trained torch_geometric ViSNet model to TorchScript for the PLUMED VISNET action.

The trained weights alone (``model.state_dict()``) are not enough to rebuild the model:
the constructor hyperparameters and the target normalization used during training are
also needed. These are provided through a "model card", a JSON dictionary such as:

    {
      "format": "plumed-visnet/1",
      "hparams": {"lmax": 1, "num_heads": 8, "num_layers": 4, "hidden_channels": 64,
                  "num_rbf": 32, "max_z": 100, "cutoff": 5.0, ...},
      "target": {"name": "gap", "units": "eV", "norm_mean": 6.85, "norm_std": 1.28,
                 "level_of_theory": "G2W2"},
      "length_unit": "angstrom",
      "provenance": {"weights_file": "gap_finetuned.pt", "finetuned_from": "gap_qm9.pt",
                     "training_data": "adenine_1000.csv", "notes": "..."}
    }

target.name is the property PLUMED must be told with VISNET PROPERTY=...; the optional
level_of_theory and provenance entries are written to the PLUMED log.

Usage:

    python export_visnet.py --weights model.pt --card model.json --example frame.xyz -o model_plumed.pt
    python export_visnet.py --bundle bundle.pt --example frame.xyz -o model_plumed.pt

where ``bundle.pt`` was saved with ``torch.save({"card": card, "state_dict": model.state_dict()}, ...)``.

The exported module has the signature ``forward(z: int64[N], pos: float[N,3]) -> float[1]``,
returns the de-normalized property and stores the card in the extra file ``card.json``.
PyG builds the graph with ``torch_cluster.radius_graph``, which is not available in LibTorch,
so it is replaced here with an equivalent pure-torch all-pairs implementation.
"""

import argparse
import copy
import json
import sys
import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor

CARD_FORMAT = "plumed-visnet/1"
LENGTH_UNITS = ("angstrom", "nm", "bohr")


class DistanceAllPairs(torch.nn.Module):
    """Drop-in replacement for ``torch_geometric.nn.models.visnet.Distance`` without torch_cluster.

    All pairs within ``cutoff`` belonging to the same graph are connected; unlike ``radius_graph``
    there is no ``max_num_neighbors`` truncation.
    """

    def __init__(self, cutoff: float, add_self_loops: bool = True) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.add_self_loops = add_self_loops

    def forward(self, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            dist = torch.cdist(pos, pos)
            mask = (dist < self.cutoff) & (batch.unsqueeze(0) == batch.unsqueeze(1))
            if not self.add_self_loops:
                mask.fill_diagonal_(False)
        edge_index = mask.nonzero().t()
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        if self.add_self_loops:
            # norm() has an undefined gradient at zero, so self loops get weight 0 without one
            nonself = edge_index[0] != edge_index[1]
            edge_weight = torch.zeros(edge_vec.size(0), dtype=pos.dtype, device=pos.device)
            edge_weight[nonself] = torch.norm(edge_vec[nonself], dim=-1)
        else:
            edge_weight = torch.norm(edge_vec, dim=-1)
        return edge_index, edge_weight, edge_vec


class ViSNetCV(torch.nn.Module):
    """Single-structure wrapper: ``(z, pos) -> property``, de-normalized."""

    def __init__(self, visnet: torch.nn.Module, norm_mean: float, norm_std: float) -> None:
        super().__init__()
        self.visnet = visnet
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def forward(self, z: Tensor, pos: Tensor) -> Tensor:
        batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        y, _ = self.visnet(z, pos, batch)
        return y.reshape(-1) * self.norm_std + self.norm_mean


def load_card_and_weights(args):
    if args.bundle:
        bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
        if not isinstance(bundle, dict) or "card" not in bundle or "state_dict" not in bundle:
            sys.exit(f"{args.bundle}: a bundle must be a dict with keys 'card' and 'state_dict'")
        return bundle["card"], bundle["state_dict"]
    if not (args.weights and args.card):
        sys.exit("either --bundle, or both --weights and --card, must be given")
    with open(args.card) as f:
        card = json.load(f)
    state_dict = torch.load(args.weights, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model_state_dict"):
        if isinstance(state_dict, dict) and key in state_dict:
            state_dict = state_dict[key]
    return card, state_dict


def validate_card(card):
    if card.get("format") != CARD_FORMAT:
        sys.exit(f"model card 'format' must be '{CARD_FORMAT}', got {card.get('format')!r}")
    for key in ("hparams", "target", "length_unit"):
        if key not in card:
            sys.exit(f"model card is missing the required key '{key}'")
    for key in ("name", "units", "norm_mean", "norm_std"):
        if key not in card["target"]:
            sys.exit(f"model card 'target' is missing the required key '{key}'")
    if not isinstance(card["target"]["name"], str) or not card["target"]["name"].strip() \
            or any(c.isspace() for c in card["target"]["name"]):
        sys.exit("model card 'target.name' must be a non-empty string without spaces (it is used as VISNET PROPERTY=...)")
    for key, value in card.get("provenance", {}).items():
        if not isinstance(value, str):
            sys.exit(f"model card 'provenance.{key}' must be a string")
    if card["length_unit"] not in LENGTH_UNITS:
        sys.exit(f"model card 'length_unit' must be one of {LENGTH_UNITS}")
    import torch_geometric

    trained = card.get("torch_geometric_version")
    if trained and trained.split(".")[:2] != torch_geometric.__version__.split(".")[:2]:
        warnings.warn(
            f"model trained with torch_geometric {trained}, exporting with {torch_geometric.__version__}"
        )


def strip_prefix(state_dict):
    """Remove a common prefix such as 'model.' added by training wrappers."""
    keys = list(state_dict)
    if all(k.startswith("model.") for k in keys):
        return {k[len("model."):]: v for k, v in state_dict.items()}
    return state_dict


def build_model(card, state_dict):
    from torch_geometric.nn.models import ViSNet

    hparams = dict(card["hparams"])
    hparams["derivative"] = False  # gradients are computed by PLUMED
    if hparams.get("atomref") is not None:
        hparams["atomref"] = torch.tensor(hparams["atomref"])
    model = ViSNet(**hparams)
    model.load_state_dict(strip_prefix(state_dict), strict=True)
    model.eval()
    return model


def parse_xyz(path):
    symbols_to_z = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
                    "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17,
                    "Ar": 18, "K": 19, "Ca": 20, "Br": 35, "I": 53}
    with open(path) as f:
        n = int(f.readline())
        f.readline()
        z, pos = [], []
        for _ in range(n):
            parts = f.readline().split()
            z.append(int(parts[0]) if parts[0].isdigit() else symbols_to_z[parts[0]])
            pos.append([float(x) for x in parts[1:4]])
    return torch.tensor(z, dtype=torch.long), torch.tensor(pos, dtype=torch.float64)


def compile_module(wrapper, z, pos):
    try:
        return torch.jit.script(wrapper), "script"
    except Exception as err:  # PyG MessagePassing is not always scriptable
        print(f"torch.jit.script failed ({type(err).__name__}), falling back to torch.jit.trace")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            traced = torch.jit.trace(wrapper, (z, pos), check_trace=False)
        return traced, f"trace (only valid for {z.numel()} atoms with these types)"


def value_and_grad(fn, z, pos):
    pos = pos.clone().requires_grad_(True)
    y = fn(z, pos)
    (g,) = torch.autograd.grad(y.sum(), pos)
    return y.detach(), g


def self_check(reference, exported, export_dtype, z, pos, norm_mean, norm_std, cutoff, max_neighbors):
    """Compare the exported module with the original PyG model and with finite differences."""
    n_neighbors = (torch.cdist(pos, pos) < cutoff).sum(1) - 1
    if int(n_neighbors.max()) > max_neighbors:
        warnings.warn(
            f"an atom has {int(n_neighbors.max())} neighbors within the cutoff, more than "
            f"max_num_neighbors={max_neighbors}: radius_graph truncated these during training, "
            "the exported all-pairs graph does not"
        )

    def ref_fn(z_, p_):
        y, _ = reference(z_, p_, torch.zeros_like(z_))
        return y.reshape(-1) * norm_std + norm_mean

    # PyG's Distance only works in float32, so the reference always runs in float32
    p = pos.float()
    y_ref, g_ref = value_and_grad(ref_fn, z, p)
    y_exp, g_exp = value_and_grad(exported, z, pos.to(export_dtype))
    y_exp, g_exp = y_exp.float(), g_exp.float()
    print(f"  value  PyG={y_ref.item():.8f}  TorchScript={y_exp.item():.8f}  "
          f"|diff|={abs(y_ref - y_exp).item():.2e}")
    print(f"  max |dCV/dx| difference PyG vs TorchScript: {(g_ref - g_exp).abs().max().item():.2e}")

    # finite differences in double precision on a copy of the model
    # (PyG's Distance hard-codes float32 edge weights, so use the pure-torch graph here)
    model64 = copy.deepcopy(reference).double()
    model64.representation_model.distance = DistanceAllPairs(cutoff, reference.representation_model.distance.add_self_loops)
    ref64 = ViSNetCV(model64, norm_mean, norm_std)
    p64 = pos.double()
    _, g64 = value_and_grad(ref64, z, p64)
    h = 1e-5
    fd = torch.zeros_like(p64)
    with torch.no_grad():
        for i in range(p64.shape[0]):
            for k in range(3):
                pp, pm = p64.clone(), p64.clone()
                pp[i, k] += h
                pm[i, k] -= h
                fd[i, k] = (ref64(z, pp) - ref64(z, pm)).item() / (2 * h)
    print(f"  max |autograd - finite difference| (float64): {(g64 - fd).abs().max().item():.2e}")
    return (g_ref - g_exp).abs().max().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", help="state_dict saved with torch.save(model.state_dict(), ...)")
    parser.add_argument("--card", help="JSON model card with hparams, target normalization and units")
    parser.add_argument("--bundle", help="torch.save({'card': ..., 'state_dict': ...}) file, instead of --weights/--card")
    parser.add_argument("--example", required=True, help="xyz file of a representative structure (in the card's length unit)")
    parser.add_argument("-o", "--output", required=True, help="output TorchScript file for PLUMED")
    parser.add_argument("--dtype", choices=("float32", "float64"),
                        help="precision of the exported model (default: float32, as trained). float64 is slower "
                             "but gives smoother derivatives, e.g. for NUMERICAL_DERIVATIVES checks")
    args = parser.parse_args()

    card, state_dict = load_card_and_weights(args)
    validate_card(card)
    model = build_model(card, state_dict)

    # keep the original module for the self-check, export a copy with the pure-torch graph
    dtype = getattr(torch, args.dtype) if args.dtype else next(model.parameters()).dtype
    export_model = copy.deepcopy(model).to(dtype)
    old = export_model.representation_model.distance
    export_model.representation_model.distance = DistanceAllPairs(old.cutoff, old.add_self_loops)

    target = card["target"]
    wrapper = ViSNetCV(export_model, float(target["norm_mean"]), float(target["norm_std"])).eval()

    z, pos = parse_xyz(args.example)
    module, how = compile_module(wrapper, z, pos.to(dtype))
    print(f"compiled with torch.jit.{how}")

    card = dict(card)
    card["export"] = {"method": how, "n_atoms_example": int(z.numel()), "torch_version": torch.__version__,
                      "dtype": str(dtype).replace("torch.", "")}
    torch.jit.save(module, args.output, _extra_files={"card.json": json.dumps(card)})
    print(f"saved {args.output}")

    print("self-check on", args.example)
    loaded = torch.jit.load(args.output)
    diff = self_check(model, loaded, dtype, z, pos, float(target["norm_mean"]), float(target["norm_std"]),
                      float(card["hparams"].get("cutoff", 5.0)), int(card["hparams"].get("max_num_neighbors", 32)))
    tol = 1e-4  # limited by the float32 PyG reference
    if diff > tol:
        sys.exit(f"exported gradients differ from the PyG model by {diff:.2e} > {tol:.0e}")


if __name__ == "__main__":
    main()
