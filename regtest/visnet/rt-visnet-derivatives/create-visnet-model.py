# Creates the small, randomly initialised ViSNet model used by this test:
#   python create-visnet-model.py
# The model is committed because torch_geometric is not available in CI.
# It is exported in float64 so that NUMERICAL_DERIVATIVES can validate the autograd ones.
import json
import subprocess
import sys

import torch
from torch_geometric.nn.models import ViSNet

torch.manual_seed(42)
hparams = dict(lmax=1, num_heads=2, num_layers=2, hidden_channels=16, num_rbf=8,
               max_z=10, cutoff=5.0, reduce_op="sum")
torch.save(ViSNet(**hparams).state_dict(), "weights.pt")
card = {
    "format": "plumed-visnet/1",
    "hparams": hparams,
    "target": {"name": "test", "units": "arbitrary", "norm_mean": 1.0, "norm_std": 2.0},
    "length_unit": "angstrom",
}
with open("card.json", "w") as f:
    json.dump(card, f, indent=2)
with open("traj.xyz") as f:
    frame = "".join(f.readlines()[:17])
with open("frame.xyz", "w") as f:
    f.write(frame)
subprocess.run([sys.executable, "../../../src/visnet/export_visnet.py", "--weights", "weights.pt",
                "--card", "card.json", "--example", "frame.xyz", "--dtype", "float64",
                "-o", "visnet_model.pt"], check=True)
