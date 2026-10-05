The VISNET module lets a ViSNet equivariant graph neural network, trained with
`torch_geometric.nn.models.ViSNet`, be used as a collective variable through the [VISNET](VISNET.md) action.
The model maps the atomic numbers and positions of a molecule to a scalar property. Its derivatives with respect to every
atomic position are computed at each step by LibTorch's automatic differentiation, so the CV can be printed, analysed or biased.

## Exporting a trained model

A `state_dict` holds only the weights. To rebuild the network, the `src/visnet/export_visnet.py` script also needs a JSON
*model card* with:
- the ViSNet constructor arguments;
- the normalization applied to the training target;
- the length unit of the training coordinates.

For example, next to the training code:

```python
import json, torch_geometric
VISNET_HPARAMS = dict(lmax=1, vecnorm_type=None, trainable_vecnorm=False, num_heads=8, num_layers=4,
                      hidden_channels=64, num_rbf=32, trainable_rbf=False, max_z=100, cutoff=5.0,
                      max_num_neighbors=32, vertex=False, reduce_op="sum", mean=0.0, std=1.0)
card = {
    "format": "plumed-visnet/1",
    "torch_geometric_version": torch_geometric.__version__,
    "hparams": VISNET_HPARAMS,
    # the model was trained on (y - norm_mean) / norm_std
    "target": {"name": "gap", "units": "eV", "norm_mean": float(target_mean[0]), "norm_std": float(target_std[0]),
               "level_of_theory": "G2W2"},
    "length_unit": "angstrom",
    # optional, written to the PLUMED log
    "provenance": {"weights_file": finetuned_weights_out, "finetuned_from": pretrained_weights,
                   "training_data": "adenine_1000_20260916.csv"},
}
json.dump(card, open("weights.json", "w"), indent=2)
```

`target.name` identifies the property the model predicts. It must be repeated in the PLUMED input with
`VISNET ... PROPERTY=gap`, and PLUMED stops if the two differ.

Then export:

```bash
python export_visnet.py --weights weights.pt --card weights.json --example frame.xyz -o model.pt
```

The script does the following:
- It replaces the `torch_cluster` neighbor search, which is not available in LibTorch, with a pure-torch all-pairs search. This search does not apply `max_num_neighbors`.
- It folds the target normalization into the model and stores the card inside the exported file.
- It checks the value and gradients of the exported model against the original model and against finite differences.

If PyG's message-passing layers cannot be scripted, the model is traced instead. A traced model is valid only for the number
and types of atoms in the example structure, which is always the case in a PLUMED run.

## Installation

This module is not installed by default and requires LibTorch. See the [pytorch module](module_pytorch.md) for how to obtain it. Then configure PLUMED with:

````
> ./configure --enable-libtorch --enable-modules=+visnet
````
