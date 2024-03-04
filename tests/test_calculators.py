import json

import metatensor
import numpy as np
import torch
from ase.io import read
from torch_geometric.loader import DataLoader

from torch_alchemical.data import AtomisticDataset
from torch_alchemical.nn import PowerSpectrumFeatures
from torch_alchemical.transforms import NeighborList

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


device = "cuda" if torch.cuda.is_available() else "cpu"
frames = read("./tests/data/hea_bulk_test_sample.xyz", index=":")
all_species = np.unique(np.hstack([frame.numbers for frame in frames])).tolist()
with open("./tests/configs/default_hypers_alchemical.json", "r") as f:
    hypers = json.load(f)
transforms = [NeighborList(cutoff_radius=hypers["cutoff radius"])]
dataset = AtomisticDataset(
    frames, target_properties=["energies"], transforms=transforms
)
dataloader = DataLoader(dataset, batch_size=len(frames), shuffle=False)
batch = next(iter(dataloader)).to(device)


def test_ps_features():
    torch.manual_seed(0)
    calculator = PowerSpectrumFeatures(
        all_species=all_species,
        cutoff_radius=hypers["cutoff radius"],
        basis_cutoff=hypers["radial basis"]["E_max"],
        radial_basis_type=hypers["radial basis"]["type"],
        trainable_basis=hypers["radial basis"]["mlp"],
        num_pseudo_species=hypers["alchemical"],
        normalize=hypers["normalize"],
    ).to(device)
    with torch.no_grad():
        ps = calculator(
            positions=batch.pos,
            cells=batch.cell,
            numbers=batch.numbers,
            edge_indices=batch.edge_index,
            edge_offsets=batch.edge_offsets,
            batch=batch.batch,
        )

    ref_ps = metatensor.torch.load("./tests/data/ps_calculator_test_data.npz")

    assert metatensor.operations.allclose(ps, ref_ps, atol=1e-4)
