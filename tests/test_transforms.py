import json

import numpy as np
import torch
from ase.io import read
from torch_geometric.loader import DataLoader

from torch_alchemical.data import AtomisticDataset
from torch_alchemical.transforms import NeighborList

torch.set_default_dtype(torch.float64)


device = "cpu"
frames = read("./tests/data/hea_samples_bulk.xyz", index=":")
all_species = np.unique(np.hstack([frame.numbers for frame in frames])).tolist()
with open("./tests/configs/default_hypers_alchemical.json", "r") as f:
    hypers = json.load(f)


def test_neighborlist():
    dataset = AtomisticDataset(
        frames,
        target_properties=["energies", "forces"],
        transforms=[NeighborList(cutoff_radius=hypers["cutoff radius"])],
    )
    dataloader = DataLoader(dataset, batch_size=len(frames), shuffle=False)
    batch = next(iter(dataloader))
    ref_edge_index = torch.load("./tests/data/hea_bulk_test_edge_index.pt")
    ref_edge_offsets = torch.load("./tests/data/hea_bulk_test_edge_offsets.pt")
    assert torch.allclose(batch.edge_index, ref_edge_index, atol=1e-4)
    assert torch.allclose(batch.edge_offsets, ref_edge_offsets, atol=1e-4)
