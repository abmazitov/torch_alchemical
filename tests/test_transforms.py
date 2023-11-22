from ase.io import read
import json
import torch
import numpy as np
from torch_alchemical.data import AtomisticDataset
from torch_alchemical.transforms import NeighborList
from torch_geometric.loader import DataLoader


torch.set_default_dtype(torch.float64)


class TestTransforms:
    device = "cpu"
    frames = read("./tests/data/hea_samples_bulk.xyz", index=":")
    all_species = np.unique(np.hstack([frame.numbers for frame in frames])).tolist()
    with open("./tests/configs/default_hypers_alchemical.json", "r") as f:
        hypers = json.load(f)

    def test_neighborlist(self):
        dataset = AtomisticDataset(
            self.frames,
            target_properties=["energies", "forces"],
            transforms=[NeighborList(cutoff_radius=self.hypers["cutoff radius"])],
        )
        dataloader = DataLoader(dataset, batch_size=len(self.frames), shuffle=False)
        batch = next(iter(dataloader))
        ref_edge_index = torch.load("./tests/data/hea_bulk_test_edge_index.pt")
        ref_edge_offsets = torch.load("./tests/data/hea_bulk_test_edge_offsets.pt")
        assert torch.allclose(batch.edge_index, ref_edge_index)
        assert torch.allclose(batch.edge_offsets, ref_edge_offsets)
