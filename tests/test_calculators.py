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


class TestCalculators:
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

    def test_ps_features(self):
        torch.manual_seed(0)
        calculator = PowerSpectrumFeatures(
            all_species=self.all_species,
            cutoff_radius=self.hypers["cutoff radius"],
            basis_cutoff=self.hypers["radial basis"]["E_max"],
            radial_basis_type=self.hypers["radial basis"]["type"],
            trainable_basis=self.hypers["radial basis"]["mlp"],
            num_pseudo_species=self.hypers["alchemical"],
        ).to(self.device)
        with torch.no_grad():
            ps = calculator(
                positions=self.batch.pos,
                cells=self.batch.cell,
                numbers=self.batch.numbers,
                edge_indices=self.batch.edge_index,
                edge_offsets=self.batch.edge_offsets,
                batch=self.batch.batch,
            )

        ref_ps = metatensor.torch.load("./tests/data/ps_calculator_test_data.npz")

        assert metatensor.operations.allclose(ps, ref_ps, atol=1e-5, rtol=1e-5)
