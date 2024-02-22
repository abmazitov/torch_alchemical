import json

import numpy as np
import torch
from ase.io import read
from torch_geometric.loader import DataLoader

from torch_alchemical.data import AtomisticDataset
from torch_alchemical.models import AlchemicalModel, BPPSModel, PowerSpectrumModel
from torch_alchemical.transforms import NeighborList
from torch_alchemical.utils import get_autograd_forces

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


def evaluate_model(model, batch, ref_energies, ref_forces):
    energies = model(
        positions=batch.pos,
        cells=batch.cell,
        numbers=batch.numbers,
        edge_indices=batch.edge_index,
        edge_offsets=batch.edge_offsets,
        batch=batch.batch,
    )
    forces = get_autograd_forces(energies, batch.pos)[0]
    assert torch.allclose(energies, ref_energies)
    assert torch.allclose(forces, ref_forces)


class TestModels:
    device = "cpu"
    frames = read("./tests/data/hea_bulk_test_sample.xyz", index=":")
    all_species = np.unique(np.hstack([frame.numbers for frame in frames])).tolist()
    with open("./tests/configs/default_hypers_alchemical.json", "r") as f:
        hypers = json.load(f)
    with open("./tests/configs/default_model_parameters.json", "r") as f:
        default_model_parameters = json.load(f)
    transforms = [NeighborList(cutoff_radius=hypers["cutoff radius"])]
    dataset = AtomisticDataset(
        frames, target_properties=["energies", "forces"], transforms=transforms
    )
    dataloader = DataLoader(dataset, batch_size=len(frames), shuffle=False)
    batch = next(iter(dataloader))

    def test_power_spectrum_model(self):
        torch.manual_seed(0)
        model = PowerSpectrumModel(
            unique_numbers=self.all_species,
            **self.default_model_parameters,
        )
        ref_energies = torch.load("./tests/data/hea_bulk_test_ps_energies.pt")
        ref_forces = torch.load("./tests/data/hea_bulk_test_ps_forces.pt")
        evaluate_model(model, self.batch, ref_energies, ref_forces)

    def test_bpps_model(self):
        torch.manual_seed(0)
        model = BPPSModel(
            unique_numbers=self.all_species,
            **self.default_model_parameters,
        )
        ref_energies = torch.load("./tests/data/hea_bulk_test_bpps_energies.pt")
        ref_forces = torch.load("./tests/data/hea_bulk_test_bpps_forces.pt")
        evaluate_model(model, self.batch, ref_energies, ref_forces)

    def test_alchemical_model(self):
        torch.manual_seed(0)
        model = AlchemicalModel(
            unique_numbers=self.all_species,
            **self.default_model_parameters,
        )
        ref_energies = torch.load("./tests/data/hea_bulk_test_alchemical_energies.pt")
        ref_forces = torch.load("./tests/data/hea_bulk_test_alchemical_forces.pt")
        evaluate_model(model, self.batch, ref_energies, ref_forces)
