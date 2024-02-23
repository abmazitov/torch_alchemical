import json

import numpy as np
import torch
from ase.io import read
from torch_geometric.loader import DataLoader

from torch_alchemical.data import AtomisticDataset
from torch_alchemical.models import AlchemicalModel, BPPSModel
from torch_alchemical.transforms import NeighborList

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


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

    def test_bpps_model(self):
        torch.manual_seed(0)
        model = BPPSModel(
            unique_numbers=self.all_species,
            **self.default_model_parameters,
        )
        with torch.no_grad():
            predictions = model(
                positions=self.batch.pos,
                cells=self.batch.cell,
                numbers=self.batch.numbers,
                edge_indices=self.batch.edge_index,
                edge_offsets=self.batch.edge_offsets,
                batch=self.batch.batch,
            )
            assert torch.allclose(predictions, torch.tensor([[7.4553], [-6.9550]]))

    def test_alchemical_model_with_contraction(self):
        torch.manual_seed(0)
        model = AlchemicalModel(
            unique_numbers=self.all_species,
            contract_center_species=True,
            **self.default_model_parameters,
        )
        with torch.no_grad():
            predictions = model(
                positions=self.batch.pos,
                cells=self.batch.cell,
                numbers=self.batch.numbers,
                edge_indices=self.batch.edge_index,
                edge_offsets=self.batch.edge_offsets,
                batch=self.batch.batch,
            )
            assert torch.allclose(predictions, torch.tensor([[-2.3405], [11.4522]]))

    def test_alchemical_model_without_contraction(self):
        torch.manual_seed(0)
        model = AlchemicalModel(
            unique_numbers=self.all_species,
            contract_center_species=False,
            **self.default_model_parameters,
        )
        with torch.no_grad():
            predictions = model(
                positions=self.batch.pos,
                cells=self.batch.cell,
                numbers=self.batch.numbers,
                edge_indices=self.batch.edge_index,
                edge_offsets=self.batch.edge_offsets,
                batch=self.batch.batch,
            )
            assert torch.allclose(predictions, torch.tensor([[3.7277], [-3.4775]]))
