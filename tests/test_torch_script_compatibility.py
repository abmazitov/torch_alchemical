import json

import metatensor
import numpy as np
import torch
from ase.io import read
from metatensor.torch import Labels
from torch_geometric.loader import DataLoader

from torch_alchemical import nn
from torch_alchemical.data import AtomisticDataset
from torch_alchemical.models import AlchemicalModel, BPPSModel
from torch_alchemical.transforms import NeighborList

torch.set_default_dtype(torch.float64)


class TestNNLayers:
    ps = metatensor.torch.load("./tests/data/ps_test_data.npz")
    unique_numbers = ps.keys.values.flatten().tolist()
    emb_ps = metatensor.torch.load("./tests/data/emb_ps_test_data.npz")
    ps_input_size = ps.block(0).values.shape[-1]
    contraction_layer = torch.load("./tests/data/contraction_layer.pt")
    num_channels = 4

    def test_alchemical_embedding(self):
        embedding = torch.jit.script(
            nn.AlchemicalEmbedding(self.unique_numbers, self.contraction_layer)
        )
        with torch.no_grad():
            embedding(self.ps)

    def test_linear(self):
        linear = torch.jit.script(nn.Linear(self.ps_input_size, 1))
        with torch.no_grad():
            linear(self.ps)

    def test_linear_map(self):
        linear_map = torch.jit.script(
            nn.LinearMap(
                keys=Labels(
                    names=["a_i"], values=torch.tensor(self.unique_numbers).view(-1, 1)
                ),
                in_features=self.ps_input_size,
                out_features=1,
                bias=False,
            )
        )
        with torch.no_grad():
            linear_map(self.ps)

    def test_layer_norm(self):
        norm = torch.jit.script(nn.LayerNorm(self.ps_input_size))
        with torch.no_grad():
            norm(self.ps)

    def test_relu(self):
        relu = torch.jit.script(nn.ReLU())
        with torch.no_grad():
            relu(self.ps)

    def test_silu(self):
        silu = torch.jit.script(nn.SiLU())
        with torch.no_grad():
            silu(self.ps)


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

    def test_alchemical_model(self):
        model = torch.jit.script(
            AlchemicalModel(
                unique_numbers=self.all_species,
                contract_center_species=False,
                num_pseudo_species=4,
                **self.default_model_parameters,
            )
        )
        model.forward(
            positions=self.batch.pos,
            cells=self.batch.cell,
            numbers=self.batch.numbers,
            edge_indices=self.batch.edge_index,
            edge_offsets=self.batch.edge_offsets,
            batch=self.batch.batch,
        )

    def test_bpps_model(self):
        model = torch.jit.script(
            BPPSModel(
                unique_numbers=self.all_species,
                **self.default_model_parameters,
            )
        )
        model.forward(
            positions=self.batch.pos,
            cells=self.batch.cell,
            numbers=self.batch.numbers,
            edge_indices=self.batch.edge_index,
            edge_offsets=self.batch.edge_offsets,
            batch=self.batch.batch,
        )
