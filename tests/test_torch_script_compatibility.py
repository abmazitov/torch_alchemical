from torch_alchemical import nn
import metatensor
import torch
from torch_alchemical.data import AtomisticDataset
from torch_alchemical.transforms import NeighborList
from torch_geometric.loader import DataLoader
from torch_alchemical.models import PowerSpectrumModel, BPPSModel
from ase.io import read
import json
import numpy as np

torch.set_default_dtype(torch.float64)


class TestNNLayers:
    ps = metatensor.torch.load("./tests/data/ps_test_data.npz")
    ps_input_size = ps.block(0).values.shape[-1]

    def test_linear(self):
        linear = torch.jit.script(nn.Linear(self.ps_input_size, 1))
        with torch.no_grad():
            linear(self.ps)

    # def test_linear_map(self):
    #     linear_map = torch.jit.script(
    #         nn.LinearMap(
    #             keys=self.ps.keys.values.flatten().tolist(),
    #             in_features=self.ps_input_size,
    #             out_features=1,
    #             bias=False,
    #         )
    #     )
    #     with torch.no_grad():
    #         linear_map(self.ps)

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
    with open("./tests/configs/ps_model_parameters.json", "r") as f:
        ps_model_parameters = json.load(f)
    with open("./tests/configs/bpps_model_parameters.json", "r") as f:
        bpps_model_parameters = json.load(f)
    transforms = [NeighborList(cutoff_radius=hypers["cutoff radius"])]
    dataset = AtomisticDataset(
        frames, target_properties=["energies", "forces"], transforms=transforms
    )
    dataloader = DataLoader(dataset, batch_size=len(frames), shuffle=False)
    batch = next(iter(dataloader))

    def test_power_spectrum_model(self):
        model = torch.jit.script(
            PowerSpectrumModel(
                unique_numbers=self.all_species,
                **self.ps_model_parameters,
            )
        )
        model.forward(
            positions=self.batch.pos,
            cells=self.batch.cell,
            numbers=self.batch.numbers,
            edge_indices=self.batch.edge_index,
            edge_shifts=self.batch.edge_shift,
            ptr=self.batch.ptr,
        )

    # def test_bpps_model(self):
    #     model = torch.jit.script(
    #         BPPSModel(
    #             unique_numbers=self.all_species,
    #             **self.bpps_model_parameters,
    #         )
    #     )
    #     model.forward(
    #         positions=self.batch.pos,
    #         cells=self.batch.cell,
    #         numbers=self.batch.numbers,
    #         edge_indices=self.batch.edge_index,
    #         edge_shifts=self.batch.edge_shift,
    #         ptr=self.batch.ptr,
    #     )
