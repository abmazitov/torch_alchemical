from ase.io import read
import json
import torch
import numpy as np
from torch_alchemical.data import AtomisticDataset
from torch_alchemical.transforms import NeighborList
from torch_geometric.loader import DataListLoader
import equistore
from torch_alchemical import nn


torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


class TestNNLayers:
    device = "cpu"
    frames = read("./tests/data/hea_bulk_test_sample.xyz", index=":")
    all_species = np.unique(np.hstack([frame.numbers for frame in frames]))
    with open("./tests/configs/default_hypers_alchemical.json", "r") as f:
        hypers = json.load(f)
    transforms = [NeighborList(cutoff_radius=hypers["cutoff radius"])]
    dataset = AtomisticDataset(
        frames, target_properties=["energies"], transforms=transforms
    )
    dataloader = DataListLoader(dataset, batch_size=len(frames), shuffle=False)
    batch = next(iter(dataloader))
    calculator = nn.PowerSpectrumFeatures(
        all_species=all_species,
        cutoff_radius=hypers["cutoff radius"],
        basis_cutoff=hypers["radial basis"]["E_max"],
        radial_basis_type=hypers["radial basis"]["type"],
        trainable_basis=hypers["radial basis"]["mlp"],
        num_pseudo_species=hypers["alchemical"],
        device=device,
    )
    with torch.no_grad():
        ps = calculator(
            positions=[data.pos for data in batch],
            cells=[data.cell for data in batch],
            numbers=[data.numbers for data in batch],
            edge_indices=[data.edge_index for data in batch],
            edge_shifts=[data.edge_shift for data in batch],
        )

    def test_linear(self):
        torch.manual_seed(0)
        ps_input_size = self.calculator.num_features
        linear = nn.Linear(ps_input_size, 1)
        with torch.no_grad():
            linear_ps = linear(self.ps)
        ref_linear_ps = equistore.core.io.load_custom_array(
            "./tests/data/linear_ps_test_data.npz", equistore.core.io.create_torch_array
        )
        assert equistore.operations.allclose(
            linear_ps, ref_linear_ps, atol=1e-5, rtol=1e-5
        )

    def test_layer_norm(self):
        torch.manual_seed(0)
        ps_input_size = self.calculator.num_features
        norm = nn.LayerNorm(ps_input_size)
        with torch.no_grad():
            norm_ps = norm(self.ps)
        ref_norm_ps = equistore.core.io.load_custom_array(
            "./tests/data/norm_ps_test_data.npz", equistore.core.io.create_torch_array
        )
        assert equistore.operations.allclose(norm_ps, ref_norm_ps, atol=1e-5, rtol=1e-5)

    def test_loss_functions(self):
        ref_energies = torch.load("./tests/data/hea_bulk_test_ps_energies.pt")
        ref_forces = torch.load("./tests/data/hea_bulk_test_ps_forces.pt")
        ref_forces = torch.cat(ref_forces, dim=0)
        loss_fns = [
            nn.MAELoss(),
            nn.MSELoss(),
            nn.WeightedMAELoss(weights=[1.0, 1.0]),
            nn.WeightedMSELoss(weights=[1.0, 1.0]),
        ]
        for loss_fn in loss_fns:
            loss = loss_fn(
                predicted=[ref_energies, ref_forces], target=[ref_energies, ref_forces]
            )
            assert torch.allclose(loss, torch.tensor(0.0))
