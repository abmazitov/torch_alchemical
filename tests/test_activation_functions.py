from ase.io import read
import json
import torch
import numpy as np
from torch_alchemical.data import AtomisticDataset
from torch_alchemical.transforms import NeighborList
from torch_geometric.loader import DataListLoader
from torch_alchemical.nn import PowerSpectrumFeatures, ReLU, SiLU
import equistore


torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


class TestActivationFunctions:
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
    calculator = PowerSpectrumFeatures(
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

    def test_relu(self):
        relu = ReLU()
        ps_relu = relu(self.ps)
        ref_ps_relu = equistore.core.io.load_custom_array(
            "./tests/data/relu_ps_test_data.npz", equistore.core.io.create_torch_array
        )
        assert equistore.operations.allclose(ps_relu, ref_ps_relu, atol=1e-5, rtol=1e-5)

    def test_silu(self):
        silu = SiLU()
        ps_silu = silu(self.ps)
        ref_ps_silu = equistore.core.io.load_custom_array(
            "./tests/data/silu_ps_test_data.npz", equistore.core.io.create_torch_array
        )
        assert equistore.operations.allclose(ps_silu, ref_ps_silu, atol=1e-5, rtol=1e-5)
