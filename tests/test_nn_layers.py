from ase.io import read
import json
import torch
import numpy as np
from torch_alchemical.data import AtomisticDataset
from torch_alchemical.transforms import NeighborList
from torch_geometric.loader import DataListLoader
from torch_alchemical.nn import PowerSpectrumFeatures, Linear
import equistore


torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


class TestNNLayers:
    """
    Test the internal datatype conversion from torch_geometric.data.Batch to a dict
    representation in torch_spex library, and a following calculation of the SphericalExpansion
    coefficients.
    """

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
        linear = Linear(ps_input_size, 1)
        with torch.no_grad():
            linear_ps = linear(self.ps)
        ref_linear_ps = equistore.core.io.load_custom_array(
            "./tests/data/linear_ps_test_data.npz", equistore.core.io.create_torch_array
        )
        print(linear_ps[0].values)
        print(ref_linear_ps[0].values)
        assert equistore.operations.allclose(
            linear_ps, ref_linear_ps, atol=1e-5, rtol=1e-5
        )
