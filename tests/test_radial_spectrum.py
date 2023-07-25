from ase.io import read
import json
import torch
import numpy as np
from torch_alchemical.data import AtomisticDataset
from torch_alchemical.transforms import NeighborList
from torch_geometric.loader import DataListLoader
from torch_alchemical.nn import RadialSpectrumFeatures
import equistore


torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


class TestRadialSpectrum:
    device = "cpu"
    frames = read("./tests/data/hea_bulk_test_sample.xyz", index=":")
    all_species = np.unique(np.hstack([frame.numbers for frame in frames]))
    with open("./tests/configs/default_hypers.json", "r") as f:
        hypers = json.load(f)
    transforms = [NeighborList(cutoff_radius=hypers["cutoff radius"])]
    dataset = AtomisticDataset(
        frames, target_properties=["energies"], transforms=transforms
    )
    dataloader = DataListLoader(dataset, batch_size=len(frames), shuffle=False)
    batch = next(iter(dataloader))
    calculator = RadialSpectrumFeatures(
        all_species=all_species,
        cutoff_radius=hypers["cutoff radius"],
        basis_cutoff=hypers["radial basis"]["E_max"],
        device=device,
    )

    def test_ps_features(self):
        positions = [data.pos for data in self.batch]
        cells = [data.cell for data in self.batch]
        numbers = [data.numbers for data in self.batch]
        edge_index = [data.edge_index for data in self.batch]
        edge_shift = [data.edge_shift for data in self.batch]
        with torch.no_grad():
            rs = self.calculator(positions, cells, numbers, edge_index, edge_shift)

        ref_rs = equistore.core.io.load_custom_array(
            "./tests/data/rs_test_data.npz", equistore.core.io.create_torch_array
        )

        assert equistore.operations.allclose(rs, ref_rs, atol=1e-5, rtol=1e-5)