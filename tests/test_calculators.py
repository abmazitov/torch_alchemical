from ase.io import read
import json
import torch
import numpy as np
from torch_alchemical.data import AtomisticDataset
from torch_alchemical.transforms import NeighborList
from torch_geometric.loader import DataLoader
from torch_alchemical.nn import PowerSpectrumFeatures
import metatensor


torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


class TestCalculators:
    device = "cpu"
    frames = read("./tests/data/hea_bulk_test_sample.xyz", index=":")
    all_species = np.unique(np.hstack([frame.numbers for frame in frames])).tolist()
    with open("./tests/configs/default_hypers_alchemical.json", "r") as f:
        hypers = json.load(f)
    transforms = [NeighborList(cutoff_radius=hypers["cutoff radius"])]
    dataset = AtomisticDataset(
        frames, target_properties=["energies"], transforms=transforms
    )
    dataloader = DataLoader(dataset, batch_size=len(frames), shuffle=False)
    batch = next(iter(dataloader))

    def test_ps_features(self):
        torch.manual_seed(0)
        calculator = PowerSpectrumFeatures(
            all_species=self.all_species,
            cutoff_radius=self.hypers["cutoff radius"],
            basis_cutoff=self.hypers["radial basis"]["E_max"],
            radial_basis_type=self.hypers["radial basis"]["type"],
            trainable_basis=self.hypers["radial basis"]["mlp"],
            num_pseudo_species=self.hypers["alchemical"],
            device=self.device,
        )
        with torch.no_grad():
            ps = calculator(
                positions=self.batch.pos,
                cells=self.batch.cell,
                numbers=self.batch.numbers,
                edge_indices=self.batch.edge_index,
                edge_shifts=self.batch.edge_shift,
                ptr=self.batch.ptr,
            )

        ref_ps = metatensor.torch.load("./tests/data/ps_calculator_test_data.npz")

        assert metatensor.operations.allclose(ps, ref_ps, atol=1e-5, rtol=1e-5)

    # def test_rs_features(self):
    #     rs_calculator = RadialSpectrumFeatures(
    #     all_species=all_species,
    #     cutoff_radius=hypers["cutoff radius"],
    #     basis_cutoff=hypers["radial basis"]["E_max"],
    #     radial_basis_type=hypers["radial basis"]["type"],
    #     trainable_basis=hypers["radial basis"]["mlp"],
    #     device=device,
    # )
    #     positions, cells, numbers, edge_index, edge_shift = extract_batch_data(
    #         self.batch
    #     )
    #     with torch.no_grad():
    #         rs = self.rs_calculator(positions, cells, numbers, edge_index, edge_shift)

    #     ref_rs = metatensor.core.io.load_custom_array(
    #         "./tests/data/rs_test_data.npz", metatensor.core.io.create_torch_array
    #     )

    #     assert metatensor.operations.allclose(rs, ref_rs, atol=1e-5, rtol=1e-5)
