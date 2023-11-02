from ase.io import read
import json
import torch
import numpy as np
from torch_alchemical.data import AtomisticDataset
from torch_alchemical.transforms import NeighborList, TargetPropertiesNormalizer
from torch_geometric.loader import DataLoader


torch.set_default_dtype(torch.float64)


class TestTransforms:
    device = "cpu"
    frames = read("./tests/data/hea_samples_bulk.xyz", index=":")
    all_species = np.unique(np.hstack([frame.numbers for frame in frames])).tolist()
    with open("./tests/configs/default_hypers_alchemical.json", "r") as f:
        hypers = json.load(f)

    def test_neighborlist(self):
        dataset = AtomisticDataset(
            self.frames,
            target_properties=["energies", "forces"],
            transforms=[NeighborList(cutoff_radius=self.hypers["cutoff radius"])],
        )
        dataloader = DataLoader(dataset, batch_size=len(self.frames), shuffle=False)
        batch = next(iter(dataloader))
        ref_edge_index = torch.load("./tests/data/hea_bulk_test_edge_index.pt")
        ref_edge_shift = torch.load("./tests/data/hea_bulk_test_edge_shift.pt")
        assert torch.allclose(batch.edge_index, ref_edge_index)
        assert torch.allclose(batch.edge_shift, ref_edge_shift)

    def test_properties_normalizer(self):
        dataset = AtomisticDataset(
            self.frames,
            target_properties=["energies", "forces"],
        )
        dataloader = DataLoader(dataset, batch_size=len(self.frames), shuffle=False)
        batch = next(iter(dataloader))
        normalizer = TargetPropertiesNormalizer(
            unique_numbers=self.all_species,
            train_frames=self.frames,
            target_properties=["energies", "forces"],
        )
        unnormalized_energies = batch.energies.clone()
        unnormalized_forces = batch.forces.clone()

        batch = normalizer(batch)

        normalized_energies = batch.energies.clone()
        normalized_forces = batch.forces.clone()

        normalizer.denormalize(batch)

        denormalized_energies = batch.energies.clone()
        denormalized_forces = batch.forces.clone()

        ref_unnormalized_energies = torch.load(
            "./tests/data/hea_bulk_test_unnormalized_energies.pt"
        )
        ref_unnormalized_forces = torch.load(
            "./tests/data/hea_bulk_test_unnormalized_forces.pt"
        )

        ref_normalized_energies = torch.load(
            "./tests/data/hea_bulk_test_normalized_energies.pt"
        )
        ref_normalized_forces = torch.load(
            "./tests/data/hea_bulk_test_normalized_forces.pt"
        )

        ref_denormalized_energies = torch.load(
            "./tests/data/hea_bulk_test_denormalized_energies.pt"
        )
        ref_denormalized_forces = torch.load(
            "./tests/data/hea_bulk_test_denormalized_forces.pt"
        )

        assert torch.allclose(unnormalized_energies, denormalized_energies)
        assert torch.allclose(unnormalized_forces, denormalized_forces)

        assert torch.allclose(unnormalized_energies, ref_unnormalized_energies)
        assert torch.allclose(unnormalized_forces, ref_unnormalized_forces)
        assert torch.allclose(normalized_energies, ref_normalized_energies)
        assert torch.allclose(normalized_forces, ref_normalized_forces)
        assert torch.allclose(denormalized_energies, ref_denormalized_energies)
        assert torch.allclose(denormalized_forces, ref_denormalized_forces)
