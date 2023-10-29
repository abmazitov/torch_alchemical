from ase.io import read
import numpy as np
import torch
from torch_alchemical.data import AtomisticDataset
from torch_geometric.loader import DataLoader
from torch_alchemical.utils import (
    get_compositions_from_numbers,
    get_autograd_forces,
    get_target_properties,
)


class TestUtils:
    device = "cpu"
    frames = read("./tests/data/hea_bulk_test_sample.xyz", index=":")
    all_species = np.unique(np.hstack([frame.numbers for frame in frames])).tolist()
    dataset = AtomisticDataset(frames, target_properties=["energies", "forces"])
    dataloader = DataLoader(dataset, batch_size=len(frames), shuffle=False)
    batch = next(iter(dataloader))

    def test_batch_numbers_to_composition_convertion(self):
        compositions = get_compositions_from_numbers(
            self.batch.numbers, self.all_species, self.batch.ptr
        )
        assert torch.equal(
            torch.stack(compositions),
            torch.stack(torch.load("./tests/data/compositions_data.pt")),
        )

    def test_autograd_forces(self):
        positions = torch.load("./tests/data/hea_bulk_test_positions.pt")
        energies = positions**2
        forces = get_autograd_forces(energies, positions)[0]
        assert torch.equal(forces, -2 * positions)

    def test_target_properties_getter(self):
        properties = get_target_properties(self.frames[0], ["energies", "forces"])
        ref_properties = torch.load("./tests/data/hea_bulk_test_target_properties.pt")
        for key in properties.keys():
            assert torch.equal(properties[key], ref_properties[key])
