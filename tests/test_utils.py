import numpy as np
import torch
from ase.io import read
from torch_geometric.loader import DataLoader

from torch_alchemical.data import AtomisticDataset
from torch_alchemical.utils import (
    get_autograd_forces,
    get_compositions_from_numbers,
    get_metatensor_systems,
    get_target_properties,
)

device = "cpu"
frames = read("./tests/data/hea_bulk_test_sample.xyz", index=":")
all_species = np.unique(np.hstack([frame.numbers for frame in frames])).tolist()
dataset = AtomisticDataset(frames, target_properties=["energies", "forces"])
dataloader = DataLoader(dataset, batch_size=len(frames), shuffle=False)
batch = next(iter(dataloader))


def test_batch_numbers_to_composition_convertion():
    compositions = get_compositions_from_numbers(
        batch.numbers, all_species, batch.batch
    )
    assert torch.equal(
        torch.stack(compositions),
        torch.stack(torch.load("./tests/data/compositions_data.pt")),
    )


def test_autograd_forces():
    positions = torch.load("./tests/data/hea_bulk_test_positions.pt")
    energies = positions**2
    forces = get_autograd_forces(energies, positions)[0]
    assert torch.equal(forces, -2 * positions)


def test_target_properties_getter():
    properties = get_target_properties(frames[0], ["energies", "forces"])
    ref_properties = torch.load("./tests/data/hea_bulk_test_target_properties.pt")
    for key in properties.keys():
        assert torch.allclose(properties[key], ref_properties[key], atol=1e-4)


def test_get_metatensor_systems():
    n_systems = 3
    batch = torch.tensor([1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7])
    species = torch.ones(len(batch))
    positions = torch.rand([len(batch), 3])
    cells = torch.rand([3 * n_systems, 3])

    systems = get_metatensor_systems(
        batch=batch, types=species, positions=positions, cells=cells
    )

    assert len(systems) == n_systems
    # TODO do some more asserts that this actually working
