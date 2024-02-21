from ase.io import read
import json
import torch
import numpy as np
from torch_alchemical.utils import get_torch_spex_dict
from torch_alchemical.data import AtomisticDataset
from torch_alchemical.transforms import NeighborList
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as SpexDataLoader
from torch_spex.structures import InMemoryDataset, TransformerNeighborList, collate_nl
from torch_spex.spherical_expansions import SphericalExpansion
import metatensor


torch.set_default_dtype(torch.float64)


def compare_dicts(data_dict, spex_batch):
    assert data_dict.keys() == spex_batch.keys()
    for key in data_dict.keys():
        obj = data_dict[key]
        target = spex_batch[key]
        equal = torch.equal(obj, target)
        dtype_equal = obj.dtype == target.dtype
        assert equal and dtype_equal


class TestTorchSpexCompatibility:
    device = "cpu"
    frames = read("./tests/data/hea_bulk_test_sample.xyz", index=":")
    all_species = np.unique(np.hstack([frame.numbers for frame in frames]))
    with open("./tests/configs/default_hypers_alchemical.json", "r") as f:
        hypers = json.load(f)
    transforms = [NeighborList(cutoff_radius=hypers["cutoff radius"])]
    dataset = AtomisticDataset(
        frames, target_properties=["energies"], transforms=transforms
    )
    dataloader = DataLoader(dataset, batch_size=len(frames), shuffle=False)
    batch = next(iter(dataloader))

    spex_transformers = [TransformerNeighborList(cutoff=hypers["cutoff radius"])]
    spex_dataset = InMemoryDataset(frames, spex_transformers)
    spex_loader = SpexDataLoader(
        spex_dataset, batch_size=len(frames), collate_fn=collate_nl
    )
    spex_batch = next(iter(spex_loader))

    def test_batch_to_torch_spex_dict_conversion(self):
        data_dict = get_torch_spex_dict(
            positions=self.batch.pos,
            cells=self.batch.cell,
            numbers=self.batch.numbers,
            edge_indices=self.batch.edge_index,
            edge_offsets=self.batch.edge_offsets,
            batch=self.batch.batch,
        )
        compare_dicts(data_dict, self.spex_batch)

    def test_spherical_expansion_coefficients(self):
        data_dict = get_torch_spex_dict(
            positions=self.batch.pos,
            cells=self.batch.cell,
            numbers=self.batch.numbers,
            edge_indices=self.batch.edge_index,
            edge_offsets=self.batch.edge_offsets,
            batch=self.batch.batch,
        )
        calculator = SphericalExpansion(
            hypers=self.hypers, all_species=self.all_species
        )
        with torch.no_grad():
            data_tm = calculator.forward(**data_dict)
            spex_tm = calculator.forward(**self.spex_batch)
        assert metatensor.operations.allclose(data_tm, spex_tm, atol=1e-5, rtol=1e-5)
