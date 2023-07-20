from ase.io import read
import json
import torch
import numpy as np
from torch_alchemical.utils import convert_batch_to_torch_spex_dict
from torch_alchemical.data import AtomisticDataset
from torch_alchemical.transforms import NeighborList
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader
from torch_spex.structures import InMemoryDataset, TransformerNeighborList, collate_nl
from torch_spex.spherical_expansions import SphericalExpansion
import equistore


torch.set_default_dtype(torch.float64)


class TestTorchSpexCompatibility:
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
        frames, target_properties=["energies", "forces"], transforms=transforms
    )
    dataloader = PyGDataLoader(dataset, batch_size=len(frames), shuffle=False)
    batch = next(iter(dataloader))

    spex_transformers = [TransformerNeighborList(cutoff=hypers["cutoff radius"])]
    spex_dataset = InMemoryDataset(frames, spex_transformers)
    spex_loader = DataLoader(
        spex_dataset, batch_size=len(frames), collate_fn=collate_nl
    )
    spex_batch = next(iter(spex_loader))
    spex_batch.pop("positions")
    spex_batch.pop("cell")

    def test_batch_to_torch_spex_dict_conversion(self):
        batch_dict = convert_batch_to_torch_spex_dict(self.batch)
        assert batch_dict.keys() == self.spex_batch.keys()
        for key in batch_dict.keys():
            obj = batch_dict[key]
            target = self.spex_batch[key]
            if obj.dim() == target.dim() == 1:
                assert torch.equal(obj, target)
            elif obj.dim() == target.dim() == 2:
                obj_content, obj_counts = obj.unique(dim=0, return_counts=True)
                target_content, target_counts = target.unique(dim=0, return_counts=True)
                assert torch.equal(obj_content, target_content)
                assert torch.equal(obj_counts, target_counts)

    def test_spherical_expansion_coefficients(self):
        batch_dict = convert_batch_to_torch_spex_dict(self.batch)
        calculator = SphericalExpansion(
            hypers=self.hypers, all_species=self.all_species, device=self.device
        )
        with torch.no_grad():
            tm = calculator.forward(**batch_dict)
            spex_tm = calculator.forward(**self.spex_batch)
        assert equistore.operations.allclose(tm, spex_tm, atol=1e-5, rtol=1e-5)
