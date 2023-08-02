from ase.io import read
import json
import torch
import numpy as np
from torch_alchemical.utils import get_torch_spex_dict_from_batch
from torch_alchemical.data import AtomisticDataset
from torch_alchemical.transforms import NeighborList
from torch_geometric.loader import DataListLoader
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
        frames, target_properties=["energies"], transforms=transforms
    )
    datalistloader = DataListLoader(dataset, batch_size=len(frames), shuffle=False)
    batch_list = next(iter(datalistloader))

    spex_transformers = [TransformerNeighborList(cutoff=hypers["cutoff radius"])]
    spex_dataset = InMemoryDataset(frames, spex_transformers)
    spex_loader = DataLoader(
        spex_dataset, batch_size=len(frames), collate_fn=collate_nl
    )
    spex_batch = next(iter(spex_loader))

    def test_data_list_to_torch_spex_dict_conversion(self):
        data_list_dict = get_torch_spex_dict_from_batch(self.batch_list)
        assert data_list_dict.keys() == self.spex_batch.keys()
        for key in data_list_dict.keys():
            obj = data_list_dict[key]
            target = self.spex_batch[key]
            if isinstance(obj, list):
                obj = torch.cat(obj, dim=0)
            if isinstance(target, list):
                target = torch.cat(target, dim=0)
            equal = torch.equal(obj, target)
            if not equal:
                obj_content, obj_counts = obj.unique(dim=0, return_counts=True)
                target_content, target_counts = target.unique(dim=0, return_counts=True)
                content_equal = torch.equal(obj_content, target_content)
                counts_equal = torch.equal(obj_counts, target_counts)
            assert equal or (content_equal and counts_equal)

    def test_spherical_expansion_coefficients(self):
        data_list_dict = get_torch_spex_dict_from_batch(self.batch_list)
        calculator = SphericalExpansion(
            hypers=self.hypers, all_species=self.all_species, device=self.device
        )
        with torch.no_grad():
            data_tm = calculator.forward(**data_list_dict)
            spex_tm = calculator.forward(**self.spex_batch)
        assert equistore.operations.allclose(data_tm, spex_tm, atol=1e-5, rtol=1e-5)
