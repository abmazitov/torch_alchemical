from ase.io import read
import json
import torch
import numpy as np
from torch_alchemical.data import AtomisticDataset
from torch_alchemical.transforms import NeighborList
from torch_geometric.loader import DataListLoader
from torch_alchemical.models import AlchemicalModel
from torch_alchemical.utils import get_autograd_forces, extract_batch_data


torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


class TestModels:
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
    with open("./tests/configs/alchemical_model_parameters.json", "r") as f:
        model_parameters = json.load(f)
    transforms = [NeighborList(cutoff_radius=hypers["cutoff radius"])]
    dataset = AtomisticDataset(
        frames, target_properties=["energies", "forces"], transforms=transforms
    )
    dataloader = DataListLoader(dataset, batch_size=len(frames), shuffle=False)
    batch = next(iter(dataloader))

    def test_alchemical_model(self):
        torch.manual_seed(0)
        model = AlchemicalModel(
            hidden_sizes=self.model_parameters["hidden_sizes"],
            output_size=self.model_parameters["output_size"],
            unique_numbers=self.all_species,
            cutoff=self.hypers["cutoff radius"],
            basis_cutoff_power_spectrum=self.hypers["radial basis"]["E_max"],
            num_pseudo_species=self.hypers["alchemical"],
            device="cpu",
        )
        positions, cells, numbers, edge_indices, edge_shifts = extract_batch_data(
            self.batch
        )
        energies = model(positions, cells, numbers, edge_indices, edge_shifts)
        forces = get_autograd_forces(energies, positions)

        ref_energies = torch.load("./tests/data/hea_bulk_test_alchemical_energies.pt")
        ref_forces = torch.load("./tests/data/hea_bulk_test_alchemical_forces.pt")

        print(energies)
        print(ref_energies)
        assert torch.allclose(energies, ref_energies)
        assert torch.allclose(torch.cat(forces), torch.cat(ref_forces))
