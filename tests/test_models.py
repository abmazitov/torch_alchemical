from ase.io import read
import json
import torch
import numpy as np
from torch_alchemical.data import AtomisticDataset
from torch_alchemical.transforms import NeighborList
from torch_geometric.loader import DataLoader
from torch_alchemical.models import PowerSpectrumModel, BPPSModel
from torch_alchemical.utils import get_autograd_forces


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
    all_species = np.unique(np.hstack([frame.numbers for frame in frames])).tolist()
    with open("./tests/configs/default_hypers_alchemical.json", "r") as f:
        hypers = json.load(f)
    with open("./tests/configs/ps_model_parameters.json", "r") as f:
        ps_model_parameters = json.load(f)
    with open("./tests/configs/bpps_model_parameters.json", "r") as f:
        bpps_model_parameters = json.load(f)
    transforms = [NeighborList(cutoff_radius=hypers["cutoff radius"])]
    dataset = AtomisticDataset(
        frames, target_properties=["energies", "forces"], transforms=transforms
    )
    dataloader = DataLoader(dataset, batch_size=len(frames), shuffle=False)
    batch = next(iter(dataloader))

    def test_power_spectrum_model(self):
        torch.manual_seed(0)
        model = PowerSpectrumModel(
            unique_numbers=self.all_species,
            **self.ps_model_parameters,
        )
        energies = model(
            positions=self.batch.pos,
            cells=self.batch.cell,
            numbers=self.batch.numbers,
            edge_indices=self.batch.edge_index,
            edge_shifts=self.batch.edge_shift,
            ptr=self.batch.ptr,
        )
        forces = get_autograd_forces(energies, self.batch.pos)[0]

        ref_energies = torch.load("./tests/data/hea_bulk_test_ps_energies.pt")
        ref_forces = torch.load("./tests/data/hea_bulk_test_ps_forces.pt")
        assert torch.allclose(energies, ref_energies)
        assert torch.allclose(forces, ref_forces)

    def test_bpps_model(self):
        torch.manual_seed(0)
        model = BPPSModel(
            unique_numbers=self.all_species,
            **self.ps_model_parameters,
        )
        energies = model(
            positions=self.batch.pos,
            cells=self.batch.cell,
            numbers=self.batch.numbers,
            edge_indices=self.batch.edge_index,
            edge_shifts=self.batch.edge_shift,
            ptr=self.batch.ptr,
        )
        forces = get_autograd_forces(energies, self.batch.pos)[0]

        ref_energies = torch.load("./tests/data/hea_bulk_test_bpps_energies.pt")
        ref_forces = torch.load("./tests/data/hea_bulk_test_bpps_forces.pt")
        assert torch.allclose(energies, ref_energies)
        assert torch.allclose(forces, ref_forces)

    # def test_alchemical_model(self):
    #     torch.manual_seed(0)
    #     model = AlchemicalModel(
    #         unique_numbers=self.all_species,
    #         **self.alchemical_model_parameters,
    #     )
    #     positions, cells, numbers, edge_indices, edge_shifts = extract_batch_data(
    #         self.batch
    #     )
    #     energies = model(positions, cells, numbers, edge_indices, edge_shifts)
    #     forces = get_autograd_forces(energies, positions)

    #     ref_energies = torch.load("./tests/data/hea_bulk_test_alchemical_energies.pt")
    #     ref_forces = torch.load("./tests/data/hea_bulk_test_alchemical_forces.pt")
    #     assert torch.allclose(energies, ref_energies)
    #     assert torch.allclose(torch.cat(forces), torch.cat(ref_forces))
