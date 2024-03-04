import json

import numpy as np
import torch
from ase.io import read
from torch_geometric.loader import DataLoader

from torch_alchemical.data import AtomisticDataset
from torch_alchemical.models import AlchemicalModel, BPPSModel
from torch_alchemical.transforms import NeighborList
from torch_alchemical.utils import get_autograd_forces

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


device = "cuda" if torch.cuda.is_available() else "cpu"
frames = read("./tests/data/hea_bulk_test_sample.xyz", index=":")
all_species = np.unique(np.hstack([frame.numbers for frame in frames])).tolist()
with open("./tests/configs/default_hypers_alchemical.json", "r") as f:
    hypers = json.load(f)
with open("./tests/configs/default_model_parameters.json", "r") as f:
    default_model_parameters = json.load(f)
transforms = [NeighborList(cutoff_radius=hypers["cutoff radius"])]
dataset = AtomisticDataset(
    frames, target_properties=["energies", "forces"], transforms=transforms
)
dataloader = DataLoader(dataset, batch_size=len(frames), shuffle=False)
batch = next(iter(dataloader)).to(device)


def test_bpps_model():
    torch.manual_seed(0)
    model = BPPSModel(
        unique_numbers=all_species,
        **default_model_parameters,
    ).to(device)
    with torch.no_grad():
        predictions = model(
            positions=batch.pos,
            cells=batch.cell,
            numbers=batch.numbers,
            edge_indices=batch.edge_index,
            edge_offsets=batch.edge_offsets,
            batch=batch.batch,
        )
        assert torch.allclose(
            predictions,
            torch.tensor([[-4.2060], [3.3690]]),
            atol=1e-4,
        )


def test_bpps_model_eval_mode():
    torch.manual_seed(0)
    model = BPPSModel(
        unique_numbers=all_species,
        **default_model_parameters,
    ).to(device)
    with torch.no_grad():
        model.eval()
        predictions = model(
            positions=batch.pos,
            cells=batch.cell,
            numbers=batch.numbers,
            edge_indices=batch.edge_index,
            edge_offsets=batch.edge_offsets,
            batch=batch.batch,
        )
        assert torch.allclose(
            predictions,
            torch.tensor([[-4.2060], [3.3690]]),
            atol=1e-4,
        )


def test_alchemical_model_with_contraction():
    torch.manual_seed(0)
    model = AlchemicalModel(
        unique_numbers=all_species,
        contract_center_species=True,
        num_pseudo_species=4,
        **default_model_parameters,
    ).to(device)
    with torch.no_grad():
        predictions = model(
            positions=batch.pos,
            cells=batch.cell,
            numbers=batch.numbers,
            edge_indices=batch.edge_index,
            edge_offsets=batch.edge_offsets,
            batch=batch.batch,
        )
        assert torch.allclose(
            predictions, torch.tensor([[-17.0218], [41.9509]]), atol=1e-4
        )


def test_alchemical_model_without_contraction():
    torch.manual_seed(0)
    model = AlchemicalModel(
        unique_numbers=all_species,
        contract_center_species=False,
        num_pseudo_species=4,
        **default_model_parameters,
    ).to(device)
    with torch.no_grad():
        predictions = model(
            positions=batch.pos,
            cells=batch.cell,
            numbers=batch.numbers,
            edge_indices=batch.edge_index,
            edge_offsets=batch.edge_offsets,
            batch=batch.batch,
        )
        assert torch.allclose(
            predictions, torch.tensor([[4.9560], [16.7199]]), atol=1e-4
        )


def test_alchemical_model_eval_mode():
    torch.manual_seed(0)
    model = AlchemicalModel(
        unique_numbers=all_species,
        contract_center_species=True,
        num_pseudo_species=4,
        **default_model_parameters,
    ).to(device)
    with torch.no_grad():
        model.eval()
        predictions = model(
            positions=batch.pos,
            cells=batch.cell,
            numbers=batch.numbers,
            edge_indices=batch.edge_index,
            edge_offsets=batch.edge_offsets,
            batch=batch.batch,
        )
        assert torch.allclose(
            predictions, torch.tensor([[-17.0218], [41.9509]]), atol=1e-4
        )


def test_model_autograd_forces():
    torch.manual_seed(0)
    model = AlchemicalModel(
        unique_numbers=all_species,
        contract_center_species=False,
        num_pseudo_species=4,
        **default_model_parameters,
    ).to(device)
    energies = model(
        positions=batch.pos,
        cells=batch.cell,
        numbers=batch.numbers,
        edge_indices=batch.edge_index,
        edge_offsets=batch.edge_offsets,
        batch=batch.batch,
    )
    autograd_forces = get_autograd_forces(energies, batch.pos)[0]
    target_forces = torch.load(
        "./tests/data/autograd_forces.pt",
    )
    assert torch.allclose(autograd_forces, target_forces, atol=1e-4)
