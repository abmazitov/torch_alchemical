import torch

from torch_alchemical.data import AtomisticDataset


def get_composition_weights(
    dataset, compositions: torch.Tensor
) -> torch.Tensor:
    energies = torch.cat([data.energies.view(1, -1) for data in dataset], dim=0)
    weights = torch.linalg.lstsq(compositions, energies).solution
    return weights.T


def get_average_number_of_atoms(dataset: AtomisticDataset) -> torch.Tensor:
    return torch.mean(
        torch.tensor([data.num_nodes for data in dataset]).to(torch.get_default_dtype())
    )
