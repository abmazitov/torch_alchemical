import torch
from torch_alchemical.data import AtomisticDataset


def get_composition_weights(
    dataset: AtomisticDataset, compositions: torch.Tensor
) -> torch.Tensor:
    energies = torch.cat([data.energies.view(1, -1) for data in dataset], dim=0)
    weights = torch.linalg.lstsq(compositions, energies).solution
    return weights.T


def get_energies_scale_factor(
    dataset: AtomisticDataset,
    compositions: torch.Tensor,
    composition_weights: torch.Tensor,
    use_second_moment: bool = True,
) -> torch.Tensor:
    energies = torch.cat([data.energies.view(1, -1) for data in dataset], dim=0)
    composition_energies = compositions @ composition_weights.T
    if use_second_moment:
        scale_factor = torch.sqrt(torch.mean((energies - composition_energies) ** 2))
    else:
        scale_factor = torch.std(energies - composition_energies)
    return scale_factor


def rescale_energies_and_forces(
    dataset: AtomisticDataset,
    compositions: torch.Tensor,
    composition_weights: torch.Tensor,
    scale_factor: torch.Tensor,
) -> None:
    composition_energies = compositions @ composition_weights.T
    for i, data in enumerate(dataset):
        data.energies = (data.energies - composition_energies[i]) / scale_factor
        data.forces = data.forces / scale_factor
    print("Training energies and forces are shifted and rescaled")


def get_average_number_of_atoms(dataset: AtomisticDataset) -> torch.Tensor:
    return torch.mean(
        torch.tensor([data.num_nodes for data in dataset]).to(torch.get_default_dtype())
    )
