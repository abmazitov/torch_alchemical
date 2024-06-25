import torch

from torch_alchemical.data import AtomisticDataset


def get_average_number_of_atoms(dataset: AtomisticDataset) -> torch.Tensor:
    return torch.mean(
        torch.tensor([data.num_nodes for data in dataset]).to(torch.get_default_dtype())
    )
