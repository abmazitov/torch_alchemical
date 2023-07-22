import torch


def get_compositions_from_numbers(numbers: torch.Tensor, unique_numbers: torch.Tensor):
    compositions = []
    for number in numbers:
        composition = torch.tensor(
            [(number == species).sum().item() for species in unique_numbers],
            dtype=torch.get_default_dtype(),
        )
        compositions.append(composition)
    return compositions
