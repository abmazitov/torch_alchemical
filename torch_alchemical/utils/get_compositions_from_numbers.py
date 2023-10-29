import torch
from typing import Union


def get_compositions_from_numbers(
    numbers: Union[torch.Tensor, list[torch.Tensor]],
    unique_numbers: torch.Tensor,
    ptr: torch.Tensor = None,
):
    compositions = []
    if isinstance(numbers, torch.Tensor):
        assert ptr is not None
        numbers = [numbers[ptr[i] : ptr[i + 1]] for i in range(len(ptr) - 1)]
    for number in numbers:
        composition = torch.tensor(
            [(number == species).sum().item() for species in unique_numbers],
            dtype=torch.get_default_dtype(),
            device=number.device,
        )
        compositions.append(composition)
    return compositions
