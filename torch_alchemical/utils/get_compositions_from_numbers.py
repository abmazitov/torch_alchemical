import torch
from typing import Union, Optional


def get_compositions_from_numbers(
    numbers: Union[torch.Tensor, list[torch.Tensor]],
    unique_numbers: list[int],
    ptr: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
) -> list[torch.Tensor]:
    dtype = dtype if dtype is not None else torch.float64
    compositions: list[torch.Tensor] = []
    if isinstance(numbers, torch.Tensor):
        assert ptr is not None
        numbers = [numbers[ptr[i] : ptr[i + 1]] for i in range(len(ptr) - 1)]
    unique_numbers = torch.tensor(
        unique_numbers, dtype=numbers[0].dtype, device=numbers[0].device
    )
    for number in numbers:
        composition = torch.zeros(
            len(unique_numbers),
            dtype=dtype,
            device=number.device,
        )
        elements, counts = torch.unique(number, return_counts=True)
        index = torch.eq(elements[:, None], unique_numbers)
        mask = torch.nonzero(index)[:, 1]
        composition[mask] = counts.to(dtype)
        compositions.append(composition)
    return compositions
