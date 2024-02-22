from typing import Optional

import torch


def get_compositions_from_numbers(
    numbers: torch.Tensor,
    unique_numbers: list[int],
    batch: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> list[torch.Tensor]:
    compositions: list[torch.Tensor] = []
    device = numbers.device
    _, counts = torch.unique(batch, return_counts=True)
    dtype = dtype if dtype is not None else torch.float64
    ptr = torch.cat([torch.tensor([0], device=device), torch.cumsum(counts, dim=0)])
    splitted_numbers = [numbers[ptr[i] : ptr[i + 1]] for i in range(len(ptr) - 1)]
    unique_numbers = torch.tensor(
        unique_numbers,
        dtype=splitted_numbers[0].dtype,
        device=splitted_numbers[0].device,
    )
    for number in splitted_numbers:
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
