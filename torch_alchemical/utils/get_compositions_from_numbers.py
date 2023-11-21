import torch
from typing import Union, Optional


def get_compositions_from_numbers(
    numbers: Union[torch.Tensor, list[torch.Tensor]],
    unique_numbers: list[int],
    batch: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
) -> list[torch.Tensor]:
    compositions: list[torch.Tensor] = []
    if isinstance(numbers, torch.Tensor):
        assert batch is not None
        device = numbers.device
        _, counts = torch.unique(batch, return_counts=True)
    else:
        device = numbers[0].device
        counts = torch.tensor([len(number) for number in numbers], device=device)
    dtype = dtype if dtype is not None else torch.float64
    ptr = torch.cat([torch.tensor([0], device=device), torch.cumsum(counts, dim=0)])
    if isinstance(numbers, torch.Tensor):
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
