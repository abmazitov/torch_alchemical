import torch


def get_compositions_from_numbers(
    numbers: torch.Tensor, unique_numbers: torch.Tensor, ptr: torch.Tensor
):
    compositions = []
    for i in range(len(ptr) - 1):
        number = numbers[ptr[i] : ptr[i + 1]]
        composition = torch.tensor(
            [(number == species).sum().item() for species in unique_numbers],
            dtype=torch.get_default_dtype(),
            device=number.device,
        )
        compositions.append(composition)
    return compositions
