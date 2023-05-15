import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class CompositionFeatures(BaseTransform):
    def __init__(
        self,
        atomic_numbers: list[int],
    ):
        self.atomic_numbers = atomic_numbers

    def __call__(
        self,
        data: Data,
    ) -> Data:
        assert hasattr(data, "numbers")
        numbers, counts = torch.unique(data.numbers, return_counts=True)
        composition = torch.tensor(
            [
                counts[numbers == number] if number in numbers else 0
                for number in self.atomic_numbers
            ],
            dtype=torch.get_default_dtype(),
        )
        data.composition = composition.view(1, -1)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(atomic_numbers={self.atomic_numbers})"
