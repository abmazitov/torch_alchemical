import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_alchemical.utils import get_compositions_from_numbers


class CompositionFeatures(BaseTransform):
    def __init__(
        self,
        unique_numbers: list[int],
    ):
        self.unique_numbers = unique_numbers

    def __call__(
        self,
        data: Data,
    ) -> Data:
        assert hasattr(data, "numbers")
        numbers, counts = torch.unique(data.numbers, return_counts=True)
        composition = get_compositions_from_numbers([numbers], self.unique_numbers)[0]
        data.composition = composition.view(1, -1)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(atomic_numbers={self.unique_numbers})"
