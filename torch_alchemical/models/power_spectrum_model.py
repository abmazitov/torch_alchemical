import equistore
import numpy as np
import torch

from typing import Union

from torch_alchemical.nn import (
    Linear,
    PowerSpectrumFeatures,
    SiLU,
)
from torch_alchemical.utils import get_compositions_from_numbers


class PowerSpectrumModel(torch.nn.Module):
    def __init__(
        self,
        hidden_sizes: int,
        output_size: int,
        unique_numbers: Union[list, np.ndarray],
        cutoff: float,
        basis_cutoff_power_spectrum: float,
        num_pseudo_species: int = None,
        device: torch.device = None,
    ):
        super().__init__()
        self.unique_numbers = unique_numbers
        self.composition_layer = torch.nn.Linear(len(unique_numbers), output_size)
        self.ps_features_layer = PowerSpectrumFeatures(
            unique_numbers,
            cutoff,
            basis_cutoff_power_spectrum,
            num_pseudo_species,
            device,
        )
        ps_input_size = self.ps_features_layer.num_features
        self.ps_linear = Linear(ps_input_size, output_size)
        layer_size = [ps_input_size] + hidden_sizes
        layers = []
        for layer_index in range(1, len(layer_size)):
            layers.append(Linear(layer_size[layer_index - 1], layer_size[layer_index]))
            layers.append(SiLU())
        layers.append(Linear(layer_size[-1], output_size))
        self.nn = torch.nn.Sequential(*layers)

    def forward(
        self,
        positions: list[torch.Tensor],
        cells: list[torch.Tensor],
        numbers: list[torch.Tensor],
        edge_indices: list[torch.Tensor],
        edge_shifts: list[torch.Tensor],
        training=True,
    ):
        compositions = torch.stack(
            get_compositions_from_numbers(numbers, self.unique_numbers)
        )
        energies = self.composition_layer(compositions)
        ps = self.ps_features_layer(
            positions, cells, numbers, edge_indices, edge_shifts
        )
        psl = self.ps_linear(ps)
        energies += (
            equistore.sum_over_samples(psl.keys_to_samples("a_i"), ["center", "a_i"])
            .block()
            .values
        )
        psnn = self.nn(ps)
        energies += (
            equistore.sum_over_samples(psnn.keys_to_samples("a_i"), ["center", "a_i"])
            .block()
            .values
        )
        return energies
