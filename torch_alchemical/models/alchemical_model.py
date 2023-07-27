import equistore
import numpy as np
import torch

from typing import Union

from torch_alchemical.nn import (
    Linear,
    RadialSpectrumFeatures,
    PowerSpectrumFeatures,
    SiLU,
)
from torch_alchemical.utils import get_compositions_from_numbers


class AlchemicalModel(torch.nn.Module):
    def __init__(
        self,
        hidden_sizes: int,
        output_size: int,
        unique_numbers: Union[list, np.ndarray],
        cutoff: float,
        basis_cutoff_radial_spectrum: float,
        basis_cutoff_power_spectrum: float,
        num_pseudo_species: int = None,
        device: torch.device = None,
    ):
        super().__init__()
        self.unique_numbers = unique_numbers
        self.composition_layer = torch.nn.Linear(len(unique_numbers), output_size)
        self.rs_features_layer = RadialSpectrumFeatures(
            unique_numbers,
            cutoff,
            basis_cutoff_radial_spectrum,
            device,
        )
        self.ps_features_layer = PowerSpectrumFeatures(
            unique_numbers,
            cutoff,
            basis_cutoff_power_spectrum,
            num_pseudo_species,
            device,
        )
        rs_input_size = self.rs_features_layer.num_features
        ps_input_size = self.ps_features_layer.num_features
        self.rs_linear = Linear(rs_input_size, output_size)
        self.ps_linear = Linear(ps_input_size, output_size)
        nn_layers_size = [ps_input_size] + hidden_sizes
        layers = []
        for layer_index in range(1, len(nn_layers_size)):
            layers.append(
                Linear(nn_layers_size[layer_index - 1], nn_layers_size[layer_index])
            )
            layers.append(SiLU())
        layers.append(Linear(nn_layers_size[-1], output_size))
        self.nn = torch.nn.Sequential(*layers)

    def forward(
        self,
        positions: list[torch.Tensor],
        cells: list[torch.Tensor],
        numbers: list[torch.Tensor],
        edge_indices: list[torch.Tensor],
        edge_shifts: list[torch.Tensor],
    ):
        compositions = torch.stack(
            get_compositions_from_numbers(numbers, self.unique_numbers)
        )
        energies = self.composition_layer(compositions)
        rs = self.rs_features_layer(
            positions, cells, numbers, edge_indices, edge_shifts
        )
        ps = self.ps_features_layer(
            positions, cells, numbers, edge_indices, edge_shifts
        )
        rsl = self.rs_linear(rs)
        psl = self.ps_linear(ps)
        psnn = self.nn(ps)

        for tensormap in [rsl, psl, psnn]:
            energies += (
                equistore.sum_over_samples(
                    tensormap.keys_to_samples("a_i"), ["center", "a_i"]
                )
                .block()
                .values
            )

        return energies
