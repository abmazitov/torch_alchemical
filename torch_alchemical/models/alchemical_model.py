import metatensor
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
        radial_basis_type: str,
        basis_normalization_factor: float = None,
        trainable_basis: bool = True,
        num_pseudo_species: int = None,
        device: torch.device = None,
    ):
        super().__init__()
        self.unique_numbers = unique_numbers
        self.composition_layer = torch.nn.Linear(len(unique_numbers), output_size)
        self.rs_features_layer = RadialSpectrumFeatures(
            all_species=unique_numbers,
            cutoff_radius=cutoff,
            basis_cutoff=basis_cutoff_radial_spectrum,
            radial_basis_type=radial_basis_type,
            basis_normalization_factor=basis_normalization_factor,
            trainable_basis=trainable_basis,
            device=device,
        )
        self.ps_features_layer = PowerSpectrumFeatures(
            all_species=unique_numbers,
            cutoff_radius=cutoff,
            basis_cutoff=basis_cutoff_power_spectrum,
            radial_basis_type=radial_basis_type,
            basis_normalization_factor=basis_normalization_factor,
            trainable_basis=trainable_basis,
            num_pseudo_species=num_pseudo_species,
            device=device,
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
        positions: torch.Tensor,
        cells: torch.Tensor,
        numbers: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_shifts: torch.Tensor,
        ptr: torch.Tensor,
    ):
        compositions = torch.stack(
            get_compositions_from_numbers(numbers, self.unique_numbers, ptr)
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
                metatensor.sum_over_samples(
                    tensormap.keys_to_samples("a_i"), ["center", "a_i"]
                )
                .block()
                .values
            )

        return energies
