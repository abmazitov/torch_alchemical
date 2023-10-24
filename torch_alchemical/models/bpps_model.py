import metatensor
import numpy as np
import torch

from typing import Union

from torch_alchemical.nn import (
    Linear,
    PowerSpectrumFeatures,
    SiLU,
)
from torch_alchemical.utils import get_compositions_from_numbers


class BPPS(torch.nn.Module):
    def __init__(
        self,
        hidden_sizes: int,
        output_size: int,
        unique_numbers: Union[list, np.ndarray],
        cutoff: float,
        basis_cutoff_power_spectrum: float,
        radial_basis_type: str,
        normalize_layers: bool = False,
        basis_normalization_factor: float = None,
        trainable_basis: bool = True,
        num_pseudo_species: int = None,
        device: torch.device = None,
    ):
        super().__init__()
        self.unique_numbers = unique_numbers
        self.normalize_layers = normalize_layers
        self.composition_layer = torch.nn.Linear(len(unique_numbers), output_size)
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
        ps_input_size = self.ps_features_layer.num_features
        layer_size = [ps_input_size] + hidden_sizes
        layers = []
        for layer_index in range(1, len(layer_size)):
            layers.append(
                Linear(
                    layer_size[layer_index - 1],
                    layer_size[layer_index],
                    bias=False,
                    normalize=self.normalize_layers,
                )
            )
            layers.append(SiLU(normalize=self.normalize_layers))
        layers.append(Linear(layer_size[-1], output_size))
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
        ps = self.ps_features_layer(
            positions, cells, numbers, edge_indices, edge_shifts
        )
        psl = self.ps_linear(ps)
        energies += (
            metatensor.sum_over_samples(psl.keys_to_samples("a_i"), ["center", "a_i"])
            .block()
            .values
        )
        psnn = self.nn(ps)
        energies += (
            metatensor.sum_over_samples(psnn.keys_to_samples("a_i"), ["center", "a_i"])
            .block()
            .values
        )
        return energies
