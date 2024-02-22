from typing import Union

import numpy as np
import torch

from torch_alchemical.nn import LayerNorm, Linear, PowerSpectrumFeatures, SiLU
from torch_alchemical.utils import get_compositions_from_numbers


class PowerSpectrumModel(torch.nn.Module):
    def __init__(
        self,
        hidden_sizes: list[int],
        output_size: int,
        unique_numbers: Union[list, np.ndarray],
        cutoff: float,
        basis_cutoff_power_spectrum: float,
        radial_basis_type: str,
        num_pseudo_species: int = None,
        trainable_basis: bool = True,
        normalize: bool = True,
        basis_normalization_factor: float = None,
        basis_scale: float = 3.0,
        energies_scale_factor: float = 1.0,
        average_number_of_atoms: float = 1.0,
    ):
        super().__init__()
        if isinstance(unique_numbers, np.ndarray):
            unique_numbers = unique_numbers.tolist()
        self.unique_numbers = unique_numbers
        self.energies_scale_factor = energies_scale_factor
        self.average_number_of_atoms = average_number_of_atoms
        self.composition_layer = torch.nn.Linear(
            len(unique_numbers), output_size, bias=False
        )
        self.ps_features_layer = PowerSpectrumFeatures(
            all_species=unique_numbers,
            cutoff_radius=cutoff,
            basis_cutoff=basis_cutoff_power_spectrum,
            radial_basis_type=radial_basis_type,
            basis_normalization_factor=basis_normalization_factor,
            basis_scale=basis_scale,
            trainable_basis=trainable_basis,
            num_pseudo_species=num_pseudo_species,
        )
        ps_input_size = self.ps_features_layer.num_features
        self.normalize = normalize
        if self.normalize:
            self.layer_norm = LayerNorm(ps_input_size)
        self.ps_linear = Linear(ps_input_size, output_size, bias=False)
        layer_size = [ps_input_size] + hidden_sizes
        layers = []
        for layer_index in range(1, len(layer_size)):
            layers.append(
                Linear(layer_size[layer_index - 1], layer_size[layer_index], bias=False)
            )
            layers.append(SiLU())
        layers.append(Linear(layer_size[-1], output_size, bias=False))
        self.nn = torch.nn.ModuleList(layers)

    def forward(
        self,
        positions: torch.Tensor,
        cells: torch.Tensor,
        numbers: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_offsets: torch.Tensor,
        batch: torch.Tensor,
    ):
        ps = self.ps_features_layer(
            positions, cells, numbers, edge_indices, edge_offsets, batch
        )
        if self.normalize:
            ps = self.layer_norm(ps)
        psl = self.ps_linear(ps).keys_to_samples("a_i")
        for layer in self.nn:
            ps = layer(ps)
        psnn = ps.keys_to_samples("a_i")

        psl_features = psl.block().values
        psnn_features = psnn.block().values
        energies = torch.zeros(
            len(torch.unique(batch)),
            1,
            device=psl_features.device,
            dtype=psl_features.dtype,
        )
        energies.index_add_(
            dim=0,
            index=batch,
            source=psl_features,
        )
        energies.index_add_(
            dim=0,
            index=batch,
            source=psnn_features,
        )
        if self.normalize:
            energies = energies / self.average_number_of_atoms
        if self.training:
            return energies
        else:
            compositions = torch.stack(
                get_compositions_from_numbers(
                    numbers,
                    self.unique_numbers,
                    batch,
                    self.composition_layer.weight.dtype,
                )
            )
            energies = energies * self.energies_scale_factor + self.composition_layer(
                compositions
            )
            return energies
