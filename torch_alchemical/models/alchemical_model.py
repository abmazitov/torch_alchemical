import numpy as np
import torch

from typing import Union

from torch_alchemical.nn import (
    LayerNorm,
    AlchemicalContraction,
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
        basis_cutoff_power_spectrum: float,
        radial_basis_type: str,
        energies_scale_factor: float = 1.0,
        normalize: bool = True,
        average_number_of_atoms: float = None,
        basis_normalization_factor: float = None,
        trainable_basis: bool = True,
        num_pseudo_species: int = None,
        device: torch.device = None,
    ):
        super().__init__()
        if isinstance(unique_numbers, np.ndarray):
            unique_numbers = unique_numbers.tolist()
        self.unique_numbers = unique_numbers
        self.normalize = normalize
        self.num_pseudo_species = num_pseudo_species
        self.average_number_of_atoms = average_number_of_atoms
        self.energies_scale_factor = energies_scale_factor
        self.composition_layer = torch.nn.Linear(
            len(unique_numbers), output_size, bias=False
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
        ps_input_size = self.ps_features_layer.num_features
        if self.normalize:
            self.layer_norm = LayerNorm(ps_input_size)
            self.contraction_layer = (
                self.ps_features_layer.spex_calculator.vector_expansion_calculator.radial_basis_calculator.combination_matrix.linear_layer
            )
        else:
            self.contraction_layer = (
                self.ps_features_layer.spex_calculator.vector_expansion_calculator.radial_basis_calculator.combination_matrix
            )
        layer_size = [ps_input_size] + hidden_sizes
        layers = []
        for layer_index in range(1, len(layer_size)):
            layers.append(
                AlchemicalContraction(
                    unique_numbers=unique_numbers,
                    contraction_matrix=self.contraction_layer.weight,
                    in_features=layer_size[layer_index - 1],
                    out_features=layer_size[layer_index],
                    bias=False,
                )
            )
            layers.append(SiLU())
        layers.append(
            AlchemicalContraction(
                unique_numbers=unique_numbers,
                contraction_matrix=self.contraction_layer.weight,
                in_features=layer_size[-1],
                out_features=output_size,
                bias=False,
            )
        )
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
        for layer in self.nn:
            ps = layer(ps)
        psnn = ps.keys_to_samples("a_i")
        features = psnn.block().values
        energies = torch.zeros(
            len(torch.unique(batch)), 1, device=features.device, dtype=features.dtype
        )
        energies.index_add_(
            dim=0,
            index=batch,
            source=features,
        )
        if self.normalize:
            energies = energies / torch.sqrt(self.num_pseudo_species)
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
