import numpy as np
import torch

from typing import Union

from torch_alchemical.nn import (
    LayerNorm,
    AlchemicalEmbedding,
    MultiChannelGATConv,
    MultiChannelLinear,
    PowerSpectrumFeatures,
    SiLU,
)
from torch_alchemical.operations import sum_over_components
from torch_alchemical.utils import get_compositions_from_numbers


class AlchemicalGAT(torch.nn.Module):
    def __init__(
        self,
        conv_sizes: list[int],
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
            basis_scale=basis_scale,
            trainable_basis=trainable_basis,
            num_pseudo_species=num_pseudo_species,
            device=device,
        )
        ps_input_size = self.ps_features_layer.num_features
        if self.normalize:
            self.layer_norm = LayerNorm(ps_input_size)
        contraction_layer = (
            self.ps_features_layer.spex_calculator.vector_expansion_calculator.radial_basis_calculator.combination_matrix
        )
        self.embedding = AlchemicalEmbedding(
            unique_numbers=unique_numbers,
            num_pseudo_species=num_pseudo_species,
            contraction_layer=contraction_layer,
        )

        conv_layer_size = [ps_input_size] + conv_sizes
        conv_layers = []
        for layer_index in range(1, len(conv_layer_size)):
            conv_layers.append(
                MultiChannelGATConv(
                    in_channels=conv_layer_size[layer_index - 1],
                    out_channels=conv_layer_size[layer_index],
                    num_channels=self.num_pseudo_species,
                    bias=False,
                )
            )
        self.act = SiLU()
        self.conv = torch.nn.ModuleList(conv_layers)

        layer_size = [conv_sizes[-1]] + hidden_sizes
        layers = []
        for layer_index in range(1, len(layer_size)):
            layers.append(
                MultiChannelLinear(
                    in_features=layer_size[layer_index - 1],
                    out_features=layer_size[layer_index],
                    num_channels=self.num_pseudo_species,
                    bias=False,
                )
            )
        self.nn = torch.nn.ModuleList(layers)
        self.out_linear = MultiChannelLinear(
            in_features=layer_size[-1],
            out_features=output_size,
            num_channels=self.num_pseudo_species,
            bias=False,
        )

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
        ps = self.embedding(ps)
        ps = ps.keys_to_samples("a_i")
        for conv in self.conv:
            ps = conv(ps, edge_indices)
            ps = self.act(ps)
        for layer in self.nn:
            ps = layer(ps)
            ps = self.act(ps)
        ps = self.out_linear(ps)
        psnn = sum_over_components(ps)
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
            energies = energies / torch.sqrt(torch.tensor(self.num_pseudo_species))
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