from typing import Optional

import metatensor.torch
import numpy as np
import torch
from metatensor.torch import Labels

from torch_alchemical.nn import (
    AlchemicalEmbedding,
    LayerNorm,
    LinearMap,
    PowerSpectrumFeatures,
    SiLU,
)
from torch_alchemical.utils import get_compositions_from_numbers


class AlchemicalModel(torch.nn.Module):
    def __init__(
        self,
        hidden_sizes: list[int],
        output_size: int,
        num_pseudo_species: int,
        unique_numbers: list[int],
        cutoff: float,
        basis_cutoff_power_spectrum: float,
        radial_basis_type: str,
        contract_center_species: Optional[bool] = False,
        trainable_basis: Optional[bool] = True,
        normalize: Optional[bool] = True,
        basis_normalization_factor: Optional[float] = None,
        basis_scale: Optional[float] = 3.0,
        energies_scale_factor: Optional[float] = 1.0,
        average_number_of_atoms: Optional[float] = 1.0,
    ):
        super().__init__()
        if isinstance(unique_numbers, np.ndarray):
            unique_numbers = unique_numbers.tolist()
        self.unique_numbers = unique_numbers
        self.normalize = normalize
        self.num_pseudo_species = num_pseudo_species
        self.average_number_of_atoms = average_number_of_atoms
        self.energies_scale_factor = energies_scale_factor
        self.contract_center_species = contract_center_species
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
        if self.normalize:
            self.layer_norm = LayerNorm(ps_input_size)
        vex_calculator = (
            self.ps_features_layer.spex_calculator.vector_expansion_calculator
        )
        contraction_layer = vex_calculator.radial_basis_calculator.combination_matrix
        device = contraction_layer.linear_layer.weight.device
        if self.contract_center_species:
            self.embedding = AlchemicalEmbedding(
                unique_numbers=unique_numbers,
                contraction_layer=contraction_layer,
            )
            linear_layer_keys = Labels(
                names=["b_i"],
                values=torch.arange(self.num_pseudo_species, device=device).view(-1, 1),
            )
        else:
            self.embedding = None  # type: ignore
            linear_layer_keys = Labels(
                names=["a_i"],
                values=torch.tensor(self.unique_numbers, device=device).view(-1, 1),
            )

        layer_size = [ps_input_size] + hidden_sizes

        layers: list[torch.nn.Module] = []

        for layer_index in range(1, len(layer_size)):
            layers.append(
                LinearMap(
                    keys=linear_layer_keys,
                    in_features=layer_size[layer_index - 1],
                    out_features=layer_size[layer_index],
                    bias=False,
                )
            )
            layers.append(SiLU())
        layers.append(
            LinearMap(
                keys=linear_layer_keys,
                in_features=layer_size[-1],
                out_features=1,
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
        if self.embedding is not None:
            ps = self.embedding(ps)
            ps = ps.keys_to_samples("a_i")
            ps = metatensor.torch.sum_over_samples(ps, "a_i")
        else:
            pass
        for layer in self.nn:
            ps = layer(ps)
        ps = ps.keys_to_samples(ps.keys.names)
        samples_names = ps.block().samples.names
        samples_names.remove("structure")
        ps = metatensor.torch.sum_over_samples(ps, samples_names)
        energies = ps.block().values
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
