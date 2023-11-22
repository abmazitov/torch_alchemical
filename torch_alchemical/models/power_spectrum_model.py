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
        hidden_sizes: list[int],
        output_size: int,
        unique_numbers: Union[list, np.ndarray],
        cutoff: float,
        basis_cutoff_power_spectrum: float,
        radial_basis_type: str,
        energies_scale_factor: float = 1.0,
        basis_normalization_factor: float = None,
        trainable_basis: bool = True,
        num_pseudo_species: int = None,
        device: torch.device = None,
    ):
        super().__init__()
        self.unique_numbers = unique_numbers
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
        compositions = torch.stack(
            get_compositions_from_numbers(
                numbers, self.unique_numbers, batch, self.composition_layer.weight.dtype
            )
        )
        energies = self.composition_layer(compositions)
        ps = self.ps_features_layer(
            positions, cells, numbers, edge_indices, edge_offsets, batch
        )
        psl = self.ps_linear(ps).keys_to_samples("a_i")
        energies_psl = torch.zeros_like(energies)
        energies_psl.index_add_(
            dim=0,
            index=psl.block().samples.column("structure"),
            source=psl.block().values,
        )
        for layer in self.nn:
            ps = layer(ps)
        psnn = ps.keys_to_samples("a_i")
        energies_psnn = torch.zeros_like(energies)
        energies_psnn.index_add_(
            dim=0,
            index=psnn.block().samples.column("structure"),
            source=psnn.block().values,
        )
        energies += (energies_psl + energies_psnn) * self.energies_scale_factor

        return energies
