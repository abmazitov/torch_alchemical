from typing import List, Optional

import numpy as np
import torch
from metatensor.torch import Labels

from torch_alchemical.nn import LayerNorm, LinearMap, PowerSpectrumFeatures, SiLU
from torch_alchemical.utils import get_compositions_from_numbers


class BPPSModel(torch.nn.Module):
    def __init__(
        self,
        hidden_sizes: List[int],
        output_size: int,
        unique_numbers: List[int],
        cutoff: float,
        basis_cutoff_power_spectrum: float,
        radial_basis_type: str,
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
        )
        ps_input_size = self.ps_features_layer.num_features
        self.normalize = normalize
        if self.normalize:
            self.layer_norm = LayerNorm(ps_input_size)
        layer_size = [ps_input_size] + hidden_sizes
        layers: List[torch.nn.Module] = []
        linear_layer_keys = Labels(
            names=["a_i"], values=torch.tensor(self.unique_numbers).view(-1, 1)
        )
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
