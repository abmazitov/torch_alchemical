import numpy as np
import torch

from typing import Union, Optional

from torch_alchemical.nn import (
    LinearMap,
    PowerSpectrumFeatures,
    SiLU,
)
from torch_alchemical.utils import get_compositions_from_numbers


class BPPSModel(torch.nn.Module):
    def __init__(
        self,
        hidden_sizes: int,
        output_size: int,
        unique_numbers: Union[list, np.ndarray],
        cutoff: float,
        basis_cutoff_power_spectrum: float,
        radial_basis_type: str,
        basis_normalization_factor: float = None,
        trainable_basis: bool = True,
        num_pseudo_species: int = None,
        device: torch.device = None,
    ):
        super().__init__()
        if isinstance(unique_numbers, np.ndarray):
            unique_numbers = unique_numbers.tolist()
        self.unique_numbers = unique_numbers
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
        layer_size = [ps_input_size] + hidden_sizes
        layers = []
        for layer_index in range(1, len(layer_size)):
            layers.append(
                LinearMap(
                    keys=self.unique_numbers,
                    in_features=layer_size[layer_index - 1],
                    out_features=layer_size[layer_index],
                    bias=False,
                )
            )
            layers.append(SiLU())
        layers.append(
            LinearMap(
                keys=self.unique_numbers,
                in_features=layer_size[-1],
                out_features=output_size,
                bias=False,
            )
        )
        self.nn = torch.nn.ModuleList(layers)

    def forward(
        self,
        positions: Union[torch.Tensor, list[torch.Tensor]],
        cells: Union[torch.Tensor, list[torch.Tensor]],
        numbers: Union[torch.Tensor, list[torch.Tensor]],
        edge_indices: Union[torch.Tensor, list[torch.Tensor]],
        edge_shifts: Union[torch.Tensor, list[torch.Tensor]],
        ptr: Optional[torch.Tensor] = None,
    ):
        compositions = torch.stack(
            get_compositions_from_numbers(
                numbers, self.unique_numbers, ptr, self.composition_layer.weight.dtype
            )
        )
        energies = self.composition_layer(compositions)
        ps = self.ps_features_layer(
            positions, cells, numbers, edge_indices, edge_shifts, ptr
        )
        for layer in self.nn:
            ps = layer(ps)
        psnn = ps.keys_to_samples("a_i")
        energies.index_add_(
            dim=0,
            index=psnn.block().samples.column("structure"),
            source=psnn.block().values,
        )
        return energies
