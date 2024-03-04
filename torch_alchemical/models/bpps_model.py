from typing import List, Optional

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
        basis_scale: Optional[float] = 3.0,
    ):
        super().__init__()
        self.unique_numbers = unique_numbers
        self.register_buffer(
            "composition_weights", torch.zeros((output_size, len(unique_numbers)))
        )
        self.register_buffer("normalization_factor", torch.tensor(1.0))
        self.register_buffer("energies_scale_factor", torch.tensor(1.0))
        self.ps_features_layer = PowerSpectrumFeatures(
            all_species=unique_numbers,
            cutoff_radius=cutoff,
            basis_cutoff=basis_cutoff_power_spectrum,
            radial_basis_type=radial_basis_type,
            basis_scale=basis_scale,
            trainable_basis=trainable_basis,
            normalize=normalize,
        )
        ps_input_size = self.ps_features_layer.num_features
        self.normalize = normalize
        if self.normalize:
            self.layer_norm = LayerNorm(ps_input_size)
        layer_size = [ps_input_size] + hidden_sizes
        layers: List[torch.nn.Module] = []
        linear_layer_keys = Labels(
            names=["center_type"], values=torch.tensor(self.unique_numbers).view(-1, 1)
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

    def set_composition_weights(
        self,
        composition_weights: torch.Tensor,
    ):
        if composition_weights.shape != self.composition_weights.shape:  # type: ignore
            raise ValueError(
                "The shape of the composition weights does not match "
                + f"the expected shape {composition_weights.shape}."
            )
        self.composition_weights = composition_weights

    def set_normalization_factor(self, normalization_factor: torch.Tensor):
        self.normalization_factor = normalization_factor

    def set_energies_scale_factor(self, energies_scale_factor: torch.Tensor):
        self.energies_scale_factor = energies_scale_factor

    def set_basis_normalization_factor(self, basis_normalization_factor: torch.Tensor):
        self.ps_features_layer.spex_calculator.normalization_factor = 1.0 / torch.sqrt(
            basis_normalization_factor
        )
        self.ps_features_layer.spex_calculator.normalization_factor_0 = (
            1.0 / basis_normalization_factor ** (3 / 4)
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
        for layer in self.nn:
            ps = layer(ps)
        psnn = ps.keys_to_samples("center_type")
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
            energies = energies / self.normalization_factor
        if self.training:
            return energies
        else:
            compositions = torch.stack(
                get_compositions_from_numbers(
                    numbers,
                    self.unique_numbers,
                    batch,
                    self.composition_weights.dtype,
                )
            )
            energies = (
                energies * self.energies_scale_factor
                + compositions @ self.composition_weights.T
            )
            return energies
