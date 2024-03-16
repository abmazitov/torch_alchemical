from typing import List, Optional

import torch
from metatensor.torch.operations import join

from torch_alchemical.nn import MeshPotentialFeatures
from torch_alchemical.nn import LayerNorm, LinearMap, PowerSpectrumFeatures, SiLU
from torch_alchemical.utils import get_compositions_from_numbers
from metatensor.torch import Labels, TensorMap, TensorBlock


class BPPSLodeModel(torch.nn.Module):
    def __init__(
        self,
        hidden_sizes_ps: List[int],
        hidden_sizes_mp: List[int],
        output_size: int,
        unique_numbers: List[int],
        cutoff: float,
        basis_cutoff_power_spectrum: float,
        radial_basis_type: str,
        lode_atomic_smearing: float,
        trainable_basis: Optional[bool] = True,
        normalize: Optional[bool] = True,
        basis_scale: Optional[float] = 3.0,
        lode_mesh_spacing: Optional[float] = None,
        lode_interpolation_order: Optional[int] = 4,
        lode_subtract_self: Optional[bool] = False,
    ):

        # Call parent `__init__` after we initlize the MeshPotentialFeatures instance to
        # have a working `_num_features` property.
        super().__init__()

        self.meshlode_features_layer = MeshPotentialFeatures(
            atomic_smearing=lode_atomic_smearing,
            mesh_spacing=lode_mesh_spacing,
            interpolation_order=lode_interpolation_order,
            subtract_self=lode_subtract_self,
            all_types=unique_numbers,
        )
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
        self.normalize = normalize
        if self.normalize:
            self.layer_norm_ps = LayerNorm(self._num_features_ps)
            self.layer_norm_mp = LayerNorm(self._num_features_mp)
        layer_size_ps = [self._num_features_ps] + hidden_sizes_ps
        layer_size_mp = [self._num_features_mp] + hidden_sizes_mp
        layers_ps: List[torch.nn.Module] = self._create_linear_layers(layer_size_ps, output_size)
        layers_mp: List[torch.nn.Module] = self._create_linear_layers(layer_size_mp, output_size)
        self.nn_ps = torch.nn.ModuleList(layers_ps)
        self.nn_mp = torch.nn.ModuleList(layers_mp)
    
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
    
    def _get_features_ps(
        self,
        positions: torch.Tensor,
        cells: torch.Tensor,
        numbers: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_offsets: torch.Tensor,
        batch: torch.Tensor,
    ):
        return self.ps_features_layer(
            positions, cells, numbers, edge_indices, edge_offsets, batch
        )
    
    def _get_features_mp(
        self,
        positions: torch.Tensor,
        cells: torch.Tensor,
        numbers: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_offsets: torch.Tensor,
        batch: torch.Tensor,
    ):
        return self.meshlode_features_layer(
            positions, cells, numbers, edge_indices, edge_offsets, batch
        )

    @property
    def _num_features_ps(self):
        return self.ps_features_layer.num_features
    
    @property
    def _num_features_mp(self):
        return self.meshlode_features_layer.num_features
    
    def _create_linear_layers(self, layer_size: List[int], output_size: int):
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
        return layers

    def forward(
        self,
        positions: torch.Tensor,
        cells: torch.Tensor,
        numbers: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_offsets: torch.Tensor,
        batch: torch.Tensor,
    ):
        features_ps = self._get_features_ps(
            positions, cells, numbers, edge_indices, edge_offsets, batch
        )
        features_mp = self._get_features_mp(
            positions, cells, numbers, edge_indices, edge_offsets, batch
        )

        if self.normalize:
            #print("Normalizing")
            #print(features_mp)
            #print(features_ps)
            #print(self.layer_norm_ps)
            #print(self.layer_norm_mp)
            #print(self.meshlode_features_layer.num_features)
            #print(self.ps_features_layer.num_features)
            features_ps = self.layer_norm_ps(features_ps)
            features_mp = self.layer_norm_mp(features_mp)
        for layer in self.nn_ps:
            features_ps = layer(features_ps)
        for layer in self.nn_mp:
            features_mp = layer(features_mp)
        psnn_ps = features_ps.keys_to_samples("center_type")
        psnn_mp = features_mp.keys_to_samples("center_type")
        features_ps = psnn_ps.block().values
        features_mp = psnn_mp.block().values
        energies_ps = torch.zeros(
            len(torch.unique(batch)), 1, device=features_ps.device, dtype=features_ps.dtype
        )
        energies_mp = torch.zeros(
            len(torch.unique(batch)), 1, device=features_mp.device, dtype=features_mp.dtype
        )
        energies_ps.index_add_(
            dim=0,
            index=batch,
            source=features_ps,
        )
        energies_mp.index_add_(
            dim=0,
            index=batch,
            source=features_mp,
        )
        if self.normalize:
            energies_ps = energies_ps / self.normalization_factor
            energies_mp = energies_mp / self.normalization_factor
        if self.training:
            return energies_ps + energies_mp
        else:
            compositions = torch.stack(
                get_compositions_from_numbers(
                    numbers,
                    self.unique_numbers,
                    batch,
                    self.composition_weights.dtype,
                )
            )
            energies_ps = (
                energies_ps * self.energies_scale_factor
                + compositions @ self.composition_weights.T
            )
            energies_mp = (
                energies_mp * self.energies_scale_factor
                + compositions @ self.composition_weights.T
            )
            return energies_ps + energies_mp
