from typing import List, Optional

import torch
from metatensor.torch import Labels

from torch_alchemical.nn import (
    LinearMap,
    MeshPotentialFeatures,
    PowerSpectrumFeatures,
    ReLU,
    LayerNorm
)
from torch_alchemical.utils import get_compositions_from_numbers


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
        charges_channels: Optional[int] = None,
        trainable_basis: Optional[bool] = False,
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
            charges_channels=charges_channels,
        )
        self.unique_numbers = unique_numbers
        self.register_buffer(
            "composition_weights", torch.zeros((output_size, len(unique_numbers)))
        )
        self.ps_features_layer = PowerSpectrumFeatures(
            all_species=unique_numbers,
            cutoff_radius=cutoff,
            basis_cutoff=basis_cutoff_power_spectrum,
            radial_basis_type=radial_basis_type,
            basis_scale=basis_scale,
            trainable_basis=trainable_basis,
        )
        layer_size_ps = [self._num_features_ps] + hidden_sizes_ps
        layer_size_mp = [self._num_features_mp] + hidden_sizes_mp
        layers_ps: List[torch.nn.Module] = self._create_linear_layers(
            layer_size_ps, output_size
        )
        layers_mp: List[torch.nn.Module] = self._create_linear_layers(
            layer_size_mp, output_size
        )
        if charges_channels is not None:
            layer_size_charges = [self._num_features_ps]  # + ...
            layers_charges: List[torch.nn.Module] = self._create_linear_layers(
                layer_size_charges, charges_channels
            )
            self.nn_charges = torch.nn.ModuleList(layers_charges)
        else:
            self.nn_charges = None
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

    def _get_features_ps(
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
        return ps

    def _get_features_mp(
        self,
        positions: torch.Tensor,
        cells: torch.Tensor,
        numbers: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_offsets: torch.Tensor,
        batch: torch.Tensor,
        charges: Optional[torch.Tensor] = None,
    ):
        mp = self.meshlode_features_layer(
            positions, cells, numbers, edge_indices, edge_offsets, batch, charges
        )
        if charges is None:
            mp = mp.keys_to_properties("neighbor_type")
        else:
            mp = mp.keys_to_properties("charges_channel")
        return mp

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
                    bias=True,
                )
            )
            layers.append(LayerNorm(layer_size[layer_index]))
            layers.append(ReLU())
        layers.append(
            LinearMap(
                keys=linear_layer_keys,
                in_features=layer_size[-1],
                out_features=output_size,
                bias=True,
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
        if self.nn_charges is not None:
            charges = self.nn_charges[0](features_ps)
            for layer in self.nn_charges[1:]:
                charges = layer(charges)
        else:
            charges = None
        features_mp = self._get_features_mp(
            positions, cells, numbers, edge_indices, edge_offsets, batch, charges
        )
        for layer in self.nn_ps:
            features_ps = layer(features_ps)
        for layer in self.nn_mp:
            features_mp = layer(features_mp)
        psnn_ps = features_ps.keys_to_samples("center_type")
        psnn_mp = features_mp.keys_to_samples("center_type")
        features_ps = psnn_ps.block().values
        features_mp = psnn_mp.block().values
        energies_ps = torch.zeros(
            len(torch.unique(batch)),
            1,
            device=features_ps.device,
            dtype=features_ps.dtype,
        )
        energies_mp = torch.zeros(
            len(torch.unique(batch)),
            1,
            device=features_mp.device,
            dtype=features_mp.dtype,
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
        compositions = torch.stack(
            get_compositions_from_numbers(
                numbers,
                self.unique_numbers,
                batch,
                self.composition_weights.dtype,
            )
        )
        energies = energies_ps + energies_mp

        energies = (
            energies
            + compositions @ self.composition_weights.T
        )
        return energies
