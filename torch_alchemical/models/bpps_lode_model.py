from typing import List, Optional

import torch
from metatensor.torch import Labels

from torch_alchemical.nn import (
    LayerNorm,
    LinearMap,
    MeshPotentialFeatures,
    PowerSpectrumFeatures,
    GELU,
    GaussianFourierEmbeddingTensor,
)

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
        charges_channels: Optional[int] = None,
        trainable_basis: Optional[bool] = False,
        gaussian_fourier_embedding: Optional[bool] = False,
        gaussian_feature_scale: Optional[int] = 1,
        n_of_gaussian_features: Optional[int] = 128,
        basis_scale: Optional[float] = 3.0,
        lode_atomic_smearing: Optional[float] = None,
        lode_mesh_spacing: Optional[float] = None,
        lode_interpolation_order: Optional[int] = 4,
        lode_subtract_self: Optional[bool] = True,
        lode_subtract_interior: Optional[bool] = False,
        lode_exponent: Optional[torch.Tensor] = torch.tensor(1.0, dtype=torch.float64),
    ):

        # Call parent `__init__` after we initlize the MeshPotentialFeatures instance to
        # have a working `_num_features` property.
        super().__init__()

        self.meshlode_features_layer = MeshPotentialFeatures(
            sr_cutoff= cutoff,
            atomic_smearing=lode_atomic_smearing,
            mesh_spacing=lode_mesh_spacing,
            interpolation_order=lode_interpolation_order,
            subtract_self=lode_subtract_self,
            all_types=unique_numbers,
            subtract_interior=lode_subtract_interior,
            charges_channels=charges_channels,
            exponent=lode_exponent,
        )
        self.unique_numbers = unique_numbers
        self.register_buffer(
            "compositions_weights", torch.zeros((output_size, len(unique_numbers)))
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
        if gaussian_fourier_embedding:
            layer_size_mp = [2 * n_of_gaussian_features] + hidden_sizes_mp
        else:
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
        if gaussian_fourier_embedding:
            linear_layer_keys = Labels(
            names=["center_type"], values=torch.tensor(self.unique_numbers).view(-1, 1)
            )

            gaussian_fourier_embedding_layer = [GaussianFourierEmbeddingTensor(linear_layer_keys, len(self.unique_numbers), n_of_gaussian_features, gaussian_feature_scale)]
            # Add Gaussian Fourier Embedding Layer
            layers_mp = gaussian_fourier_embedding_layer + layers_mp
        self.nn_mp = torch.nn.ModuleList(layers_mp)

    def set_compositions_weights(
        self,
        compositions_weights: torch.Tensor,
    ):
        if compositions_weights.shape != self.compositions_weights.shape:  # type: ignore
            raise ValueError(
                "The shape of the composition weights does not match "
                + f"the expected shape {compositions_weights.shape}."
            )
        self.compositions_weights = compositions_weights

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
        layer_keys = Labels(
            names=["center_type"], values=torch.tensor(self.unique_numbers).view(-1, 1)
        )
        for layer_index in range(1, len(layer_size)):
            layers.append(
                LinearMap(
                    keys=layer_keys,
                    in_features=layer_size[layer_index - 1],
                    out_features=layer_size[layer_index],
                    bias=True,
                )
            )
            layers.append(LayerNorm(layer_keys, layer_size[layer_index]))
            layers.append(GELU())
        layers.append(
            LinearMap(
                keys=layer_keys,
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

        energies = energies_ps + energies_mp

        return energies
