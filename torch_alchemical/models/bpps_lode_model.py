from typing import List, Optional

import torch
from metatensor.operations import join

from torch_alchemical.nn import MeshPotentialFeatures

from . import BPPSModel


class BPPSLodeModel(BPPSModel):
    def __init__(
        self,
        hidden_sizes: List[int],
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

        self.meshlode_features_layer = MeshPotentialFeatures(
            all_species=unique_numbers,
            atomic_smearing=lode_atomic_smearing,
            mesh_spacing=lode_mesh_spacing,
            interpolation_order=lode_interpolation_order,
            subtract_self=lode_subtract_self,
        )

        # Call parent `__init__` after we initlize the MeshPotentialFeatures instance to
        # have a working `_num_features` property.
        super().__init__(
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            unique_numbers=unique_numbers,
            cutoff=cutoff,
            basis_cutoff_power_spectrum=basis_cutoff_power_spectrum,
            radial_basis_type=radial_basis_type,
            trainable_basis=trainable_basis,
            normalize=normalize,
            basis_scale=basis_scale,
        )

    def _get_features(
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

        mp = self.meshlode_features_layer(
            positions, cells, numbers, edge_indices, edge_offsets, batch
        )

        return join([ps, mp], axis="properties", remove_tensor_name=True)

    @property
    def _num_features(self):
        return (
            self.ps_features_layer.num_features
            + self.meshlode_features_layer.num_features
        )
