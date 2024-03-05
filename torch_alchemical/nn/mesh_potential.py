from typing import List, Optional

import torch
from meshlode.metatensor import MeshPotential
from metatensor.torch import TensorMap

from ..utils import get_metatensor_systems


class MeshPotentialFeatures(torch.nn.Module):
    def __init__(
        self,
        atomic_smearing: float,
        mesh_spacing: Optional[float] = None,
        interpolation_order: Optional[int] = 4,
        subtract_self: Optional[bool] = False,
        all_types: Optional[List[int]] = None,
    ):
        super().__init__()
        self.calcultor = MeshPotential(
            atomic_smearing=atomic_smearing,
            mesh_spacing=mesh_spacing,
            interpolation_order=interpolation_order,
            subtract_self=subtract_self,
            all_types=all_types,
        )

        # In MeshLODE we create one subgrid per atomic type
        self._num_features_per_atom = len( all_types)

    def forward(
        self,
        positions: torch.Tensor,
        cells: torch.Tensor,
        numbers: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_offsets: torch.Tensor,
        batch: torch.Tensor,
    ) -> TensorMap:
        systems = get_metatensor_systems(
            batch=batch,
            species=numbers,
            positions=positions,
            cells=cells,
        )

        return self.calcultor.compute(systems)

    @property
    def num_features(self) -> int:
        # Is this total or per atom? if total multiply
        return self._num_features_per_atom
