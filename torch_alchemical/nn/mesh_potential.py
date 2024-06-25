from typing import List, Optional

import metatensor.torch
import torch
from meshlode.metatensor import MeshEwaldPotential
from metatensor.torch import Labels, TensorMap


from ..utils import get_metatensor_systems
from ..utils import split_edges_by_batch

class MeshPotentialFeatures(torch.nn.Module):
    def __init__(
        self,
        sr_cutoff: float,
        atomic_smearing: Optional[float] = None,
        mesh_spacing: Optional[float] = None,
        interpolation_order: Optional[int] = 4,
        subtract_self: Optional[bool] = True,
        all_types: Optional[List[int]] = None,
        subtract_interior: Optional[bool] = False,
        charges_channels: Optional[int] = None,
        exponent: Optional[torch.Tensor] = torch.tensor(1.0, dtype=torch.float64),
    ):
        super().__init__()
        self.calcultor = MeshEwaldPotential(
            sr_cutoff= sr_cutoff,
            atomic_smearing=atomic_smearing,
            mesh_spacing=mesh_spacing,
            interpolation_order=interpolation_order,
            subtract_self=subtract_self,
            all_types=all_types,
            subtract_interior=subtract_interior,
            exponent=exponent,
        )

        # In MeshLODE we create one subgrid per atomic type
        if charges_channels is not None:
            self._num_features_per_atom = charges_channels
        else:
            self._num_features_per_atom = len(all_types)

    def forward(
        self,
        positions: torch.Tensor,
        cells: torch.Tensor,
        numbers: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_offsets: torch.Tensor,
        batch: torch.Tensor,
        charges: Optional[torch.Tensor] = None,
    ) -> TensorMap:
        systems = get_metatensor_systems(
            batch=batch,
            species=numbers,
            positions=positions,
            cells=cells,
        )
        if charges is not None:
            labels = [
                Labels(names=["structure"], values=torch.tensor([[i]]))
                for i in range(len(systems))
            ]
            list_of_charges = metatensor.torch.operations.split(
                charges, axis="samples", grouped_labels=labels
            )
            for charge, system in zip(list_of_charges, systems):
                charge = charge.keys_to_samples("center_type").block().values
                samples = metatensor.torch.Labels(
                    "atom", torch.arange(len(system)).reshape(-1, 1)
                ).to(charge.device)
                properties = metatensor.torch.Labels(
                    "charge", torch.arange(charge.shape[1]).reshape(-1, 1)
                ).to(charge.device)

                charges_block = metatensor.torch.TensorBlock(
                    samples=samples,
                    components=[],
                    properties=properties,
                    values=charge,
                )

                system.add_data("charges", charges_block)
        edge_indices_batched, edge_offsets_batched = split_edges_by_batch(
            edge_indices, edge_offsets, batch
        )
        return self.calcultor.compute(systems)#, edge_indices_batched, edge_offsets_batched)

    @property
    def num_features(self) -> int:

        return self._num_features_per_atom
