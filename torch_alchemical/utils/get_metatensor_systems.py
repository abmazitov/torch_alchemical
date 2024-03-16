from typing import List

from metatensor.torch.atomistic import System
import torch


def _find_change_indices(tensor: torch.Tensor) -> torch.Tensor:
    """Find indices where values within a contiguous tensor change."""
    return (torch.where(tensor[1:] != tensor[:-1])[0] + 1).to("cpu")


def get_metatensor_systems(
    batch: torch.Tensor,
    species: torch.Tensor,
    positions: torch.Tensor,
    cells: torch.Tensor,
) -> List[System]:
    """Convert arrays to meshlode systems based on contiguous indices in ``batch``."""

    # Find the indices where the batch index changes
    change_indices = _find_change_indices(batch)

    species_split = torch.tensor_split(species, change_indices)
    positions_split = torch.tensor_split(positions, change_indices, dim=0)
    cells_split = cells.reshape(-1, 3, 3)

    systems: List[System] = []
    for s, p, c in zip(species_split, positions_split, cells_split):
        systems.append(System(types=s, positions=p, cell=c))

    return systems
