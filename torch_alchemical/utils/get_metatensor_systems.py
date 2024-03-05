from typing import List

from metatensor.torch.atomistic import System
import torch


def _find_change_indices(tensor):
    """Find indices where values within a contiguous tensor change."""
    return [i for i in range(1, len(tensor)) if tensor[i] != tensor[i - 1]]


def get_metatensor_systems(
    batch: torch.tensor,
    species: torch.tensor,
    positions: torch.tensor,
    cells: torch.tensor,
) -> List[System]:
    """Convert arrays to meshlode systems based on contiguous indices in ``batch``."""

    # Find the indices where the batch index changes
    change_indices = _find_change_indices(batch)

    species_split = torch.tensor_split(species, change_indices)
    positions_split = torch.tensor_split(positions, change_indices, dim=0)
    cells_split = cells.reshape(-1, 3, 3)

    systems = []
    for s, p, c in zip(species_split, positions_split, cells_split):
        systems.append(System(types=s, positions=p, cell=c))

    return systems
