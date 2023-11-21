import torch
from typing import Dict


def get_torch_spex_dict_from_data_lists(
    positions: list[torch.Tensor],
    cells: list[torch.Tensor],
    numbers: list[torch.Tensor],
    edge_indices: list[torch.Tensor],
    edge_shifts: list[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    device = positions[0].device
    for tensor in positions + cells + numbers + edge_indices + edge_shifts:
        assert tensor.device == device
    species = torch.cat(numbers)
    cell_shifts = torch.cat(edge_shifts).to(torch.int64)
    centers = torch.cat([torch.arange(len(pos), device=device) for pos in positions])
    pairs = torch.cat(edge_indices, dim=1).T

    structure_centers = torch.cat(
        [torch.tensor([i] * len(pos), device=device) for i, pos in enumerate(positions)]
    )

    structure_pairs = torch.cat(
        [
            torch.tensor([i] * edge.shape[1], device=device)
            for i, edge in enumerate(edge_indices)
        ]
    )

    structure_offsets = torch.cumsum(
        torch.tensor([0] + [len(pos) for pos in positions[:-1]], device=device), dim=0
    )

    positions = torch.cat(positions, dim=0)
    cells = torch.stack(cells)

    batch_dict = dict(
        positions=positions,
        cells=cells,
        species=species,
        cell_shifts=cell_shifts,
        centers=centers,
        pairs=pairs,
        structure_centers=structure_centers,
        structure_pairs=structure_pairs,
        structure_offsets=structure_offsets,
    )
    return batch_dict


def get_torch_spex_dict(
    positions: torch.Tensor,
    cells: torch.Tensor,
    numbers: torch.Tensor,
    edge_indices: torch.Tensor,
    edge_shifts: torch.Tensor,
    batch: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    device = positions.device
    batches, counts = torch.unique(batch, return_counts=True)
    num_batches = len(batches)
    ptr = torch.cat([torch.tensor([0], device=device), torch.cumsum(counts, dim=0)])
    if cells.ndim == 2:
        cells = cells.reshape(-1, 3, 3)
    centers = torch.cat([torch.arange(length, device=device) for length in counts])
    pairs = edge_indices.T.clone().to(device)
    structure_pairs = torch.zeros(len(pairs), device=device, dtype=torch.int64)
    for i in range(num_batches):
        mask = torch.bitwise_and(pairs < ptr[i + 1], pairs >= ptr[i]).all(dim=1)
        structure_pairs[mask] = i
        pairs[mask] -= ptr[i]
    structure_centers = torch.repeat_interleave(
        torch.arange(num_batches, device=device), ptr[1:] - ptr[:-1]
    )
    structure_offsets = ptr[:-1]

    batch_dict = dict(
        positions=positions,
        cells=cells,
        species=numbers,
        cell_shifts=edge_shifts.to(torch.int64),
        centers=centers,
        pairs=pairs,
        structure_centers=structure_centers,
        structure_pairs=structure_pairs,
        structure_offsets=structure_offsets,
    )
    return batch_dict
