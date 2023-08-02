import torch
from torch_geometric.data import Data
from .extract_batch_data import extract_batch_data


def get_torch_spex_dict(
    positions: list[torch.Tensor],
    cells: list[torch.Tensor],
    numbers: list[torch.Tensor],
    edge_indices: list[torch.Tensor],
    edge_shifts: list[torch.Tensor],
) -> dict:
    species = torch.cat(numbers)
    cell_shifts = torch.cat(edge_shifts)
    centers = torch.cat(
        [torch.arange(len(pos), device=pos.device) for pos in positions]
    )
    pairs = torch.cat(edge_indices, dim=1).T

    structure_centers = torch.cat(
        [
            torch.tensor([i] * len(pos), device=pos.device)
            for i, pos in enumerate(positions)
        ]
    )

    structure_pairs = torch.cat(
        [
            torch.tensor([i] * edge.shape[1], device=edge.device)
            for i, edge in enumerate(edge_indices)
        ]
    )

    structure_offsets = torch.cumsum(
        torch.tensor([0] + [len(pos) for pos in positions[:-1]]), dim=0
    )

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


def get_torch_spex_dict_from_batch(batch: list[Data]) -> dict:
    positions, cells, numbers, edge_indices, edge_shifts = extract_batch_data(batch)
    batch_dict = get_torch_spex_dict(
        positions=positions,
        cells=cells,
        numbers=numbers,
        edge_indices=edge_indices,
        edge_shifts=edge_shifts,
    )
    return batch_dict
