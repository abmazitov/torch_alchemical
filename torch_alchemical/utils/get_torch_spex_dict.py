import torch
from torch_geometric.data import Batch


def get_torch_spex_dict(
    positions: list[torch.Tensor],
    cells: list[torch.Tensor],
    numbers: list[torch.Tensor],
    edge_index: list[torch.Tensor],
    edge_shift: list[torch.Tensor],
) -> dict:
    batch_dict = {}
    structure_centers = torch.cat(
        [
            torch.tensor([i] * len(pos), device=pos.device)
            for i, pos in enumerate(positions)
        ]
    )
    species = torch.cat(numbers)
    cell_shifts = torch.cat(edge_shift)
    structure_pairs = torch.cat(
        [
            torch.tensor([i] * edge.shape[1], device=edge.device)
            for i, edge in enumerate(edge_index)
        ]
    )
    centers = torch.cat([torch.arange(len(pos)) for pos in positions])
    pairs = torch.cat(edge_index, dim=1).T
    direction_vectors = torch.cat(
        [
            positions[i][edge_index[i][1]]
            - positions[i][edge_index[i][0]]
            + edge_shift[i] @ cells[i]
            for i in range(len(positions))
        ]
    )

    batch_dict["structure_centers"] = structure_centers
    batch_dict["structure_pairs"] = structure_pairs
    batch_dict["species"] = species
    batch_dict["centers"] = centers
    batch_dict["pairs"] = pairs
    batch_dict["cell_shifts"] = cell_shifts
    batch_dict["direction_vectors"] = direction_vectors
    return batch_dict


def get_torch_spex_dict_from_batch(batch: Batch) -> dict:
    batch_dict = {}
    batch_dict["structure_centers"] = batch.batch
    slice_dict = batch._slice_dict
    num_neighbors = [
        slice_dict["edge_index"][i + 1] - slice_dict["edge_index"][i]
        for i in range(len(slice_dict["edge_index"]) - 1)
    ]
    batch_dict["structure_pairs"] = torch.cat(
        [
            torch.tensor([i] * (num_neighbors[i]))
            for i in range(len(slice_dict["edge_index"]) - 1)
        ]
    )
    batch_dict["species"] = batch.numbers
    ptr = batch.ptr
    batch_dict["centers"] = torch.cat(
        [torch.arange(ptr[i + 1] - ptr[i]) for i in range(len(ptr) - 1)]
    )
    batch_dict["pairs"] = torch.cat(
        [batch[i].edge_index for i in range(len(batch))], dim=1
    ).T
    batch_dict["cell_shifts"] = batch.edge_shift
    edge_src, edge_dst = batch.edge_index[0], batch.edge_index[1]
    edge_batch = batch.batch[edge_src]
    edge_shift = batch.edge_shift
    cell = batch.cell.reshape(-1, 3, 3)[edge_batch]
    pos = batch.pos
    batch_dict["direction_vectors"] = (
        pos[edge_dst] - pos[edge_src] + torch.einsum("ni,nij->nj", edge_shift, cell)
    )
    return batch_dict
