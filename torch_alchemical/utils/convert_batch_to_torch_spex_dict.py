import torch
from torch_geometric.data import Batch


def convert_batch_to_torch_spex_dict(batch: Batch) -> dict:
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
