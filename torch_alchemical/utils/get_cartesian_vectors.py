import torch
import numpy as np
from torch_geometric.data import Batch
from equistore import Labels, TensorBlock


def get_cartesian_vectors(batch: Batch) -> TensorBlock:
    assert all(
        hasattr(batch, attr)
        for attr in ["edge_index", "edge_shift", "pos", "batch", "cell"]
    )
    edge_src, edge_dst = batch.edge_index[0], batch.edge_index[1]
    edge_batch = batch.batch[edge_src]
    edge_shift = batch.edge_shift
    cell = batch.cell.reshape(-1, 3, 3)[edge_batch]
    pos = batch.pos
    vectors = (
        pos[edge_dst] - pos[edge_src] + torch.einsum("ni,nij->nj", edge_shift, cell)
    )
    labels = torch.stack(
        (
            edge_batch,
            edge_src,
            edge_dst,
            batch.numbers[edge_src],
            batch.numbers[edge_dst],
            edge_shift[:, 0],
            edge_shift[:, 1],
            edge_shift[:, 2],
        ),
        dim=-1,
    )

    block = TensorBlock(
        values=vectors.unsqueeze(dim=-1),
        samples=Labels(
            names=[
                "structure",
                "center",
                "neighbor",
                "species_center",
                "species_neighbor",
                "cell_x",
                "cell_y",
                "cell_z",
            ],
            values=np.array(labels, dtype=np.int32),
        ),
        components=[
            Labels(
                names=["cartesian_dimension"],
                values=np.array([-1, 0, 1], dtype=np.int32).reshape((-1, 1)),
            )
        ],
        properties=Labels.single(),
    )
    return block
