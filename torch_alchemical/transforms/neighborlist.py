import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from ase.neighborlist import primitive_neighbor_list


class NeighborList(BaseTransform):
    def __init__(
        self,
        cutoff_radius: float,
    ):
        self.cutoff_radius = cutoff_radius

    def __call__(
        self,
        data: Data,
    ) -> Data:
        assert hasattr(data, "numbers")
        assert hasattr(data, "pos")
        assert hasattr(data, "cell")
        assert hasattr(data, "pbc")
        edge_src, edge_dst, edge_shift = primitive_neighbor_list(
            quantities="ijS",
            pbc=data.pbc,
            cell=data.cell.detach().cpu().numpy(),
            positions=data.pos.detach().cpu().numpy(),
            cutoff=self.cutoff_radius,
            self_interaction=True,
            use_scaled_positions=False,
        )
        pairs_to_throw = np.logical_and(
            edge_src == edge_dst, np.all(edge_shift == 0, axis=1)
        )
        pairs_to_keep = np.logical_not(pairs_to_throw)
        edge_src = edge_src[pairs_to_keep]
        edge_dst = edge_dst[pairs_to_keep]
        edge_shift = edge_shift[pairs_to_keep]
        edge_index = torch.stack(
            [torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0
        )
        edge_shift = torch.tensor(edge_shift, dtype=torch.get_default_dtype())
        data.edge_index = edge_index
        data.edge_shift = edge_shift
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cutoff_radius={self.cutoff_radius})"
