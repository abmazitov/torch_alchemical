import torch
from ase.neighborlist import primitive_neighbor_list
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


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
        edge_src, edge_dst, edge_offsets = primitive_neighbor_list(
            quantities="ijS",
            pbc=data.pbc,
            cell=data.cell.detach().cpu().numpy(),
            positions=data.pos.detach().cpu().numpy(),
            cutoff=self.cutoff_radius,
            self_interaction=False,
            use_scaled_positions=False,
        )
        edge_index = torch.stack(
            [
                torch.tensor(edge_src, dtype=torch.int32),
                torch.tensor(edge_dst, dtype=torch.int32),
            ],
            dim=0,
        )
        edge_offsets = torch.tensor(edge_offsets, dtype=torch.int32)
        data.edge_index = edge_index
        data.edge_offsets = edge_offsets
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cutoff_radius={self.cutoff_radius})"
