import torch
from typing import Optional
from torch_alchemical.nn import TensorConv, TensorEmbedding, RBFEmbedding
from torch_geometric.nn import global_add_pool


# TODO move this to utils
def decomposition_transform(tensor: torch.Tensor):
    I = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[
        ..., None, None
    ] * torch.eye(3, 3, device=tensor.device, dtype=tensor.dtype)
    A = 0.5 * (tensor - tensor.transpose(-2, -1))
    S = 0.5 * (tensor + tensor.transpose(-2, -1)) - I
    X = torch.stack((I, A, S), dim=-1)
    return X


class TensorNet(torch.nn.Module):
    def __init__(
        self,
        unique_numbers: torch.Tensor,
        hidden_size: int,
        radial_embedding_size: int,
        cutoff: float,
        num_layers: int = 2,
        node_dim: int = 0,
        flow: str = "target_to_source",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.unique_numbers = unique_numbers
        self.hidden_size = hidden_size
        self.radial_embedding_size = radial_embedding_size
        self.cutoff = cutoff
        self.radial_embedding = RBFEmbedding(radial_embedding_size, cutoff)
        self.tensor_embedding = TensorEmbedding(
            unique_numbers,
            hidden_size,
            radial_embedding_size,
            cutoff,
            node_dim,
            flow,
            device,
            dtype,
        ).jittable()
        self.tensor_conv = torch.nn.ModuleList(
            [
                TensorConv(
                    hidden_size,
                    radial_embedding_size,
                    cutoff,
                    node_dim,
                    flow,
                    device,
                    dtype,
                ).jittable()
                for _ in range(num_layers)
            ]
        )
        self.linear = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    3 * hidden_size, hidden_size, bias=True, device=device, dtype=dtype
                ),
                torch.nn.Linear(
                    hidden_size, hidden_size, bias=True, device=device, dtype=dtype
                ),
                torch.nn.Linear(hidden_size, 1, bias=True, device=device, dtype=dtype),
            ]
        )
        self.layer_norm = torch.nn.LayerNorm(3 * hidden_size, dtype=dtype)
        self.act = torch.nn.SiLU()

    def forward(
        self,
        positions: torch.Tensor,
        cells: torch.Tensor,
        numbers: torch.Tensor,
        edge_index: torch.Tensor,
        edge_offsets: torch.Tensor,
        batch: torch.Tensor,
    ):
        cells = cells.view(-1, 3, 3)[batch][edge_index[0]]
        cartesian_vectors = (
            positions[edge_index[1]]
            - positions[edge_index[0]]
            + torch.einsum("ij,ijk->ik", edge_offsets, cells)
        )
        edge_weights = torch.norm(cartesian_vectors, dim=-1)
        edge_attrs = self.radial_embedding(edge_weights)
        X = self.tensor_embedding(
            numbers=numbers,
            edge_index=edge_index,
            edge_weights=edge_weights,
            edge_attrs=edge_attrs,
            cartesian_vectors=cartesian_vectors,
        )
        for layer in self.tensor_conv:
            X = layer(
                X=X,
                edge_index=edge_index,
                edge_weights=edge_weights,
                edge_attrs=edge_attrs,
            )
        X = decomposition_transform(X)
        X = torch.linalg.norm(X, dim=(-1, -2), ord="fro") ** 2
        X = self.layer_norm(torch.cat(X.unbind(dim=-1), dim=-1))
        for layer in self.linear[:-1]:
            X = self.act(layer(X))
        X = self.linear[-1](X)
        energy = global_add_pool(X, batch)
        return energy
