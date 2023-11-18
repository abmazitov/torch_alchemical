import torch
from torch_geometric.nn import MessagePassing
from typing import Optional
from torch_alchemical.nn.cutoff_functions import CosineCutoff


# TODO move this to utils
def decomposition_transform(tensor: torch.Tensor):
    I = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[
        ..., None, None
    ] * torch.eye(3, 3, device=tensor.device, dtype=tensor.dtype)
    A = 0.5 * (tensor - tensor.transpose(-2, -1))
    S = 0.5 * (tensor + tensor.transpose(-2, -1)) - I
    X = torch.stack((I, A, S), dim=-1)
    return X


class TensorConv(MessagePassing):
    propagate_type = {
        "X": torch.Tensor,
        "X_old": torch.Tensor,
        "Y": torch.Tensor,
        "edge_attrs": torch.Tensor,
    }

    def __init__(
        self,
        hidden_size: int,
        radial_embedding_size: int,
        cutoff: float,
        node_dim: int = 0,
        flow: str = "target_to_source",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        *args,
        **kwargs,
    ):
        super().__init__(node_dim=node_dim, flow=flow, *args, **kwargs)
        self.hidden_size = hidden_size
        self.radial_embedding_size = radial_embedding_size
        self.cutoff_function = CosineCutoff(cutoff)
        self.radial_layer = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    radial_embedding_size,
                    hidden_size,
                    bias=True,
                    dtype=dtype,
                    device=device,
                ),
                torch.nn.Linear(
                    hidden_size,
                    2 * hidden_size,
                    bias=True,
                    dtype=dtype,
                    device=device,
                ),
                torch.nn.Linear(
                    2 * hidden_size,
                    3 * hidden_size,
                    bias=True,
                    dtype=dtype,
                    device=device,
                ),
            ]
        )
        self.act = torch.nn.SiLU()
        self.linear = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    hidden_size, hidden_size, bias=False, dtype=dtype, device=device
                )
                for _ in range(6)
            ]
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.radial_layer:
            layer.reset_parameters()
        for layer in self.linear:
            layer.reset_parameters()

    def message(
        self, X: torch.Tensor, edge_index: torch.Tensor, edge_attrs: torch.Tensor
    ):
        X_j = X.index_select(0, edge_index[1]) * edge_attrs[..., None, None, :]
        return X_j

    def aggregate(self, X_j: torch.Tensor, index: torch.Tensor, size_j: int):
        X_agg = torch.zeros(size_j, *X_j.shape[1:], dtype=X_j.dtype, device=X_j.device)
        X_agg = X_agg.index_add(0, index, X_j)
        return X_agg

    def update(self, X_aggr: torch.Tensor, X_old: torch.Tensor, Y: torch.Tensor):
        M = X_aggr.sum(dim=-1)
        X = Y @ M + M @ Y
        X = decomposition_transform(X)
        X = (
            X
            / (torch.linalg.norm(X, dim=(-1, -2), ord="fro") ** 2 + 1.0)[
                ..., None, None
            ]
        )
        X = X.transpose(1, -2)
        for i, linear in enumerate(self.linear[3:]):
            X[..., i] = linear(X[..., i])
        X = X.transpose(1, -2)
        X = X.sum(dim=-1)
        return X_old + X + torch.matrix_power(X, 2)

    def forward(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor,
        edge_attrs: torch.Tensor,
    ):
        for layer in self.radial_layer:
            edge_attrs = self.act(layer(edge_attrs))
        edge_attrs = (
            self.cutoff_function(edge_weights).view(-1, 1) * edge_attrs
        ).reshape(-1, self.hidden_size, 3)
        X = (
            X
            / (torch.linalg.norm(X, dim=(-1, -2), ord="fro") ** 2 + 1.0)[
                ..., None, None
            ]
        )
        X_old = X.clone()
        X = decomposition_transform(X).transpose(1, -2)
        for i, linear in enumerate(self.linear[:3]):
            X[..., i] = linear(X[..., i])
        X = X.transpose(1, -2)
        Y = X.sum(dim=-1)
        X = self.propagate(
            edge_index,
            X=X,
            X_old=X_old,
            Y=Y,
            edge_attrs=edge_attrs,
            size=(X.size(0), X.size(1)),
        )
        return X

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_size={self.hidden_size}, "
            f"radial_embedding_size={self.radial_embedding_size}, "
            f"cutoff={self.cutoff})"
        )
