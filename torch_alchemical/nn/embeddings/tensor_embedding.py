import torch
from torch_geometric.nn import MessagePassing
from typing import Optional
from torch_alchemical.nn.cutoff_functions import CosineCutoff


# TODO move this to utils
def vector_to_skewtensor(vector: torch.Tensor) -> torch.Tensor:
    batch_size = vector.size(0)
    zero = torch.zeros(batch_size, device=vector.device, dtype=vector.dtype)
    tensor = torch.stack(
        (
            zero,
            -vector[:, 2],
            vector[:, 1],
            vector[:, 2],
            zero,
            -vector[:, 0],
            -vector[:, 1],
            vector[:, 0],
            zero,
        ),
        dim=1,
    )
    tensor = tensor.view(-1, 3, 3)
    return tensor.squeeze(0)


def vector_to_symtensor(vector: torch.Tensor) -> torch.Tensor:
    tensor = torch.matmul(vector.unsqueeze(-1), vector.unsqueeze(-2))
    I = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[
        ..., None, None
    ] * torch.eye(3, 3, device=tensor.device, dtype=tensor.dtype)
    S = 0.5 * (tensor + tensor.transpose(-2, -1)) - I
    return S


class TensorEmbedding(MessagePassing):
    propagate_type = {
        "X": torch.Tensor,
        "edge_weights": torch.Tensor,
        "edge_attrs": torch.Tensor,
        "cartesian_vectors": torch.Tensor,
    }

    def __init__(
        self,
        unique_numbers: torch.Tensor,
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
        self.unique_numbers = unique_numbers
        self.hidden_size = hidden_size
        self.radial_embedding_size = radial_embedding_size
        self.cutoff = cutoff
        self.cutoff_function = CosineCutoff(cutoff)
        self.radial_linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    radial_embedding_size, hidden_size, dtype=dtype, device=device
                )
                for _ in range(3)
            ]
        )
        self.chemical_embedding = torch.nn.Embedding(
            len(unique_numbers), hidden_size, dtype=dtype, device=device
        )
        self.chemical_mapping = {
            number.item(): i for i, number in enumerate(unique_numbers)
        }
        self.chemical_linear = torch.nn.Linear(
            2 * hidden_size, hidden_size, bias=True, dtype=dtype, device=device
        )
        self.normalization_layer = torch.nn.ModuleList(
            [
                torch.nn.LayerNorm(hidden_size),
                torch.nn.Linear(
                    hidden_size, 2 * hidden_size, bias=True, dtype=dtype, device=device
                ),
                torch.nn.Linear(
                    2 * hidden_size,
                    3 * hidden_size,
                    bias=True,
                    dtype=dtype,
                    device=device,
                ),
                torch.nn.SiLU(),
            ]
        )
        self.linear = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    hidden_size, hidden_size, bias=False, dtype=dtype, device=device
                )
                for _ in range(3)
            ]
        )

    def get_chemical_messages(self, numbers: torch.Tensor, edge_index: torch.Tensor):
        mapped_numbers = torch.tensor(
            [self.chemical_mapping[number.item()] for number in numbers],
            dtype=numbers.dtype,
            device=numbers.device,
        )
        X = self.chemical_embedding(mapped_numbers)
        X_i = X.index_select(0, edge_index[0])
        X_j = X.index_select(0, edge_index[1])
        X = self.chemical_linear(torch.cat((X_i, X_j), dim=-1))
        return X

    def message(
        self,
        X: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_weights: torch.Tensor,
        cartesian_vectors: torch.Tensor,
    ):
        cartesian_vectors = cartesian_vectors / edge_weights[..., None]
        X = (self.cutoff_function(edge_weights).view(-1, 1) * X)[..., None, None]
        I_j = (
            self.radial_linears[0](edge_attrs)[..., None, None]
            * X
            * torch.eye(3, 3)[None, None, ...]
        )
        A_j = (
            self.radial_linears[1](edge_attrs)[..., None, None]
            * X
            * vector_to_skewtensor(cartesian_vectors)[:, None, ...]
        )
        S_j = (
            self.radial_linears[2](edge_attrs)[..., None, None]
            * X
            * vector_to_symtensor(cartesian_vectors)[:, None, ...]
        )
        X_j = torch.stack((I_j, A_j, S_j), dim=-1)
        return X_j

    def forward(
        self,
        numbers: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor,
        edge_attrs: torch.Tensor,
        cartesian_vectors: torch.Tensor,
    ):
        X = self.get_chemical_messages(numbers, edge_index)
        X = self.propagate(
            edge_index,
            X=X,
            edge_weights=edge_weights,
            edge_attrs=edge_attrs,
            cartesian_vectors=cartesian_vectors,
        )
        norm = torch.linalg.norm(X.sum(dim=-1), dim=(-2, -1), ord="fro") ** 2
        for layer in self.normalization_layer:
            norm = layer(norm)
        norm = norm.reshape(-1, self.hidden_size, 3)
        X = (X * norm[..., None, None, :]).transpose(1, -2)
        for i, layer in enumerate(self.linear):
            X[..., i] = layer(X[..., i])
        X = X.transpose(1, -2).sum(dim=-1)
        return X

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"unique_numbers={self.unique_numbers}, "
            f"hidden_size={self.hidden_size}, "
            f"radial_embedding_size={self.radial_embedding_size}, "
            f"cutoff={self.cutoff})"
        )
