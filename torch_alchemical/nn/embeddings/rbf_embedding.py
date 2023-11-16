import torch
from typing import Optional


class RBFEmbedding(torch.nn.Module):
    def __init__(
        self,
        n_max: int,
        cutoff: float,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.n_max = n_max
        self.cutoff = torch.tensor(cutoff, device=device, dtype=dtype)
        mu = torch.linspace(torch.exp(-self.cutoff).item(), 1.0, n_max)
        beta = torch.ones(n_max, device=device, dtype=dtype) * (
            2 * n_max ** (-1) * (1 - torch.exp(-self.cutoff)) ** (-2)
        )
        self.register_buffer("mu", mu)
        self.register_buffer("beta", beta)

    def forward(self, distances: torch.Tensor):
        distances = distances.unsqueeze(-1)
        return torch.exp(-self.beta * (torch.exp(-distances) - self.mu) ** 2)
