import torch


class CosineCutoff(torch.nn.Module):
    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances):
        cutoff_distances = 0.5 * (torch.cos(distances * torch.pi / self.cutoff) + 1)
        cutoff_distances[distances > self.cutoff] = 0
        return cutoff_distances
