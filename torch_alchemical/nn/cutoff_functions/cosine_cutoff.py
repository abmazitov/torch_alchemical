import torch


class CosineCutoff(torch.nn.Module):
    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances):
        return 0.5 * (torch.cos(distances * torch.pi / self.cutoff) + 1)
