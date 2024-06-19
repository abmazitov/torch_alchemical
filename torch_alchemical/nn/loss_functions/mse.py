from typing import Optional
from torch.nn import MSELoss

import torch


class WeightedMSELoss(torch.nn.Module):
    def __init__(self, energies_weight: float, forces_weight: Optional[float] = None):
        super().__init__()
        self.energies_weight = energies_weight
        self.forces_weight = forces_weight

    def forward(
        self,
        predicted_properties: Optional[torch.Tensor] = None,
        target_properties: Optional[torch.Tensor] = None,
    ):

        loss = MSELoss()
        
        return loss_value