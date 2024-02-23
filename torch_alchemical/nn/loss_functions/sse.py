from typing import Optional, Union

import torch


class WeightedSSELoss(torch.nn.Module):
    def __init__(self, energies_weight: float, forces_weight: Optional[float] = None):
        super().__init__()
        self.energies_weight = energies_weight
        self.forces_weight = forces_weight

    def forward(
        self,
        predicted_energies: Optional[torch.Tensor] = None,
        target_energies: Optional[torch.Tensor] = None,
        predicted_forces: Optional[torch.Tensor] = None,
        target_forces: Optional[torch.Tensor] = None,
    ):
        loss: Union[int, torch.Tensor] = 0
        weights = []
        predicted_tensors = []
        target_tensors = []
        if self.energies_weight is not None:
            weights.append(self.energies_weight)
        if self.forces_weight is not None:
            weights.append(self.forces_weight)
        if predicted_energies is not None and target_energies is not None:
            predicted_tensors.append(predicted_energies)
            target_tensors.append(target_energies)
        if predicted_forces is not None and target_forces is not None:
            predicted_tensors.append(predicted_forces)
            target_tensors.append(target_forces)
        assert len(weights) == len(predicted_tensors) == len(target_tensors)
        for weight, predicted_tensor, target_tensor in zip(
            weights, predicted_tensors, target_tensors
        ):
            assert predicted_tensor.shape == target_tensor.shape
            loss += torch.sum((predicted_tensor - target_tensor) ** 2) * weight
        return loss


class SSELoss(torch.nn.Module):
    def forward(
        self,
        predicted_energies: Optional[torch.Tensor] = None,
        target_energies: Optional[torch.Tensor] = None,
        predicted_forces: Optional[torch.Tensor] = None,
        target_forces: Optional[torch.Tensor] = None,
    ):
        loss: Union[int, torch.Tensor] = 0
        predicted_tensors = []
        target_tensors = []
        if predicted_energies is not None and target_energies is not None:
            predicted_tensors.append(predicted_energies)
            target_tensors.append(target_energies)
        if predicted_forces is not None and target_forces is not None:
            predicted_tensors.append(predicted_forces)
            target_tensors.append(target_forces)
        assert len(predicted_tensors) == len(target_tensors)
        for predicted_tensor, target_tensor in zip(predicted_tensors, target_tensors):
            assert predicted_tensor.shape == target_tensor.shape
            loss += torch.sum((predicted_tensor - target_tensor) ** 2)
        return loss
