import torch


class WeightedMAELoss(torch.nn.Module):
    def __init__(self, energies_weight: float, forces_weight: float = None):
        super().__init__()
        self.energies_weight = energies_weight
        self.forces_weight = forces_weight

    def forward(
        self,
        predicted_energies: torch.Tensor = None,
        target_energies: torch.Tensor = None,
        predicted_forces: torch.Tensor = None,
        target_forces: torch.Tensor = None,
    ):
        loss = 0
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
            loss += torch.mean(torch.abs(predicted_tensor - target_tensor)) * weight
        return loss


class MAELoss(torch.nn.Module):
    def forward(
        self,
        predicted_energies: torch.Tensor = None,
        target_energies: torch.Tensor = None,
        predicted_forces: torch.Tensor = None,
        target_forces: torch.Tensor = None,
    ):
        loss = 0
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
            loss += torch.mean(torch.abs(predicted_tensor - target_tensor))
        return loss
