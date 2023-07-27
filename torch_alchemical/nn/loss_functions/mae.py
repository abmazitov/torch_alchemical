import torch


class WeightedMAELoss(torch.nn.Module):
    def __init__(self, weights: float | list[float]):
        super().__init__()
        if isinstance(weights, float):
            weights = [weights]
        self.weights = weights

    def forward(
        self,
        predicted: torch.Tensor | list[torch.Tensor],
        target: torch.Tensor | list[torch.Tensor],
    ):
        if isinstance(predicted, torch.Tensor):
            predicted = [predicted]
        if isinstance(target, torch.Tensor):
            target = [target]
        assert len(predicted) == len(target)
        loss = 0
        for weight, predicted_tensor, target_tensor in zip(
            self.weights, predicted, target
        ):
            assert predicted_tensor.shape == target_tensor.shape
            loss += torch.mean(torch.abs(predicted_tensor - target_tensor)) * weight
        return loss


class MAELoss(torch.nn.Module):
    def forward(
        self,
        predicted: torch.Tensor | list[torch.Tensor],
        target: torch.Tensor | list[torch.Tensor],
    ):
        if isinstance(predicted, torch.Tensor):
            predicted = [predicted]
        if isinstance(target, torch.Tensor):
            target = [target]
        assert len(predicted) == len(target)
        loss = 0
        for predicted_tensor, target_tensor in zip(predicted, target):
            assert predicted_tensor.shape == target_tensor.shape
            loss += torch.mean(torch.abs(predicted_tensor - target_tensor))
        return loss
