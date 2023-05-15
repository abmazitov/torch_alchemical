import numpy as np
import torch
from equistore import Labels, TensorBlock


class LinearBlock(torch.nn.Linear):
    def forward(self, block: TensorBlock) -> TensorBlock:
        labels = Labels(
            names=["out_features_idx"],
            values=np.arange(self.out_features, dtype=np.int32).reshape(-1, 1),
        )
        return TensorBlock(
            values=super().forward(block.values),
            samples=block.samples,
            components=block.components,
            properties=labels,
        )
