from typing import List

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


class Linear(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(*args, **kwargs)
        self.linear.weight.data.normal_(mean=0.0, std=self.linear.in_features ** (-0.5))
        if self.linear.bias is not None:
            self.linear.bias.data.zero_()

    def forward(self, tensormap: TensorMap) -> TensorMap:
        output_blocks: List[TensorBlock] = []
        for block in tensormap.blocks():
            labels = Labels(
                names=["property"],
                values=torch.arange(
                    self.linear.out_features,
                    dtype=torch.int64,
                    device=block.values.device,
                ).reshape(-1, 1),
            )
            new_block = TensorBlock(
                values=self.linear(block.values),
                samples=block.samples,
                components=block.components,
                properties=labels,
            )
            output_blocks.append(new_block)
        return TensorMap(keys=tensormap.keys, blocks=output_blocks)
