from typing import List

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


class LinearMap(torch.nn.Module):
    def __init__(self, keys, *args, **kwargs):
        super().__init__()
        self.keys = keys
        self.linear = torch.nn.ModuleDict()
        for i in range(len(keys)):
            layer = torch.nn.Linear(*args, **kwargs)
            layer.weight.data.normal_(mean=0.0, std=layer.in_features ** (-0.5))
            if layer.bias is not None:
                layer.bias.data.zero_()
            self.linear[str(i)] = layer

    def forward(self, tensormap: TensorMap) -> TensorMap:
        output_blocks: List[TensorBlock] = []
        for key, linear in self.linear.items():
            block = tensormap.block(int(key))
            labels = Labels(
                names=["property"],
                values=torch.arange(
                    linear.out_features, dtype=torch.int64, device=block.values.device
                ).reshape(-1, 1),
            )
            new_block = TensorBlock(
                values=linear(block.values),
                samples=block.samples,
                components=block.components,
                properties=labels,
            )
            output_blocks.append(new_block)
        return TensorMap(keys=tensormap.keys, blocks=output_blocks)
