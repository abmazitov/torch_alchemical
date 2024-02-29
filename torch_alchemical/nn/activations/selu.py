from typing import List

import torch
from metatensor.torch import TensorBlock, TensorMap


class SELU(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.selu = torch.nn.SELU(*args, **kwargs)
        self.scaling = 1.0

    def forward(self, tensormap: TensorMap) -> TensorMap:
        output_blocks: List[TensorBlock] = []
        for block in tensormap.blocks():
            new_block = TensorBlock(
                values=self.selu(block.values) * self.scaling,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
            output_blocks.append(new_block)
        return TensorMap(keys=tensormap.keys, blocks=output_blocks)
