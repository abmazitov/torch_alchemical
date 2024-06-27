from typing import List

import torch
from metatensor.torch import TensorBlock, TensorMap


class LayerNorm(torch.nn.Module):
    def __init__(self, keys, *args, **kwargs):
        super().__init__()
        self.keys = keys
        self.layernorm = torch.nn.ModuleDict()
        for i in range(len(keys)):
            layer = torch.nn.LayerNorm(*args, **kwargs)
            self.layernorm[str(i)] = layer

    def forward(self, tensormap: TensorMap) -> TensorMap:
        output_blocks: List[TensorBlock] = []
        for block, key in zip(tensormap.blocks(), self.layernorm):
            new_block = TensorBlock(
                values=self.layernorm[key](block.values),
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
            output_blocks.append(new_block)
        return TensorMap(keys=tensormap.keys, blocks=output_blocks)
