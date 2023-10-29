import torch
from metatensor.torch import TensorBlock, TensorMap


class SiLU(torch.nn.SiLU):
    def forward(self, tensormap: TensorMap) -> TensorMap:
        output_blocks = []
        for block in tensormap.blocks():
            new_block = TensorBlock(
                values=super().forward(block.values),
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
            output_blocks.append(new_block)
        return TensorMap(keys=tensormap.keys, blocks=output_blocks)
