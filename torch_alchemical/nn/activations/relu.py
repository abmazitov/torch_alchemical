import torch
from equistore import TensorBlock, TensorMap


class ReLU(torch.nn.ReLU):
    def __init__(self):
        super().__init__()

    def forward(self, tensormap: TensorMap) -> TensorMap:
        output_blocks = []
        for block in tensormap:
            new_block = TensorBlock(
                values=super().forward(block.values),
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
            output_blocks.append(new_block)
        return TensorMap(keys=tensormap.keys, blocks=output_blocks)
