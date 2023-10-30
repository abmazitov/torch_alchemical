import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


class Linear(torch.nn.Linear):
    def forward(self, tensormap: TensorMap) -> TensorMap:
        output_blocks = []
        for block in tensormap.blocks():
            labels = Labels(
                names=["out_features_idx"],
                values=torch.arange(
                    self.out_features, dtype=torch.int64, device=block.values.device
                ).reshape(-1, 1),
            )
            new_block = TensorBlock(
                values=super().forward(block.values),
                samples=block.samples,
                components=block.components,
                properties=labels,
            )
            output_blocks.append(new_block)
        return TensorMap(keys=tensormap.keys, blocks=output_blocks)
