import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


class LinearMap(torch.nn.Module):
    def __init__(self, keys, *args, **kwargs):
        super().__init__()
        self.keys = keys
        self.linear = torch.nn.ModuleDict(
            {str(key): torch.nn.Linear(*args, **kwargs) for key in keys}
        )

    def forward(self, tensormap: TensorMap) -> TensorMap:
        output_blocks: list[TensorBlock] = []
        for key, linear in self.linear.items():
            block = tensormap.block({"a_i": int(key)})
            labels = Labels(
                names=["out_features_idx"],
                values=torch.arange(
                    linear.out_features, dtype=torch.int64, device=block.values.device
                ).reshape(-1, 1),
            )
            new_block = TensorBlock(
                values=linear.forward(block.values),
                samples=block.samples,
                components=block.components,
                properties=labels,
            )
            output_blocks.append(new_block)
        return TensorMap(keys=tensormap.keys, blocks=output_blocks)
