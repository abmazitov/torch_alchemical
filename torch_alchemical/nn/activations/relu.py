import torch
from metatensor.torch import TensorBlock, TensorMap


class ReLU(torch.nn.ReLU):
    def __init__(self, normalize: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if normalize:
            with torch.no_grad():
                z = torch.randn(1000000, dtype=torch.get_default_dtype())
                self.normalization_factor = (
                    torch.nn.ReLU()(z).pow(2).mean().pow(-0.5).item()
                )
        else:
            self.normalization_factor = 1.0

    def forward(self, tensormap: TensorMap) -> TensorMap:
        output_blocks = []
        for block in tensormap:
            new_block = TensorBlock(
                values=self.normalization_factor * super().forward(block.values),
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
            output_blocks.append(new_block)
        return TensorMap(keys=tensormap.keys, blocks=output_blocks)
