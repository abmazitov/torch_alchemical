import torch
from equistore import TensorMap

from .block import LinearBlock


class LinearMap(torch.nn.Linear):
    def __init__(self, keys, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys = keys
        self.linear_blocks = torch.nn.ModuleDict(
            {str(key): LinearBlock(*args, **kwargs) for key in keys}
        )

    def forward(self, tensormap: TensorMap) -> TensorMap:
        output_blocks = []
        for key_value in self.keys:
            key_name = tensormap.keys.dtype.names[0]
            block = tensormap.block(**{key_name: key_value})
            new_block = self.linear_blocks[str(key_value)](block)
            output_blocks.append(new_block)
        return TensorMap(keys=tensormap.keys, blocks=output_blocks)
