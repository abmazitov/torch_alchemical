import torch_geometric as pyg
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from typing import Optional, List


class GCNConv(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = pyg.nn.GCNConv(*args, **kwargs).jittable()

    def forward(
        self,
        tensormap: TensorMap,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> TensorMap:
        output_blocks: List[TensorBlock] = []
        for block in tensormap.blocks():
            labels = Labels(
                names=["out_features_idx"],
                values=torch.arange(
                    self.conv.out_channels,
                    dtype=torch.int64,
                    device=block.values.device,
                ).reshape(-1, 1),
            )
            new_block = TensorBlock(
                values=self.conv.forward(block.values, edge_index, edge_weight),
                samples=block.samples,
                components=block.components,
                properties=labels,
            )
            output_blocks.append(new_block)
        return TensorMap(keys=tensormap.keys, blocks=output_blocks)
