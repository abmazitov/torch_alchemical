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


class MultiChannelGCNConv(torch.nn.Module):
    def __init__(self, num_channels, *args, **kwargs):
        super().__init__()
        self.num_channels = num_channels
        self.conv = torch.nn.ModuleList(
            [pyg.nn.GCNConv(*args, **kwargs).jittable() for i in range(num_channels)]
        )

    def forward(
        self,
        tensormap: TensorMap,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> TensorMap:
        output_blocks: List[TensorBlock] = []
        channels_values: List[torch.Tensor] = []
        for block in tensormap.blocks():
            for i, conv in enumerate(self.conv):
                values = conv.forward(block.values[:, i, :], edge_index, edge_weight)
                channels_values.append(values)
            labels = Labels(
                names=["out_features_idx"],
                values=torch.arange(
                    values.shape[1],
                    dtype=torch.int64,
                    device=block.values.device,
                ).reshape(-1, 1),
            )
            new_block = TensorBlock(
                values=torch.stack(channels_values, dim=1),
                samples=block.samples,
                components=block.components,
                properties=labels,
            )
            output_blocks.append(new_block)
        return TensorMap(keys=tensormap.keys, blocks=output_blocks)
