import numpy as np
import torch_geometric as pyg
from equistore import Labels, TensorBlock, TensorMap


class GCNConv(pyg.nn.GCNConv):
    def forward(
        self,
        tensormap: TensorMap,
        edge_index: pyg.typing.Adj,
        edge_weight: pyg.typing.OptTensor = None,
    ) -> TensorMap:
        output_blocks = []
        for _, block in tensormap:
            labels = Labels(
                names=["out_features_idx"],
                values=np.arange(self.out_channels, dtype=np.int32).reshape(-1, 1),
            )
            new_block = TensorBlock(
                values=super().forward(block.values, edge_index, edge_weight),
                samples=block.samples,
                components=block.components,
                properties=labels,
            )
            output_blocks.append(new_block)
        return TensorMap(keys=tensormap.keys, blocks=output_blocks)
