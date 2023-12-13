import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


class MultiChannelLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_channels: int,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_channels = num_channels
        self.device = device
        self.dtype = dtype
        self.linear = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    in_features, out_features, bias=bias, device=device, dtype=dtype
                )
                for _ in range(num_channels)
            ]
        )
        for layer in self.linear:
            layer.weight.data.normal_(mean=0.0, std=in_features ** (-0.5))
            if bias:
                layer.bias.data.zero_()

    def forward(self, tensormap: TensorMap) -> TensorMap:
        output_blocks: list[TensorBlock] = []
        for i, block in enumerate(tensormap.blocks()):
            assert (
                len(block.components) == 1
            )  # only a single components group is supported
            assert len(block.components[0].values) == self.num_channels
            values = block.values.transpose(0, 1)
            out_values = []
            for i, layer in enumerate(self.linear):
                out_values.append(layer(values[i]))
            out_values = torch.stack(out_values, dim=0)
            out_values = out_values.transpose(0, 1)
            labels = Labels(
                names=["out_features_idx"],
                values=torch.arange(
                    self.out_features,
                    dtype=torch.int64,
                    device=block.values.device,
                ).reshape(-1, 1),
            )
            new_block = TensorBlock(
                values=out_values,
                samples=block.samples,
                components=block.components,
                properties=labels,
            )
            output_blocks.append(new_block)
        return TensorMap(keys=tensormap.keys, blocks=output_blocks)
