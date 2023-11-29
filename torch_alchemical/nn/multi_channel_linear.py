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
        self.weight = torch.nn.Parameter(
            torch.randn(
                (num_channels, in_features, out_features),
                device=device,
                dtype=dtype,
            )
        )
        self.weight.data.normal_(mean=0.0, std=in_features ** (-0.5))
        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros((num_channels, 1, out_features), device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, tensormap: TensorMap) -> TensorMap:
        output_blocks: list[TensorBlock] = []
        for i, block in enumerate(tensormap.blocks()):
            assert (
                len(block.components) == 1
            )  # only a single components group is supported
            assert len(block.components[0].values) == self.num_channels
            values = block.values.transpose(0, 1)
            out_values = torch.bmm(values, self.weight)
            if self.bias is not None:
                out_values += self.bias
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
                values=out_values.contiguous(),
                samples=block.samples,
                components=block.components,
                properties=labels,
            )
            output_blocks.append(new_block)
        return TensorMap(keys=tensormap.keys, blocks=output_blocks)
