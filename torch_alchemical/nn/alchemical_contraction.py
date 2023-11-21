import torch
from metatensor.torch import TensorBlock, TensorMap, Labels


class AlchemicalContraction(torch.nn.Module):
    def __init__(
        self,
        unique_numbers: list[int],
        contraction_matrix: torch.Tensor,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.unique_numbers = unique_numbers
        self.contraction_matrix = contraction_matrix
        self.in_features = in_features
        self.out_features = out_features
        self.alchemical_features = len(contraction_matrix)
        self.weight = torch.nn.Parameter(
            torch.zeros(
                self.alchemical_features,
                in_features,
                out_features,
            )
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(self.alchemical_features, 1, out_features)
        )
        self.weight.data.normal_(mean=0.0, std=in_features ** (-0.5))

    def forward(self, tensormap: TensorMap) -> TensorMap:
        output_blocks: list[TensorBlock] = []
        for i, block in enumerate(tensormap.blocks()):
            one_hot_ai = torch.zeros(len(block.samples), len(self.unique_numbers))
            one_hot_ai[:, i] = 1.0
            pseudo_species_weights = one_hot_ai @ self.contraction_matrix.T
            features = block.values
            embedded_features = (
                features[None, ...] * pseudo_species_weights.T[:, :, None]
            )
            out = torch.bmm(embedded_features, self.weight) + self.bias
            out = torch.sum(out, dim=0)
            labels = Labels(
                names=["out_features_idx"],
                values=torch.arange(
                    self.out_features,
                    dtype=torch.int64,
                    device=block.values.device,
                ).reshape(-1, 1),
            )
            new_block = TensorBlock(
                values=out,
                samples=block.samples,
                components=block.components,
                properties=labels,
            )
            output_blocks.append(new_block)
        return TensorMap(keys=tensormap.keys, blocks=output_blocks)
