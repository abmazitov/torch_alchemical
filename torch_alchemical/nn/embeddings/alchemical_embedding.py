import torch
from metatensor.torch import TensorBlock, TensorMap, Labels


class AlchemicalEmbedding(torch.nn.Module):
    def __init__(
        self,
        unique_numbers: list[int],
        contraction_matrix: torch.Tensor,
    ):
        super().__init__()
        self.unique_numbers = unique_numbers
        self.contraction_matrix = contraction_matrix
        self.num_pseudo_species = len(contraction_matrix)

    def forward(self, tensormap: TensorMap) -> TensorMap:
        output_blocks: list[TensorBlock] = []
        for i, block in enumerate(tensormap.blocks()):
            assert not block.components
            one_hot_ai = torch.zeros(len(block.samples), len(self.unique_numbers))
            one_hot_ai[:, i] = 1.0
            pseudo_species_weights = one_hot_ai @ self.contraction_matrix.T
            features = block.values
            embedded_features = pseudo_species_weights[..., None] * features[:, None, :]
            components = Labels(
                names=["pseudo_species"],
                values=torch.arange(
                    self.num_pseudo_species,
                    dtype=torch.int64,
                    device=block.values.device,
                ).reshape(-1, 1),
            )
            new_block = TensorBlock(
                values=embedded_features,
                samples=block.samples,
                components=[components],
                properties=block.properties,
            )
            output_blocks.append(new_block)
        return TensorMap(keys=tensormap.keys, blocks=output_blocks)
