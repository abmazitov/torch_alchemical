import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


class AlchemicalEmbedding(torch.nn.Module):
    def __init__(
        self,
        unique_numbers: list[int],
        contraction_layer: torch.nn.Module,
    ):
        super().__init__()
        self.unique_numbers = unique_numbers
        self.contraction_layer = contraction_layer

    def forward(self, tensormap: TensorMap) -> TensorMap:
        output_blocks: list[TensorBlock] = []
        output_key_values: list[torch.Tensor] = []
        for i, (key, block) in enumerate(tensormap.items()):
            assert not block.components
            one_hot_ai = torch.zeros(
                len(block.samples),
                len(self.unique_numbers),
                device=block.values.device,
                dtype=block.values.dtype,
            )
            one_hot_ai[:, i] = 1.0
            pseudo_species_weights = self.contraction_layer(one_hot_ai)
            features = block.values
            embedded_features = pseudo_species_weights[..., None] * features[:, None, :]
            for j in range(pseudo_species_weights.shape[1]):
                new_block = TensorBlock(
                    values=embedded_features[:, j, :].contiguous(),
                    samples=block.samples,
                    components=[],
                    properties=block.properties,
                )
                output_blocks.append(new_block)
                output_key_values.append(
                    torch.cat(
                        (key.values, torch.tensor([j], device=block.values.device))
                    )
                )
        keys = Labels(
            names=tensormap.keys.names + ["b_i"],
            values=torch.stack(output_key_values),
        )
        return TensorMap(keys=keys, blocks=output_blocks)
