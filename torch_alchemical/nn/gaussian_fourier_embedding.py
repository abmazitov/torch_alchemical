from typing import List

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
class GaussianFourierEmbedding(torch.nn.Module):
    def __init__(self, input_dim, embed_dim, scale=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.embeddings = torch.randn(input_dim, embed_dim) * scale * torch.pi * 2

    def forward(self, input):
        res = input @ self.embeddings.to(input.device)
        return torch.cat([torch.sin(res), torch.cos(res)], dim=-1)


# Define Gaussian Fourier Embedding
class GaussianFourierEmbeddingTensor(torch.nn.Module):
    def __init__(self, keys, input_dim, embed_dim, scale=1.0):
        super().__init__()
        self.keys = keys
        self.emb = torch.nn.ModuleDict()
        for i in range(len(keys)):
            embeddings = GaussianFourierEmbedding(input_dim, embed_dim, scale)
            self.emb[str(i)] = embeddings

    def forward(self, tensormap: TensorMap) -> TensorMap:
        output_blocks: List[TensorBlock] = []
        for key, emb in self.emb.items():
            block = tensormap.block(int(key))
            labels = Labels(
                names=["property"],
                values=torch.arange(
                    emb.embed_dim * 2, dtype=torch.int64, device=block.values.device
                ).reshape(-1, 1),
            )
            new_block = TensorBlock(
                values=emb(block.values),
                samples=block.samples,
                components=block.components,
                properties=labels,
            )
            output_blocks.append(new_block)
        return TensorMap(keys=tensormap.keys, blocks=output_blocks)