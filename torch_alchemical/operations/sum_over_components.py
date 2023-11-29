from metatensor.torch import TensorBlock, TensorMap


def sum_over_components(tensormap: TensorMap) -> TensorMap:
    output_blocks: list[TensorBlock] = []
    for block in tensormap.blocks():
        output_blocks.append(sum_over_components_block(block))
    return TensorMap(blocks=output_blocks, keys=tensormap.keys)


def sum_over_components_block(block: TensorBlock) -> TensorBlock:
    assert len(block.components) == 1  # only a single components group is supported
    new_block = TensorBlock(
        values=block.values.sum(dim=1),
        samples=block.samples,
        components=[],
        properties=block.properties,
    )
    return new_block
