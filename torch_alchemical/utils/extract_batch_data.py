from torch_geometric.data import Data


def extract_batch_data(batch: list[Data]):
    positions = [data.pos for data in batch]
    cells = [data.cell for data in batch]
    numbers = [data.numbers for data in batch]
    edge_indices = [data.edge_index for data in batch]
    edge_shifts = [data.edge_shift for data in batch]
    return positions, cells, numbers, edge_indices, edge_shifts
