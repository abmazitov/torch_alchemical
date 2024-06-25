import torch

def split_edges_by_batch(edge_indices, edge_offsets, batch):
    # Get unique batch IDs
    unique_batches = torch.unique(batch, sorted = False)
    
    # Initialize lists to store results
    batched_edge_indices = []
    batched_edge_offsets = []

    # Iterate over each unique batch ID
    for b in unique_batches:
        # Get the indices of edges belonging to the current batch
        batch_indices = (batch == b).nonzero(as_tuple=True)[0]

        # Extract the edges for the current batch
        batched_edge_indices.append(edge_indices[:, batch_indices])
        batched_edge_offsets.append(edge_offsets[batch_indices])

    return batched_edge_indices, batched_edge_offsets