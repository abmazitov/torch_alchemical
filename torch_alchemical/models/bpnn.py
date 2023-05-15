import equistore
import torch

from torch_alchemical.nn import LinearMap, PowerSpectrumFeatures, SiLU


class BPNN(torch.nn.Module):
    def __init__(
        self,
        hidden_size_1,
        hidden_size_2,
        output_size,
        unique_numbers,
        cutoff_radius,
        cutoff_energy,
        num_pseudo_species,
        device=None,
    ):
        super().__init__()
        self.composition_layer = torch.nn.Linear(len(unique_numbers), output_size)
        self.features_layer = PowerSpectrumFeatures(
            unique_numbers, cutoff_radius, cutoff_energy, num_pseudo_species, device
        )
        self.device = device
        input_size = self.features_layer.num_features
        keys = unique_numbers.astype(str).tolist()
        self.linear = torch.nn.Sequential(
            LinearMap(keys, input_size, hidden_size_1),
            SiLU(),
            LinearMap(keys, hidden_size_1, hidden_size_1),
            SiLU(),
            LinearMap(keys, hidden_size_1, hidden_size_2),
            SiLU(),
            LinearMap(keys, hidden_size_2, output_size),
        )

    def forward(self, batch, training=True):
        energies = self.composition_layer(batch.composition)
        x = self.features_layer(batch)
        x = self.linear(x)
        energies += (
            equistore.sum_over_samples(x.keys_to_samples("a_i"), ["center", "a_i"])
            .block()
            .values
        )
        forces = -torch.autograd.grad(
            energies,
            batch.pos,
            grad_outputs=torch.ones_like(energies),
            create_graph=training,
            retain_graph=training,
        )[0]
        return energies, forces
