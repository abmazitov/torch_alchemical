import equistore
import torch

from torch_alchemical.nn import Linear, PowerSpectrumFeatures, SiLU


class FCNN(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
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
        input_size = self.features_layer.num_features
        self.linear1 = Linear(input_size, hidden_size)
        self.linear2 = Linear(hidden_size, hidden_size)
        self.linear3 = Linear(hidden_size, output_size)

    def forward(self, batch, training=True):
        energies = self.composition_layer(batch.composition)
        x = self.features_layer(batch)
        x = self.linear1(x)
        x = SiLU()(x)
        x = self.linear2(x)
        x = SiLU()(x)
        x = self.linear3(x)
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
