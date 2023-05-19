from typing import Union

import equistore
import numpy as np
import torch

from torch_alchemical.nn import (
    Linear,
    PowerSpectrumFeatures,
    RadialSpectrumFeatures,
    SiLU,
)


class FCNN(torch.nn.Module):
    def __init__(
        self,
        hidden_sizes: int,
        output_size: int,
        unique_numbers: Union[list, np.ndarray],
        cutoff: float,
        basis_cutoff_radial_spectrum: float,
        basis_cutoff_power_spectrum: float,
        num_pseudo_species: int = None,
        device: torch.device = None,
    ):
        super().__init__()
        self.composition_layer = torch.nn.Linear(len(unique_numbers), output_size)
        self.rs_features_layer = RadialSpectrumFeatures(
            unique_numbers, cutoff, basis_cutoff_radial_spectrum, device
        )
        self.ps_features_layer = PowerSpectrumFeatures(
            unique_numbers,
            cutoff,
            basis_cutoff_power_spectrum,
            num_pseudo_species,
            device,
        )
        rs_input_size = self.rs_features_layer.num_features
        ps_input_size = self.ps_features_layer.num_features
        input_size = rs_input_size + ps_input_size
        layer_size = [input_size] + hidden_sizes + [output_size]
        layers = []
        for layer_index in range(1, len(layer_size)):
            layers.append(Linear(layer_size[layer_index - 1], layer_size[layer_index]))
            layers.append(SiLU())
        self.nn = torch.nn.Sequential(*layers)

    def forward(self, batch, training=True):
        energies = self.composition_layer(batch.composition)
        rs = self.rs_features_layer(batch)
        ps = self.ps_features_layer(batch)
        x = equistore.join((rs, ps), axis='properties')
        x = self.nn(x)
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
