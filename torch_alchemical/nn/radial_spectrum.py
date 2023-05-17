from typing import Union

import numpy as np
import torch
from torch_geometric.data import Batch

from .calculators import RadialSpectrumCalculator


class RadialSpectrumFeatures(torch.nn.Module):
    def __init__(
        self,
        all_species: Union[list, np.ndarray],
        cutoff_radius: float,
        basis_cutoff: float,
        device: torch.device = None,
    ):
        super().__init__()
        self.all_species = all_species
        self.cutoff_radius = cutoff_radius
        self.basis_cutoff = basis_cutoff
        self.device = device

        self.radial_spectrum_calculator = RadialSpectrumCalculator(
            all_species,
            cutoff_radius=cutoff_radius,
            basis_cutoff=basis_cutoff,
            device=device,
        )

    def forward(self, batch: Batch):
        radial_spectrum = self.radial_spectrum_calculator(batch)
        return radial_spectrum

    @property
    def num_features(self):
        calculator = self.radial_spectrum_calculator
        n_max_l = calculator.radial_basis_calculator.n_max_l
        n_feat = len(self.all_species) * sum(n_max_l)
        return n_feat
