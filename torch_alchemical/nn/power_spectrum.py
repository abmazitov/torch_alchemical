from typing import Union

import numpy as np
import torch
from torch_geometric.data import Batch

from .calculators import PowerSpectrumCalculator, SphericalExpansionCalculator


class PowerSpectrumFeatures(torch.nn.Module):
    def __init__(
        self,
        all_species: Union[list, np.ndarray],
        cutoff_radius: float,
        basis_cutoff: float,
        num_pseudo_species: int = None,
        device: torch.device = None,
    ):
        super().__init__()
        self.all_species = all_species
        self.cutoff_radius = cutoff_radius
        self.basis_cutoff = basis_cutoff
        self.num_pseudo_species = num_pseudo_species
        self.device = device

        self.spex_calculator = SphericalExpansionCalculator(
            all_species,
            cutoff_radius=cutoff_radius,
            basis_cutoff=basis_cutoff,
            num_pseudo_species=num_pseudo_species,
        )
        self.ps_calculator = PowerSpectrumCalculator(all_species)

    def forward(self, batch: Batch):
        spex = self.spex_calculator(batch)
        power_spectrum = self.ps_calculator(spex)
        return power_spectrum

    @property
    def num_features(self):
        vex_calculator = self.spex_calculator.vector_expansion_calculator
        n_max = vex_calculator.radial_basis_calculator.n_max_l
        l_max = len(n_max) - 1
        n_feat = sum(
            [n_max[l] ** 2 * self.num_pseudo_species**2 for l in range(l_max + 1)]
        )
        return n_feat
