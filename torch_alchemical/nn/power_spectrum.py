from typing import Union
import numpy as np
import torch
from torch_spex.spherical_expansions import SphericalExpansion
from torch_alchemical.utils import get_torch_spex_dict
from equistore import TensorBlock, TensorMap, Labels


class PowerSpectrumFeatures(torch.nn.Module):
    def __init__(
        self,
        all_species: Union[list, np.ndarray],
        cutoff_radius: float,
        basis_cutoff: float,
        radial_basis_type: str = "le",
        basis_normalization_factor: float = None,
        trainable_basis: bool = True,
        num_pseudo_species: int = None,
        device: torch.device = None,
    ):
        super().__init__()
        self.all_species = all_species
        self.cutoff_radius = cutoff_radius
        self.basis_cutoff = basis_cutoff
        self.radial_basis_type = radial_basis_type
        self.basis_normalization_factor = basis_normalization_factor
        self.trainable_basis = trainable_basis
        self.num_pseudo_species = num_pseudo_species
        self.device = device
        hypers = {
            "cutoff radius": self.cutoff_radius,
            "radial basis": {
                "type": self.radial_basis_type,
                "E_max": self.basis_cutoff,
                "mlp": self.trainable_basis,
                "scale": 3.0,
                "cost_trade_off": False,
            },
        }
        if self.num_pseudo_species is not None:
            hypers["alchemical"] = self.num_pseudo_species
        if self.basis_normalization_factor:
            hypers["normalize"] = self.basis_normalization_factor
        self.spex_calculator = SphericalExpansion(
            hypers=hypers,
            all_species=self.all_species,
            device=self.device,
        )
        self.l_max = self.spex_calculator.vector_expansion_calculator.l_max
        self.ps_calculator = PowerSpectrum(self.l_max, all_species)

    def forward(
        self,
        positions: list[torch.Tensor],
        cells: list[torch.Tensor],
        numbers: list[torch.Tensor],
        edge_indices: list[torch.Tensor],
        edge_shifts: list[torch.Tensor],
    ):
        batch_dict = get_torch_spex_dict(
            positions, cells, numbers, edge_indices, edge_shifts
        )
        spex = self.spex_calculator(**batch_dict)
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


class PowerSpectrum(torch.nn.Module):
    def __init__(self, l_max, all_species):
        super(PowerSpectrum, self).__init__()

        self.l_max = l_max
        self.all_species = all_species

    def forward(self, spex):
        keys = []
        blocks = []
        for a_i in self.all_species:
            ps_values_ai = []
            for l in range(self.l_max + 1):
                cg = 1.0 / np.sqrt(2 * l + 1)
                block_ai_l = spex.block(lam=l, a_i=a_i)
                c_ai_l = block_ai_l.values

                # same as this:
                # ps_ai_l = cg*torch.einsum("ima, imb -> iab", c_ai_l, c_ai_l)
                # but faster:
                ps_ai_l = cg * torch.sum(
                    c_ai_l.unsqueeze(2) * c_ai_l.unsqueeze(3), dim=1
                )

                ps_ai_l = ps_ai_l.reshape(c_ai_l.shape[0], c_ai_l.shape[2] ** 2)
                ps_values_ai.append(ps_ai_l)
            ps_values_ai = torch.concatenate(ps_values_ai, dim=-1)

            block = TensorBlock(
                values=ps_values_ai,
                samples=block_ai_l.samples,
                components=[],
                properties=Labels.range("property", ps_values_ai.shape[-1]),
            )
            keys.append([a_i])
            blocks.append(block)

        power_spectrum = TensorMap(
            keys=Labels(
                names=("a_i",),
                values=np.array(keys),  # .reshape((-1, 2)),
            ),
            blocks=blocks,
        )

        return power_spectrum
