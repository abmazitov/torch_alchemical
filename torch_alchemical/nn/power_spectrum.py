from typing import Union, Optional
import numpy as np
import torch
from torch_spex.spherical_expansions import SphericalExpansion
from torch_alchemical.utils import (
    get_torch_spex_dict,
    get_torch_spex_dict_from_data_lists,
)
from metatensor.torch import TensorBlock, TensorMap, Labels


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
        if isinstance(all_species, np.ndarray):
            all_species = all_species.tolist()
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
        positions: Union[torch.Tensor, list[torch.Tensor]],
        cells: Union[torch.Tensor, list[torch.Tensor]],
        numbers: Union[torch.Tensor, list[torch.Tensor]],
        edge_indices: Union[torch.Tensor, list[torch.Tensor]],
        edge_shifts: Union[torch.Tensor, list[torch.Tensor]],
        ptr: Optional[torch.Tensor] = None,
    ):
        if (
            isinstance(positions, torch.Tensor)
            and isinstance(cells, torch.Tensor)
            and isinstance(numbers, torch.Tensor)
            and isinstance(edge_indices, torch.Tensor)
            and isinstance(edge_shifts, torch.Tensor)
        ):
            assert ptr is not None
            batch_dict = get_torch_spex_dict(
                positions, cells, numbers, edge_indices, edge_shifts, ptr
            )
        elif (
            isinstance(positions, list)
            and isinstance(cells, list)
            and isinstance(numbers, list)
            and isinstance(edge_indices, list)
            and isinstance(edge_shifts, list)
        ):
            batch_dict = get_torch_spex_dict_from_data_lists(
                positions, cells, numbers, edge_indices, edge_shifts
            )
        else:
            raise ValueError()
        spex = self.spex_calculator(
            positions=batch_dict["positions"],
            cells=batch_dict["cells"],
            species=batch_dict["species"],
            cell_shifts=batch_dict["cell_shifts"],
            centers=batch_dict["centers"],
            pairs=batch_dict["pairs"],
            structure_centers=batch_dict["structure_centers"],
            structure_pairs=batch_dict["structure_pairs"],
            structure_offsets=batch_dict["structure_offsets"],
        )
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

    def forward(self, spex: TensorMap):
        keys: list[list[int]] = []
        blocks: list[TensorBlock] = []
        for a_i in self.all_species:
            ps_values_ai = []
            for l in range(self.l_max + 1):
                cg = (2 * l + 1) ** (-0.5)
                block_ai_l = spex.block({"lam": l, "a_i": a_i})
                c_ai_l = block_ai_l.values

                # same as this:
                # ps_ai_l = cg*torch.einsum("ima, imb -> iab", c_ai_l, c_ai_l)
                # but faster:
                ps_ai_l = cg * torch.sum(
                    c_ai_l.unsqueeze(2) * c_ai_l.unsqueeze(3), dim=1
                )

                ps_ai_l = ps_ai_l.reshape(
                    c_ai_l.shape[0], c_ai_l.shape[2] * c_ai_l.shape[2]
                )
                ps_values_ai.append(ps_ai_l)
            ps_values_ai = torch.concatenate(ps_values_ai, dim=-1)

            block = TensorBlock(
                values=ps_values_ai,
                samples=spex.block({"lam": 0, "a_i": a_i}).samples,
                components=[],
                properties=Labels(
                    names=("property",),
                    values=torch.arange(
                        ps_values_ai.shape[-1], device=ps_values_ai.device
                    ).reshape(-1, 1),
                ),
            )
            keys.append([a_i])
            blocks.append(block)

        power_spectrum = TensorMap(
            keys=Labels(
                names=("a_i",),
                values=torch.tensor(keys),  # .reshape((-1, 2)),
            ).to(ps_values_ai.device),
            blocks=blocks,
        )

        return power_spectrum
