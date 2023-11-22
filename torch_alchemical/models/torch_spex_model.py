import numpy as np
import torch
from torch_spex.spherical_expansions import SphericalExpansion
from torch_spex.normalize import (
    normalize_true,
    normalize_false,
)
import metatensor.torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from typing import Union
from torch_alchemical.utils import get_torch_spex_dict
from torch_alchemical.utils import get_compositions_from_numbers


def normalize_ps(ps):
    new_keys = []
    new_blocks = []
    for key, block in ps.items():
        new_keys.append(key.values)
        values = block.values
        mean = torch.mean(values, dim=-1, keepdim=True)
        centered_values = values - mean
        variance = torch.mean(centered_values**2, dim=-1, keepdim=True)
        new_values = centered_values / torch.sqrt(variance)
        new_blocks.append(
            TensorBlock(
                values=new_values,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
        )
    return TensorMap(
        keys=Labels(
            names=("a_i",),
            values=torch.tensor(new_keys, device=new_blocks[0].values.device).reshape(
                -1, 1
            ),
        ),
        blocks=new_blocks,
    )


class TorchSpexModel(torch.nn.Module):
    def __init__(
        self,
        hidden_sizes: int,
        output_size: int,
        unique_numbers: Union[list, np.ndarray],
        cutoff: float,
        basis_cutoff_power_spectrum: float,
        radial_basis_type: str,
        energies_scale_factor: float = 1.0,
        normalize: bool = False,
        basis_normalization_factor: float = None,
        average_number_of_atoms: float = 1.0,
        trainable_basis: bool = True,
        basis_scale: float = 3.0,
        num_pseudo_species: int = None,
        device: torch.device = None,
    ) -> None:
        super().__init__()
        self.unique_numbers = unique_numbers
        self.composition_layer = torch.nn.Linear(
            len(unique_numbers), output_size, bias=False
        )
        self.n_pseudo = num_pseudo_species
        self.normalize = normalize
        self.normalize_func = normalize_true if normalize else normalize_false
        self.average_number_of_atoms = average_number_of_atoms
        self.energies_scale_factor = energies_scale_factor
        assert basis_normalization_factor is not None
        hypers = {
            "alchemical": num_pseudo_species,
            "normalize": basis_normalization_factor,
            "cutoff radius": cutoff,
            "radial basis": {
                "type": radial_basis_type,
                "E_max": basis_cutoff_power_spectrum,
                "mlp": trainable_basis,
                "scale": basis_scale,
                "cost_trade_off": False,
            },
        }
        self.spherical_expansion_calculator = SphericalExpansion(
            hypers, unique_numbers, device=device
        )
        self.device = device
        n_max = (
            self.spherical_expansion_calculator.vector_expansion_calculator.radial_basis_calculator.n_max_l
        )
        print(n_max)
        l_max = len(n_max) - 1
        n_feat = sum([n_max[l] ** 2 * self.n_pseudo**2 for l in range(l_max + 1)])
        self.ps_calculator = PowerSpectrum(l_max, unique_numbers)
        self.combination_matrix = (
            self.spherical_expansion_calculator.vector_expansion_calculator.radial_basis_calculator.combination_matrix
        )
        self.all_species_labels = metatensor.torch.Labels(
            names=["a_i"],
            values=torch.tensor(unique_numbers, device=device).reshape(-1, 1),
        )
        assert len(hidden_sizes) > 0
        self.nu2_model = torch.nn.ModuleDict()
        for alpha_i in range(self.n_pseudo):
            layers = [
                self.normalize_func(
                    "linear_no_bias",
                    torch.nn.Linear(n_feat, hidden_sizes[0], bias=False),
                ),
                self.normalize_func("activation", torch.nn.SiLU()),
            ]
            for i in range(1, len(hidden_sizes)):
                layers.append(
                    self.normalize_func(
                        "linear_no_bias",
                        torch.nn.Linear(
                            hidden_sizes[i - 1], hidden_sizes[i], bias=False
                        ),
                    )
                )
                layers.append(self.normalize_func("activation", torch.nn.SiLU()))
            layers.append(
                self.normalize_func(
                    "linear_no_bias",
                    torch.nn.Linear(hidden_sizes[-1], output_size, bias=False),
                )
            )
            self.nu2_model[str(alpha_i)] = torch.nn.Sequential(*layers)
        # """
        # self.zero_body_energies = torch.nn.Parameter(torch.zeros(len(all_species)))

    def forward(
        self,
        positions: torch.Tensor,
        cells: torch.Tensor,
        numbers: torch.Tensor,
        edge_indices: torch.Tensor,
        edge_offsets: torch.Tensor,
        batch: torch.Tensor,
    ):
        structures = get_torch_spex_dict(
            positions=positions,
            cells=cells,
            numbers=numbers,
            edge_indices=edge_indices,
            edge_offsets=edge_offsets,
            batch=batch,
        )

        n_structures = len(structures["cells"])
        energies = torch.zeros(
            (n_structures,), device=self.device, dtype=torch.get_default_dtype()
        )

        # print("Calculating spherical expansion")
        spherical_expansion = self.spherical_expansion_calculator(**structures)
        ps = self.ps_calculator(spherical_expansion)
        if self.normalize:
            ps = normalize_ps(ps)

        # print("Calculating energies")
        self._apply_layer(energies, ps, self.nu2_model)
        energies = energies.view(-1, 1)
        if self.normalize:
            energies = energies / self.average_number_of_atoms
        # print("Final", torch.mean(energies), get_2_mom(energies))
        # energies += comp @ self.zero_body_energies

        # print("Computing forces by backpropagation")
        # if self.do_forces:
        #     forces = compute_forces(energies, positions, is_training=is_training)
        # else:
        #     forces = None  # Or zero-dimensional tensor?

        if self.training:
            return energies
        else:
            compositions = torch.stack(
                get_compositions_from_numbers(
                    numbers,
                    self.unique_numbers,
                    ptr,
                    self.composition_layer.weight.dtype,
                )
            )
            energies = energies * self.energies_scale_factor + self.composition_layer(
                compositions
            )
            return energies

    def _apply_layer(self, energies, tmap, layer):
        atomic_energies = []
        structure_indices = []
        # print(tmap.block(0).values)
        tmap = tmap.keys_to_samples("a_i")
        block = tmap.block()
        # print(block.values)
        samples = block.samples
        one_hot_ai = torch.tensor(
            metatensor.torch.one_hot(samples, self.all_species_labels),
            dtype=torch.get_default_dtype(),
            device=block.values.device,
        )
        pseudo_species_weights = self.combination_matrix(one_hot_ai)
        features = block.values.squeeze(dim=1)
        # print("features", torch.mean(features), get_2_mom(features))
        embedded_features = features[:, :, None] * pseudo_species_weights[:, None, :]
        atomic_energies = torch.zeros(
            (block.values.shape[0],),
            dtype=torch.get_default_dtype(),
            device=block.values.device,
        )
        for alpha_i in range(self.n_pseudo):
            atomic_energies += layer[str(alpha_i)](
                embedded_features[:, :, alpha_i]
            ).squeeze(dim=-1)
            # print("individual", torch.mean(layer[str(alpha_i)](embedded_features[:, :, alpha_i]).squeeze(dim=-1)), get_2_mom(layer[str(alpha_i)](embedded_features[:, :, alpha_i]).squeeze(dim=-1)))
        if self.normalize:
            atomic_energies = atomic_energies / np.sqrt(self.n_pseudo)
        # print("total", torch.mean(atomic_energies), get_2_mom(atomic_energies))
        structure_indices = block.samples["structure"]
        energies.index_add_(
            dim=0, index=structure_indices.to(self.device), source=atomic_energies
        )
        # print("in", torch.mean(energies), get_2_mom(energies))
        # THIS IN-PLACE MODIFICATION HAS TO CHANGE!

    # def print_state()... Would print loss, train errors, validation errors, test errors, ...


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
                block_ai_l = spex.block({"lam": l, "a_i": a_i})
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
                values=torch.tensor(
                    keys, device=blocks[0].values.device
                ),  # .reshape((-1, 2)),
            ),
            blocks=blocks,
        )

        return power_spectrum
