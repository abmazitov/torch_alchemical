import torch
from torch_spex.spherical_expansions import SphericalExpansion
from torch_alchemical.nn.power_spectrum import PowerSpectrum
from torch_spex.normalize import normalize_false as normalize_func
from torch_spex.atomic_composition import AtomicComposition
from torch_spex.forces import compute_forces


class TorchSpexModel(torch.nn.Module):
    def __init__(self, hypers, hidden_size, all_species, do_forces, device) -> None:
        super().__init__()
        self.all_species = all_species
        self.spherical_expansion_calculator = SphericalExpansion(
            hypers, all_species, device=device
        )
        n_max = (
            self.spherical_expansion_calculator.vector_expansion_calculator.radial_basis_calculator.n_max_l
        )
        l_max = len(n_max) - 1
        n_feat = sum(
            [n_max[l] ** 2 * hypers["alchemical"] ** 2 for l in range(l_max + 1)]
        )
        self.ps_calculator = PowerSpectrum(l_max, all_species)
        """
        self.nu2_model = torch.nn.ModuleDict({
            str(a_i): torch.nn.Linear(n_feat, 1, bias=False) for a_i in self.all_species
        })
        """
        self.nu2_model = torch.nn.ModuleDict(
            {
                str(a_i): torch.nn.Sequential(
                    normalize_func(
                        "linear_no_bias",
                        torch.nn.Linear(n_feat, hidden_size, bias=False),
                    ),
                    normalize_func("activation", torch.nn.SiLU()),
                    normalize_func(
                        "linear_no_bias",
                        torch.nn.Linear(hidden_size, hidden_size, bias=False),
                    ),
                    normalize_func("activation", torch.nn.SiLU()),
                    normalize_func(
                        "linear_no_bias", torch.nn.Linear(hidden_size, 1, bias=False)
                    ),
                )
                for a_i in self.all_species
            }
        )
        # """
        self.comp_calculator = AtomicComposition(all_species)
        self.composition_coefficients = None  # Needs to be set from outside
        self.do_forces = do_forces

    def forward(
        self, structure_batch: dict[str, torch.Tensor], is_training: bool = True
    ):
        n_structures = structure_batch["cells"].shape[0]
        energies = torch.zeros(
            (n_structures,),
            dtype=structure_batch["positions"].dtype,
            device=structure_batch["positions"].device,
        )

        if self.do_forces:
            structure_batch["positions"].requires_grad_(True)

        # print("Calculating spherical expansion")
        spherical_expansion = self.spherical_expansion_calculator(
            positions=structure_batch["positions"],
            cells=structure_batch["cells"],
            species=structure_batch["species"],
            cell_shifts=structure_batch["cell_shifts"],
            centers=structure_batch["centers"],
            pairs=structure_batch["pairs"],
            structure_centers=structure_batch["structure_centers"],
            structure_pairs=structure_batch["structure_pairs"],
            structure_offsets=structure_batch["structure_offsets"],
        )
        ps = self.ps_calculator(spherical_expansion)

        # print("Calculating energies")
        atomic_energies = []
        structure_indices = []
        for ai, layer_ai in self.nu2_model.items():
            block = ps.block({"a_i": int(ai)})
            # print(block.values)
            features = block.values.squeeze(dim=1)
            structure_indices.append(block.samples.column("structure"))
            atomic_energies.append(layer_ai(features).squeeze(dim=-1))
        atomic_energies = torch.concat(atomic_energies)
        structure_indices = torch.concatenate(structure_indices)
        # print("Before aggregation", torch.mean(atomic_energies), get_2_mom(atomic_energies))
        energies.index_add_(dim=0, index=structure_indices, source=atomic_energies)

        comp = self.comp_calculator(
            positions=structure_batch["positions"],
            cells=structure_batch["cells"],
            species=structure_batch["species"],
            cell_shifts=structure_batch["cell_shifts"],
            centers=structure_batch["centers"],
            pairs=structure_batch["pairs"],
            structure_centers=structure_batch["structure_centers"],
            structure_pairs=structure_batch["structure_pairs"],
            structure_offsets=structure_batch["structure_offsets"],
        )
        energies += comp @ self.composition_coefficients

        # print("Computing forces by backpropagation")
        if self.do_forces:
            forces = compute_forces(
                energies, structure_batch["positions"], is_training=is_training
            )
        else:
            forces = None  # Or zero-dimensional tensor?

        return energies, forces


def predict_epoch(model, data_loader):
    predicted_energies = []
    predicted_forces = []
    for batch in data_loader:
        batch.pop("energies")
        batch.pop("forces")
        predicted_energies_batch, predicted_forces_batch = model(
            batch, is_training=False
        )
        predicted_energies.append(predicted_energies_batch)
        predicted_forces.append(predicted_forces_batch)

    predicted_energies = torch.concatenate(predicted_energies, dim=0)
    predicted_forces = torch.concatenate(predicted_forces, dim=0)
    return predicted_energies, predicted_forces


def set_composition_coefficients(
    model, predict_train_data_loader, train_structures, all_species
):
    train_energies = torch.tensor(
        [structure.info["energy"] for structure in train_structures]
    )
    comp_calculator = AtomicComposition(all_species)
    train_comp = []
    for batch in predict_train_data_loader:
        train_comp.append(comp_calculator(**batch))
    train_comp = torch.concatenate(train_comp)
    c_comp = torch.linalg.solve(
        train_comp.T @ train_comp, train_comp.T @ train_energies
    )
    model.composition_coefficients = c_comp
