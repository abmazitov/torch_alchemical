import torch

from torch_alchemical.utils import (
    get_compositions_from_numbers,
    get_species_coupling_matrix,
)


def initialize_composition_layer_weights(model, datamodule, trainable=False):
    assert hasattr(model, "composition_layer")
    dataset = datamodule.train_dataset
    composition_layer = model.composition_layer
    numbers = torch.cat([data.numbers for data in dataset])
    batch = torch.cat(
        [
            torch.repeat_interleave(torch.tensor([i]), data.num_nodes)
            for i, data in enumerate(dataset)
        ]
    )
    compositions = torch.stack(
        get_compositions_from_numbers(numbers, datamodule.unique_numbers, batch)
    )
    bias = composition_layer.bias is not None
    if bias:
        compositions = torch.cat(
            (torch.ones(len(dataset)).view(-1, 1), compositions), dim=1
        )  # bias
    energies = torch.cat([data.energies.view(1, -1) for data in dataset], dim=0)
    weights = torch.linalg.lstsq(compositions, energies).solution
    if bias:
        composition_layer.bias = torch.nn.Parameter(
            weights[0].contiguous(), requires_grad=True
        )
        weights = weights[1:]
    composition_layer.weight = torch.nn.Parameter(
        weights.T.contiguous(), requires_grad=True
    )
    if not trainable:
        composition_layer.requires_grad_(False)
    print("Composition layer weights are initialized with least squares solution")


def initialize_energies_forces_scale_factor(
    model, datamodule, use_second_moment=True, trainable=False
):
    assert hasattr(model, "composition_layer")
    assert hasattr(model, "energies_scale_factor")
    dataset = datamodule.train_dataset
    composition_layer = model.composition_layer
    numbers = torch.cat([data.numbers for data in dataset])
    batch = torch.cat(
        [
            torch.repeat_interleave(torch.tensor([i]), data.num_nodes)
            for i, data in enumerate(dataset)
        ]
    )
    compositions = torch.stack(
        get_compositions_from_numbers(numbers, datamodule.unique_numbers, batch)
    )
    energies = torch.cat([data.energies.view(1, -1) for data in dataset], dim=0)
    composition_energies = composition_layer(compositions)
    if use_second_moment:
        scale = torch.sqrt(torch.mean((energies - composition_energies) ** 2))
        scale_description = "second moments"
    else:
        scale = torch.std(energies - composition_energies)
        scale_description = "standard deviation"
    model.energies_scale_factor = torch.nn.Parameter(scale, requires_grad=trainable)
    print(f"Energies scale is initialized with shifted energies {scale_description}")


def rescale_energies_and_forces(model, datamodule, scale_factor):
    dataset = datamodule.train_dataset
    composition_layer = model.composition_layer
    numbers = torch.cat([data.numbers for data in dataset])
    batch = torch.cat(
        [
            torch.repeat_interleave(torch.tensor([i]), data.num_nodes)
            for i, data in enumerate(dataset)
        ]
    )
    compositions = torch.stack(
        get_compositions_from_numbers(numbers, datamodule.unique_numbers, batch)
    )
    composition_energies = composition_layer(compositions)
    composition_energies = composition_energies.squeeze()
    for i, data in enumerate(dataset):
        data.energies = (data.energies - composition_energies[i]) / scale_factor
        data.forces = data.forces / scale_factor
    print("Training energies and forces are shifted and rescaled")


def initialize_combining_matrix(model, datamodule, trainable=True):
    assert hasattr(model, "ps_features_layer")
    vex_calculator = model.ps_features_layer.spex_calculator.vector_expansion_calculator
    radial_basis_calculator = vex_calculator.radial_basis_calculator
    n_pseudo_species = radial_basis_calculator.n_pseudo_species
    radial_basis_calculator.combination_matrix.weight = torch.nn.Parameter(
        get_species_coupling_matrix(
            datamodule.unique_numbers, n_pseudo_species
        ).contiguous(),
        requires_grad=trainable,
    )
    print("Combinining matrix is initialized manually")


def initialize_average_number_of_atoms(model, datamodule):
    assert hasattr(model, "average_number_of_atoms")
    dataset = datamodule.train_dataset
    model.average_number_of_atoms = torch.mean(
        torch.tensor([data.num_nodes for data in dataset]).to(torch.get_default_dtype())
    )
    print("Average number of atoms is initialized manually")
