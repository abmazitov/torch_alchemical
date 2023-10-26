import torch
from torch_alchemical.utils import (
    get_compositions_from_numbers,
    get_species_coupling_matrix,
)


def initialize_composition_layer_weights(model, datamodule):
    assert hasattr(model, "composition_layer")
    dataset = datamodule.train_dataset
    composition_layer = model.composition_layer
    numbers = torch.cat([data.numbers for data in dataset])
    ptr = torch.cumsum(torch.tensor([0] + [data.num_nodes for data in dataset]), dim=0)
    compositions = torch.stack(
        get_compositions_from_numbers(numbers, datamodule.unique_numbers, ptr)
    )
    compositions = torch.cat(
        (torch.ones(len(dataset)).view(-1, 1), compositions), dim=1
    )  # bias
    energies = torch.cat([data.energies.view(1, -1) for data in dataset], dim=0)
    weights = torch.linalg.lstsq(compositions, energies).solution
    composition_layer.weight = torch.nn.Parameter(
        weights[1:].T.contiguous(), requires_grad=True
    )
    composition_layer.bias = torch.nn.Parameter(
        weights[0].contiguous(), requires_grad=True
    )
    print("Composition layer weights are initialized with least squares solution")


def initialize_combining_matrix(model, datamodule):
    assert hasattr(model, "ps_features_layer")
    radial_basis_calculator = (
        model.ps_features_layer.spex_calculator.vector_expansion_calculator.radial_basis_calculator
    )
    n_pseudo_species = radial_basis_calculator.n_pseudo_species
    model.ps_features_layer.spex_calculator.vector_expansion_calculator.radial_basis_calculator.combination_matrix.weight = torch.nn.Parameter(
        get_species_coupling_matrix(
            datamodule.unique_numbers, n_pseudo_species
        ).contiguous(),
        requires_grad=True,
    )
    print("Combinining matrix is initialized manually")
