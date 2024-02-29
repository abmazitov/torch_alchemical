import json

import torch

from torch_alchemical.models import AlchemicalModel
from torch_alchemical.tools.train import LitDataModule, LitModel
from torch_alchemical.tools.train.initialize import (
    get_average_number_of_atoms,
    get_composition_weights,
    get_energies_scale_factor,
    rescale_energies_and_forces,
)
from torch_alchemical.utils import get_compositions_from_numbers

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


class TestTools:
    with open("./tests/configs/default_datamodule_parameters.json", "r") as f:
        datamodule_parameters = json.load(f)
    with open("./tests/configs/default_model_parameters.json", "r") as f:
        default_model_parameters = json.load(f)
    with open("./tests/configs/default_litmodel_parameters.json", "r") as f:
        litmodel_parameters = json.load(f)

    def test_datamodule(self):
        datamodule = LitDataModule(**self.datamodule_parameters)
        datamodule.prepare_data()
        datamodule.setup()

    def test_initialization_utilities(self):
        datamodule = LitDataModule(**self.datamodule_parameters)
        datamodule.prepare_data()
        datamodule.setup()

        train_dataset = datamodule.train_dataset
        unique_numbers = datamodule.unique_numbers
        numbers = torch.cat([data.numbers for data in train_dataset])
        batch = torch.cat(
            [
                torch.repeat_interleave(torch.tensor([i]), data.num_nodes)
                for i, data in enumerate(train_dataset)
            ]
        )
        compositions = torch.stack(
            get_compositions_from_numbers(numbers, unique_numbers, batch)
        )
        composition_weights = get_composition_weights(train_dataset, compositions)
        energies_scale_factor = get_energies_scale_factor(
            train_dataset, compositions, composition_weights
        )
        average_number_of_atoms = get_average_number_of_atoms(train_dataset)
        average_number_of_neighbors = torch.mean(
            torch.tensor(
                [data.edge_index.shape[1] / data.pos.shape[0] for data in train_dataset]
            )
        )

        assert torch.allclose(average_number_of_atoms, torch.tensor(39.7200), atol=1e-4)
        assert torch.allclose(
            average_number_of_neighbors, torch.tensor(44.4383), atol=1e-4
        )
        assert torch.allclose(energies_scale_factor, torch.tensor(21.8422), atol=1e-4)
        assert torch.allclose(
            composition_weights,
            torch.tensor(
                [
                    [
                        -6.5062,
                        -8.3999,
                        -9.4898,
                        -8.8347,
                        -8.7395,
                        -7.4130,
                        -6.3196,
                        -5.4111,
                        -3.2030,
                        -1.4150,
                        -6.8346,
                        -8.6563,
                        -10.2193,
                        -10.5220,
                        -8.3924,
                        -5.8605,
                        -5.0343,
                        -2.4630,
                        -4.6872,
                        -10.5623,
                        -11.9004,
                        -12.7991,
                        -6.9026,
                        -4.9986,
                        -3.2470,
                    ]
                ]
            ),
            atol=1e-4,
        )

    def test_litmodel(self):
        datamodule = LitDataModule(**self.datamodule_parameters)
        datamodule.prepare_data()
        datamodule.setup()

        model = AlchemicalModel(
            unique_numbers=datamodule.unique_numbers,
            num_pseudo_species=4,
            contract_center_species=True,
            **self.default_model_parameters,
        )

        train_dataset = datamodule.train_dataset
        unique_numbers = datamodule.unique_numbers
        numbers = torch.cat([data.numbers for data in train_dataset])
        batch = torch.cat(
            [
                torch.repeat_interleave(torch.tensor([i]), data.num_nodes)
                for i, data in enumerate(train_dataset)
            ]
        )
        compositions = torch.stack(
            get_compositions_from_numbers(numbers, unique_numbers, batch)
        )
        composition_weights = get_composition_weights(train_dataset, compositions)
        energies_scale_factor = get_energies_scale_factor(
            train_dataset, compositions, composition_weights
        )
        average_number_of_atoms = get_average_number_of_atoms(train_dataset)
        average_number_of_neighbors = torch.mean(
            torch.tensor(
                [data.edge_index.shape[1] / data.pos.shape[0] for data in train_dataset]
            )
        )

        model.set_normalization_factor(average_number_of_atoms)
        model.set_energies_scale_factor(energies_scale_factor)
        model.set_basis_normalization_factor(average_number_of_neighbors)
        model.set_composition_weights(composition_weights)

        # Rescaling the energies and forces
        rescale_energies_and_forces(
            train_dataset, compositions, composition_weights, energies_scale_factor
        )

        _ = LitModel(model=model, **self.litmodel_parameters)
