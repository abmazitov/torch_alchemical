import json

import torch

from torch_alchemical.models import AlchemicalModel
from torch_alchemical.tools.train import LitDataModule, LitModel
from torch_alchemical.tools.train.initialize import (
    initialize_average_number_of_atoms,
    initialize_combining_matrix,
    initialize_composition_layer_weights,
    initialize_energies_forces_scale_factor,
)

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

    def test_litmodel(self):
        datamodule = LitDataModule(**self.datamodule_parameters)
        datamodule.prepare_data()
        datamodule.setup()

        model = AlchemicalModel(
            unique_numbers=datamodule.unique_numbers,
            num_pseudo_species=4,
            **self.default_model_parameters
        )
        initialize_composition_layer_weights(model, datamodule)
        initialize_combining_matrix(model, datamodule)
        initialize_average_number_of_atoms(model, datamodule)
        initialize_energies_forces_scale_factor(model, datamodule)
        _ = LitModel(model=model, **self.litmodel_parameters)
