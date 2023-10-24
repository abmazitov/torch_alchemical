from torch_alchemical.models import PowerSpectrumModel
from torch_alchemical.tools.train import LitDataModule, LitModel
import json
import torch


torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


class TestTools:
    with open("./tests/configs/default_datamodule_parameters.json", "r") as f:
        datamodule_parameters = json.load(f)
    with open("./tests/configs/ps_model_parameters.json", "r") as f:
        model_parameters = json.load(f)
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

        model = PowerSpectrumModel(
            unique_numbers=datamodule.unique_numbers, **self.model_parameters
        )
        litmodel = LitModel(model=model, **self.litmodel_parameters)
        litmodel.initialize_composition_layer_weights(litmodel.model, datamodule)
        litmodel.initialize_combining_matrix(litmodel.model, datamodule)
