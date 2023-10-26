from torch_alchemical.models import PowerSpectrumModel
from torch_alchemical.tools.train import LitDataModule, LitModel
from torch_alchemical.tools.train.initialize import (
    initialize_combining_matrix,
    initialize_composition_layer_weights,
)
import json
import torch
import lightning.pytorch as pl
import warnings

warnings.filterwarnings("ignore")


torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


class TestTrainingRoutine:
    with open("./tests/configs/default_datamodule_parameters.json", "r") as f:
        datamodule_parameters = json.load(f)
    with open("./tests/configs/ps_model_parameters.json", "r") as f:
        model_parameters = json.load(f)
    with open("./tests/configs/default_litmodel_parameters.json", "r") as f:
        litmodel_parameters = json.load(f)

    def test_training_routine(self):
        datamodule = LitDataModule(**self.datamodule_parameters)
        datamodule.prepare_data()
        datamodule.setup()

        model = PowerSpectrumModel(
            unique_numbers=datamodule.unique_numbers, **self.model_parameters
        )
        litmodel = LitModel(model=model, **self.litmodel_parameters)
        initialize_composition_layer_weights(litmodel.model, datamodule)
        initialize_combining_matrix(litmodel.model, datamodule)

        trainer = pl.Trainer(
            max_steps=1,
            enable_model_summary=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
        )
        trainer.fit(litmodel, datamodule)
