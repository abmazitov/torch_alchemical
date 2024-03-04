import json
import warnings

import lightning.pytorch as pl
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

warnings.filterwarnings("ignore")


torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


with open("./tests/configs/default_datamodule_parameters.json", "r") as f:
    datamodule_parameters = json.load(f)
with open("./tests/configs/default_model_parameters.json", "r") as f:
    default_model_parameters = json.load(f)
with open("./tests/configs/default_litmodel_parameters.json", "r") as f:
    litmodel_parameters = json.load(f)


def test_training_routine():
    datamodule = LitDataModule(**datamodule_parameters)
    datamodule.prepare_data()
    datamodule.setup()

    model = AlchemicalModel(
        unique_numbers=datamodule.unique_numbers,
        num_pseudo_species=4,
        contract_center_species=True,
        **default_model_parameters,
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

    litmodel = LitModel(model=model, **litmodel_parameters)

    trainer = pl.Trainer(
        max_steps=2,
        enable_model_summary=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )
    trainer.fit(litmodel, datamodule)
