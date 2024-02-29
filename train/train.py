from torch_alchemical.models import AlchemicalModel, BPPSModel
from torch_alchemical.tools.train import LitDataModule, LitModel
from torch_alchemical.utils import get_compositions_from_numbers
from torch_alchemical.tools.train.initialize import (
    get_average_number_of_atoms,
    get_composition_weights,
    get_energies_scale_factor,
    rescale_energies_and_forces,
)
import torch
from ruamel.yaml import YAML
import argparse
import lightning.pytorch as pl
import os
from datetime import datetime


torch.set_default_dtype(torch.float64)
SUPPORTED_ARCHITECTURES = ["alchemical_model", "soap-bpnn"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parameters", type=str)
    args = parser.parse_args()
    with open(args.parameters, "r") as f:
        yaml = YAML(typ="safe", pure=True)
        parameters = yaml.load(f)

    datamodule = LitDataModule(**parameters["datamodule"])
    datamodule.prepare_data()
    datamodule.setup()

    architecture = parameters.pop("architecture")
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Architecture {architecture} is not supported. Supported architectures are {SUPPORTED_ARCHITECTURES}"
        )
    if architecture == "soap-bpnn":
        model = BPPSModel(
            unique_numbers=datamodule.unique_numbers,
            **parameters["model"],
        )
    else:
        model = AlchemicalModel(
            unique_numbers=datamodule.unique_numbers,
            **parameters["model"],
        )

    # Calclating the normalization factors
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

    # Rescaling the energies and forces
    rescale_energies_and_forces(
        train_dataset, compositions, composition_weights, energies_scale_factor
    )

    # Setting the normalization factors for the model
    model.set_normalization_factor(average_number_of_atoms)
    model.set_energies_scale_factor(energies_scale_factor)
    model.set_basis_normalization_factor(average_number_of_neighbors)
    model.set_composition_weights(composition_weights)

    model = torch.jit.script(model)
    restart = parameters["litmodel"].pop("restart")
    if restart:
        litmodel = LitModel.load_from_checkpoint(
            restart, model=model, **parameters["litmodel"]
        )
    else:
        litmodel = LitModel(model=model, **parameters["litmodel"])

    early_stopping_callback = parameters["trainer"].pop("early_stopping_callback")
    checkpoint_callback = parameters["trainer"].pop("checkpoint_callback")
    callbacks = [
        pl.callbacks.EarlyStopping(**early_stopping_callback),
        pl.callbacks.ModelCheckpoint(**checkpoint_callback),
    ]
    logname = parameters["logging"].pop("name")
    logname += f"_{datetime.now().strftime('%d-%m-%Y--%H:%M:%S')}"
    logdir = parameters["logging"].pop("save_dir")
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logger = pl.loggers.WandbLogger(
        name=logname, save_dir=logdir, **parameters["logging"]
    )
    logger.experiment.config.update(parameters)

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        **parameters["trainer"],
    )
    trainer.fit(litmodel, datamodule)
