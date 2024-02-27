from torch_alchemical.models import AlchemicalModel, BPPSModel
from torch_alchemical.tools.train import LitDataModule, LitModel
from torch_alchemical.tools.train.initialize import (
    initialize_composition_layer_weights,
    initialize_energies_forces_scale_factor,
    rescale_energies_and_forces,
)
import torch
from ruamel.yaml import YAML
import argparse
import lightning.pytorch as pl
import os
import numpy as np
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

    basis_normalization_factor = np.mean(
        [
            data.edge_index.shape[1] / data.pos.shape[0]
            for data in datamodule.train_dataset
        ]
    )
    architecture = parameters.pop("architecture")
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Architecture {architecture} is not supported. Supported architectures are {SUPPORTED_ARCHITECTURES}"
        )
    if architecture == "soap-bpnn":
        model = BPPSModel(
            unique_numbers=datamodule.unique_numbers,
            basis_normalization_factor=basis_normalization_factor,
            **parameters["model"],
        )
    else:
        model = AlchemicalModel(
            unique_numbers=datamodule.unique_numbers,
            basis_normalization_factor=basis_normalization_factor,
            **parameters["model"],
        )

    initialize_composition_layer_weights(model, datamodule, trainable=False)
    initialize_energies_forces_scale_factor(model, datamodule, trainable=False)
    rescale_energies_and_forces(model, datamodule)
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
