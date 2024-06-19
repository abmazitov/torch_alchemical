from torch_alchemical.models import AlchemicalModel, BPPSModel, BPPSLodeModel
from torch_alchemical.tools.train import LitDataModule, LitModel
from torch_alchemical.utils import get_compositions_from_numbers
from torch_alchemical.tools.train.initialize import (
    get_composition_weights,
)
import torch
from ruamel.yaml import YAML
import argparse
import lightning.pytorch as pl
import os
from datetime import datetime

torch.manual_seed(0)
torch.set_float32_matmul_precision("high")

SUPPORTED_ARCHITECTURES = ["alchemical_model", "soap-bpnn", "soap-bpnn-lode"]


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
    elif architecture == "alchemical_model":
        model = AlchemicalModel(
            unique_numbers=datamodule.unique_numbers,
            **parameters["model"],
        )
    else:
        model = BPPSLodeModel(
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
    model.set_composition_weights(composition_weights)
    #model = torch.compile(model)
    print(model)
    restart = parameters["litmodel"].pop("restart")
    if restart:
        litmodel = LitModel.load_from_checkpoint(
            restart, model=model, **parameters["litmodel"]
        )
    else:
        litmodel = LitModel(model=model, **parameters["litmodel"])
    #litmodel = torch.compile(litmodel, fullgraph=True)
    checkpoint_callback = parameters["trainer"].pop("checkpoint_callback")
    callbacks = [
        pl.callbacks.ModelCheckpoint(**checkpoint_callback),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
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
