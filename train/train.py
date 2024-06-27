import os
from datetime import datetime
import argparse
from ruamel.yaml import YAML

import torch
import lightning.pytorch as pl

from torch_alchemical.models import BPPSLodeModel, LitModel
from torch_alchemical.data import LitDataModule

DATE_FORMAT = '%d-%m-%Y--%H:%M:%S'

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

def load_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("parameters", type=str)
    args = parser.parse_args()
    with open(args.parameters, "r") as f:
        yaml = YAML(typ="safe", pure=True)
        parameters = yaml.load(f)
    return parameters

def prepare_datamodule(parameters: dict) -> LitDataModule:
    datamodule = LitDataModule(**parameters["datamodule"])
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule

def prepare_model(datamodule: LitDataModule, parameters: dict) -> BPPSLodeModel:
    composition_weights = datamodule.prepare_compositions_weights()
    model = BPPSLodeModel(unique_numbers=datamodule.unique_numbers, **parameters["model"])
    model.set_compositions_weights(composition_weights)
    print(model)
    return model

def prepare_litmodel(model: BPPSLodeModel, parameters: dict) -> LitModel:
    restart = parameters["litmodel"].pop("restart")
    if restart:
        litmodel = LitModel.load_from_checkpoint(restart, model=model, **parameters["litmodel"])
    else:
        litmodel = LitModel(model=model, **parameters["litmodel"])
    return litmodel

def prepare_logger(parameters: dict) -> pl.loggers.WandbLogger:
    logname = parameters["logging"].pop("name") + f"_{datetime.now().strftime(DATE_FORMAT)}"
    logdir = parameters["logging"].pop("save_dir")
    os.makedirs(logdir, exist_ok=True)
    logger = pl.loggers.WandbLogger(name=logname, save_dir=logdir, **parameters["logging"])
    logger.experiment.config.update(parameters)
    return logger

def prepare_trainer(logger: pl.loggers.WandbLogger, parameters: dict) -> pl.Trainer:
    checkpoint_callback = parameters["trainer"].pop("checkpoint_callback")
    callbacks = [
        pl.callbacks.ModelCheckpoint(**checkpoint_callback),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **parameters["trainer"])
    return trainer

def main():
    parameters = load_parameters()
    datamodule = prepare_datamodule(parameters)
    model = prepare_model(datamodule, parameters)
    litmodel = prepare_litmodel(model, parameters)
    logger = prepare_logger(parameters)
    trainer = prepare_trainer(logger, parameters)

    trainer.fit(litmodel, datamodule)

if __name__ == "__main__":
    main()