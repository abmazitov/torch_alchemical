from torch_alchemical.models import PowerSpectrumModel
from torch_alchemical.tools.train import LitDataModule, LitModel
from torch_alchemical.tools.train.initialize import (
    initialize_combining_matrix,
    initialize_composition_layer_weights,
)
import torch
import ruamel.yaml as yaml
import argparse
import lightning.pytorch as pl


torch.set_default_dtype(torch.float64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parameters", type=str)
    args = parser.parse_args()
    with open(args.parameters, "r") as f:
        parameters = yaml.safe_load(f)

    datamodule = LitDataModule(**parameters["datamodule"])
    datamodule.prepare_data()
    datamodule.setup()

    model = PowerSpectrumModel(
        unique_numbers=datamodule.unique_numbers, **parameters["model"]
    )

    restart = parameters["litmodel"].pop("restart")
    if restart:
        litmodel = LitModel.load_from_checkpoint(
            restart, model=model, **parameters["litmodel"]
        )
    else:
        litmodel = LitModel(model=model, **parameters["litmodel"])
        initialize_composition_layer_weights(litmodel.model, datamodule)
        initialize_combining_matrix(litmodel.model, datamodule)

    early_stopping_callback = parameters["trainer"].pop("early_stopping_callback")
    checkpoint_callback = parameters["trainer"].pop("checkpoint_callback")
    callbacks = [
        pl.callbacks.EarlyStopping(**early_stopping_callback),
        pl.callbacks.ModelCheckpoint(**checkpoint_callback),
    ]
    logger = pl.loggers.WandbLogger(**parameters["logging"])
    logger.experiment.config.update(parameters)

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        **parameters["trainer"],
    )
    trainer.fit(litmodel, datamodule)
