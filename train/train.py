from torch_alchemical.models import BPPSLodeModel, LitModel
from torch_alchemical.data import LitDataModule
from torch_alchemical.utils import load_parameters

import torch
import lightning.pytorch as pl
import os
from datetime import datetime

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

def main():
    parameters = load_parameters()

    datamodule = LitDataModule(**parameters["datamodule"])
    datamodule.prepare_data()
    datamodule.setup()

    composition_weights = datamodule.prepare_compositions_weights()

    model = BPPSLodeModel(
        unique_numbers=datamodule.unique_numbers,
        **parameters["model"],
    )

    model.set_compositions_weights(composition_weights)

    print(model)

    restart = parameters["litmodel"].pop("restart")
    if restart:
        litmodel = LitModel.load_from_checkpoint(
            restart, model=model, **parameters["litmodel"]
        )
    else:
        litmodel = LitModel(model=model, **parameters["litmodel"])

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

if __name__ == "__main__":
    main()