import lightning.pytorch as pl
import torch
import os
from torch_alchemical.nn import WeightedMSELoss, MAELoss
from torch_alchemical.utils import get_autograd_forces
from torch_alchemical.tools.logging.wandb import log_wandb_data


class LitModel(pl.LightningModule):
    def __init__(
        self,
        model,
        energies_weight: float,
        forces_weight: float,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.model = model
        self.energies_weight = energies_weight
        self.forces_weight = forces_weight
        self.lr = lr
        self.weight_decay = weight_decay

    def on_train_epoch_start(self):
        self.train_energies_mae = 0.0
        self.train_forces_mae = 0.0

    def forward(self, batch, training=True):
        predicted_energies = self.model(
            positions=batch.pos,
            cells=batch.cell,
            numbers=batch.numbers,
            edge_indices=batch.edge_index,
            edge_shifts=batch.edge_shift,
            ptr=batch.ptr,
        )
        predicted_forces = get_autograd_forces(predicted_energies, batch.pos)[0]
        target_energies = batch.energies.view(-1, 1)
        target_forces = batch.forces
        return predicted_energies, predicted_forces, target_energies, target_forces

    def training_step(self, batch, batch_idx):
        (
            predicted_energies,
            predicted_forces,
            target_energies,
            target_forces,
        ) = self.forward(batch)

        loss_fn = WeightedMSELoss(
            energies_weight=self.energies_weight, forces_weight=self.forces_weight
        )
        loss = loss_fn(
            predicted_energies=predicted_energies,
            predicted_forces=predicted_forces,
            target_energies=target_energies,
            target_forces=target_forces,
        )
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.trainer.datamodule.batch_size,
        )

        loss_fn = MAELoss()
        train_energies_mae = loss_fn(
            predicted_energies=predicted_energies.detach(),
            target_energies=target_energies,
        ).item()
        train_forces_mae = loss_fn(
            predicted_forces=predicted_forces.detach(), target_forces=target_forces
        ).item()

        self.log(
            "train_energies_mae",
            train_energies_mae,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.trainer.datamodule.batch_size,
        )
        self.log(
            "train_forces_mae",
            train_forces_mae,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.trainer.datamodule.batch_size,
        )
        self.train_energies_mae += train_energies_mae
        self.train_forces_mae += train_forces_mae
        return loss

    def on_train_epoch_end(self):
        num_batches = len(self.trainer.datamodule.train_dataloader())
        train_energies_mae = self.train_energies_mae / num_batches
        train_forces_mae = self.train_forces_mae / num_batches
        print("\n")
        print(f"Energies MAE: Train {train_energies_mae:.3f}")
        print(f"Forces MAE: Train {train_forces_mae:.3f}")

    def on_validation_epoch_start(self):
        torch.set_grad_enabled(True)
        self.val_energies_mae = 0.0
        self.val_forces_mae = 0.0
        self.predicted_energies = []
        self.predicted_forces = []
        self.target_energies = []
        self.target_forces = []

    def validation_step(self, batch, batch_idx):
        (
            predicted_energies,
            predicted_forces,
            target_energies,
            target_forces,
        ) = self.forward(batch)
        loss_fn = MAELoss()
        val_energies_mae = loss_fn(
            predicted_energies=predicted_energies.detach(),
            target_energies=target_energies,
        ).item()
        val_forces_mae = loss_fn(
            predicted_forces=predicted_forces.detach(), target_forces=target_forces
        ).item()
        self.log(
            "val_energies_mae",
            val_energies_mae,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.trainer.datamodule.batch_size,
        )
        self.log(
            "val_forces_mae",
            val_forces_mae,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.trainer.datamodule.batch_size,
        )
        self.val_energies_mae += val_energies_mae
        self.val_forces_mae += val_forces_mae
        self.predicted_energies.append(predicted_energies.detach())
        self.predicted_forces.append(predicted_forces.detach())
        self.target_energies.append(target_energies.detach())
        self.target_forces.append(target_forces.detach())

    def on_validation_epoch_end(self):
        num_batches = len(self.trainer.datamodule.val_dataloader())
        val_energies_mae = self.val_energies_mae / num_batches
        val_forces_mae = self.val_forces_mae / num_batches
        print("\n")
        print(f"Energies MAE: Val {val_energies_mae:.3f}")
        print(f"Forces MAE: Val {val_forces_mae:.3f}")
        if isinstance(self.logger, pl.loggers.WandbLogger):
            log_wandb_data(
                self.predicted_energies,
                self.predicted_forces,
                self.target_energies,
                self.target_forces,
                val_energies_mae,
                val_forces_mae,
            )
            torch.save(
                self.predicted_energies,
                os.path.join(self.logger.experiment.dir, "val_predicted_energies.pt"),
            )
            torch.save(
                self.predicted_forces,
                os.path.join(self.logger.experiment.dir, "val_predicted_forces.pt"),
            )
            torch.save(
                self.target_energies,
                os.path.join(self.logger.experiment.dir, "val_target_energies.pt"),
            )
            torch.save(
                self.target_forces,
                os.path.join(self.logger.experiment.dir, "val_target_forces.pt"),
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
