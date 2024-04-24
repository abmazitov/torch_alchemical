import os
from typing import Optional

import lightning.pytorch as pl
import torch

from torch_alchemical.nn import MAELoss, MSELoss, WeightedSSELoss
from torch_alchemical.tools.logging.wandb import log_wandb_data
from torch_alchemical.utils import get_autograd_forces


class LitModel(pl.LightningModule):
    def __init__(
        self,
        model,
        energies_weight: float,
        forces_weight: float,
        lr: Optional[float] = 1e-4,
        weight_decay: Optional[float] = 1e-5,
        lambda_lr: Optional[float] = 1.0,
        log_wandb_tables: Optional[bool] = True,
    ):
        super().__init__()
        self.model = model
        self.energies_weight = energies_weight
        self.forces_weight = forces_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_lr = lambda_lr
        self.log_wandb_tables = log_wandb_tables

    def on_train_epoch_start(self):
        self.train_energies_mae = 0.0
        self.train_forces_mae = 0.0
        self.train_energies_rmse = 0.0

    def forward(self, batch):
        predicted_energies = self.model(
            positions=batch.pos,
            cells=batch.cell,
            numbers=batch.numbers,
            edge_indices=batch.edge_index,
            edge_offsets=batch.edge_offsets,
            batch=batch.batch,
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

        loss_fn = WeightedSSELoss(
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
            sync_dist=True,
        )

        predicted_energies = predicted_energies.detach()
        predicted_forces = predicted_forces.detach()
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
            sync_dist=True,
        )
        self.log(
            "train_forces_mae",
            train_forces_mae,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.trainer.datamodule.batch_size,
            sync_dist=True,
        )

        loss_fn = MSELoss()
        train_energies_rmse = torch.sqrt(
            loss_fn(
                predicted_energies=predicted_energies.detach(),
                target_energies=target_energies,
            )
        ).item()

        self.log(
            "train_energies_rmse",
            train_energies_rmse,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.trainer.datamodule.batch_size,
            sync_dist=True,
        )

        self.train_energies_mae += train_energies_mae
        self.train_forces_mae += train_forces_mae
        self.train_energies_rmse += train_energies_rmse
        return loss

    def on_train_epoch_end(self):
        num_batches = (
            len(self.trainer.datamodule.train_dataloader()) / self.trainer.num_devices
        )
        train_energies_mae = self.train_energies_mae / num_batches
        train_forces_mae = self.train_forces_mae / num_batches
        train_energies_rmse = self.train_energies_rmse / num_batches
        print("\n")
        print(f"Energies MAE: Train {train_energies_mae:.3f}")
        print(f"Forces MAE: Train {train_forces_mae:.3f}")
        print(f"Energies RMSE: Train {train_energies_rmse:.3f}")

    def on_validation_epoch_start(self):
        torch.set_grad_enabled(True)
        self.val_energies_mae = 0.0
        self.val_forces_mae = 0.0
        self.val_energies_rmse = 0.0
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
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )
        self.log(
            "val_forces_mae",
            val_forces_mae,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )

        loss_fn = MSELoss()
        val_energies_rmse = torch.sqrt(
            loss_fn(
                predicted_energies=predicted_energies.detach(),
                target_energies=target_energies,
            )
        ).item()
        self.log(
            "val_energies_rmse",
            val_energies_rmse,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )

        self.val_energies_mae += val_energies_mae
        self.val_forces_mae += val_forces_mae
        self.val_energies_rmse += val_energies_rmse
        self.predicted_energies.append(predicted_energies.cpu().detach())
        self.predicted_forces.append(predicted_forces.cpu().detach())
        self.target_energies.append(target_energies.cpu().detach())
        self.target_forces.append(target_forces.cpu().detach())

    def on_validation_epoch_end(self):
        num_batches = (
            len(self.trainer.datamodule.val_dataloader()) / self.trainer.num_devices
        )
        val_energies_mae = self.val_energies_mae / num_batches
        val_forces_mae = self.val_forces_mae / num_batches
        val_energies_rmse = self.val_energies_rmse / num_batches
        print("\n")
        print(f"Energies MAE: Val {val_energies_mae:.3f}")
        print(f"Forces MAE: Val {val_forces_mae:.3f}")
        print(f"Energies RMSE: Val {val_energies_rmse:.3f}")
        if isinstance(self.logger, pl.loggers.WandbLogger):
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
            if self.log_wandb_tables:
                log_wandb_data(
                    self.predicted_energies,
                    self.predicted_forces,
                    self.target_energies,
                    self.target_forces,
                    val_energies_mae,
                    val_forces_mae,
                )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-2, total_steps=self.trainer.estimated_stepping_batches
        )
        return [optimizer], [scheduler]
