import os
from typing import Optional

import lightning.pytorch as pl
import torch

from torch.nn import MSELoss
from torch_alchemical.tools.logging.wandb import log_wandb_data
from torch_alchemical.utils import get_autograd_forces
from torch_alchemical.utils import get_compositions_from_numbers


class LitModel(pl.LightningModule):
    def __init__(
        self,
        model,
        predict_forces: bool,
        energies_weight: Optional[float] = 1.0,
        forces_weight: Optional[float] = 1.0,
        scheduler: Optional[bool] = False,
        lr: Optional[float] = 1e-4,
        weight_decay: Optional[float] = 1e-5,
        log_wandb_tables: Optional[bool] = True,
    ):
        super().__init__()
        self.model = model
        self.energies_weight = energies_weight
        if predict_forces:
            self.forces_weight = forces_weight
        else:
            self.forces_weight = None
        self.lr = lr
        self.weight_decay = weight_decay
        self.log_wandb_tables = log_wandb_tables
        self.predict_forces = predict_forces
        self.scheduler = scheduler

    def on_train_epoch_start(self):
        self.predicted_energies = []
        self.target_energies = []
        if self.predict_forces:
            self.predicted_forces = []
            self.target_forces = []

    def forward(self, batch):
        self.compositions = torch.stack(
            get_compositions_from_numbers(
                batch.numbers,
                self.model.unique_numbers,
                batch.batch,
                self.model.composition_weights.dtype,
            )
        )
        predicted_energies = self.model(
            positions=batch.pos,
            cells=batch.cell,
            numbers=batch.numbers,
            edge_indices=batch.edge_index,
            edge_offsets=batch.edge_offsets,
            batch=batch.batch,
        )
        
        target_energies = batch.energies.view(-1, 1) - self.compositions @ self.model.composition_weights.T

        if self.predict_forces:
            predicted_forces = get_autograd_forces(predicted_energies, batch.pos)[0]
            target_forces = batch.forces
        else:
            predicted_forces = None
            target_forces = None

        return predicted_energies, predicted_forces, target_energies, target_forces

    def training_step(self, batch, batch_idx):
        (
            predicted_energies,
            predicted_forces,
            target_energies,
            target_forces,
        ) = self.forward(batch)

        loss_fn = MSELoss()

        loss_energy = loss_fn(
            predicted_energies, target_energies
        )

        if self.predict_forces:
            loss_force = loss_fn(
                predicted_forces, target_forces
            )
            loss = self.energies_weight * loss_energy + self.forces_weight * loss_force
        else:
            loss = loss_energy

        self.log(
            "train_loss",
            loss.item(),
            on_step=True,
            prog_bar=True,
            batch_size=self.trainer.datamodule.batch_size,
            sync_dist=True,
        )

        predicted_energies = predicted_energies.cpu().detach()
        self.predicted_energies.append(predicted_energies)
        target_energies = target_energies.cpu().detach()
        self.target_energies.append(target_energies)
        if self.predict_forces:
            predicted_forces = predicted_forces.cpu().detach()
            self.predicted_forces.append(predicted_forces)
            target_forces = target_forces.cpu().detach()
            self.target_forces.append(target_forces)
        return loss

    def on_train_epoch_end(self):
        preds_energy = torch.cat(self.predicted_energies)
        targets_energy = torch.cat(self.target_energies)
        if self.predict_forces:
            preds_forces = torch.cat(self.predicted_forces)
            targets_forces = torch.cat(self.target_forces)
        loss_fn = MSELoss()
        train_energies_rmse = torch.sqrt(
            loss_fn(
                preds_energy, targets_energy
            )
        ).item()
        print("\n")
        print(f"Energies RMSE: Train {train_energies_rmse:.4f}")
        if self.predict_forces:
            train_forces_rmse = torch.sqrt(
                loss_fn(
                    preds_forces, targets_forces
                )
            ).item()
            print(f"Forces RMSE: Train {train_forces_rmse:.4f}")
        
        self.log(
            "train_energies_rmse",
            train_energies_rmse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )
        if self.predict_forces:
            self.log(
                "train_forces_rmse",
                train_forces_rmse,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=self.trainer.datamodule.batch_size,
            )

    def on_validation_epoch_start(self):
        torch.set_grad_enabled(True)
        self.predicted_energies = []
        self.target_energies = []
        if self.predict_forces:
            self.predicted_forces = []
            self.target_forces = []

    def validation_step(self, batch, batch_idx):
        (
            predicted_energies,
            predicted_forces,
            target_energies,
            target_forces,
        ) = self.forward(batch)
        loss_fn = MSELoss()
        val_energies_rmse = torch.sqrt(
            loss_fn(
                predicted_energies, target_energies
            )
        ).item()
        if self.predict_forces:

            val_forces_rmse = torch.sqrt(
                loss_fn(
                    predicted_forces, target_forces
                )
            ).item()

            self.log(
                "val_forces_rmse",
                val_forces_rmse,
                on_step=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=self.trainer.datamodule.batch_size,
            )
        self.log(
            "val_energies_rmse",
            val_energies_rmse,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )

        self.predicted_energies.append(predicted_energies.cpu().detach())
        self.target_energies.append(target_energies.cpu().detach())
        if self.predict_forces:
            self.predicted_forces.append(predicted_forces.cpu().detach())
            self.target_forces.append(target_forces.cpu().detach())

    def on_validation_epoch_end(self):
        preds_energy = torch.cat(self.predicted_energies)
        targets_energy = torch.cat(self.target_energies)
        if self.predict_forces:
            preds_forces = torch.cat(self.predicted_forces)
            targets_forces = torch.cat(self.target_forces)
        loss_fn = MSELoss()
        val_energies_rmse = torch.sqrt(
            loss_fn(
                preds_energy, targets_energy
            )
        ).item()
        print("\n")
        print(f"Energies RMSE: Val {val_energies_rmse:.4f}")
        if self.predict_forces:
            val_forces_rmse = torch.sqrt(
                loss_fn(
                    preds_forces, targets_forces
                )
            ).item()
            print(f"Forces RMSE: Val {val_forces_rmse:.4f}")

        if isinstance(self.logger, pl.loggers.WandbLogger):
            torch.save(
                self.predicted_energies,
                os.path.join(self.logger.experiment.dir, "val_predicted_energies.pt"),
            )
            torch.save(
                self.target_energies,
                os.path.join(self.logger.experiment.dir, "val_target_energies.pt"),
            )
            if self.predict_forces:
                torch.save(
                    self.predicted_forces,
                    os.path.join(self.logger.experiment.dir, "val_predicted_forces.pt"),
                )
                torch.save(
                    self.target_forces,
                    os.path.join(self.logger.experiment.dir, "val_target_forces.pt"),
                )

            if self.log_wandb_tables:
                if self.predict_forces:
                    log_wandb_data(
                        self.predicted_energies,
                        self.predicted_forces,
                        self.target_energies,
                        self.target_forces,
                        val_energies_rmse,
                        val_forces_rmse,
                    )
        self.log(
            "val_energies_rmse",
            val_energies_rmse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.trainer.datamodule.batch_size,
        )
        if self.predict_forces:
            self.log(
                "val_forces_rmse",
                val_forces_rmse,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=self.trainer.datamodule.batch_size,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.scheduler:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=1e-2, total_steps=self.trainer.estimated_stepping_batches
            )
        else: # Dummy scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=99999999, gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
