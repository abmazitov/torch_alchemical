import lightning.pytorch as pl
import torch
from torch_alchemical.nn import WeightedMSELoss, MAELoss
from torch_alchemical.utils import (
    get_species_coupling_matrix,
    get_compositions_from_numbers,
    get_autograd_forces,
)


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

    def initialize_composition_layer_weights(self, model, datamodule):
        assert hasattr(model, "composition_layer")
        dataset = datamodule.train_dataset
        composition_layer = model.composition_layer
        numbers = torch.cat([data.numbers for data in dataset])
        ptr = torch.cumsum(
            torch.tensor([0] + [data.num_nodes for data in dataset]), dim=0
        )
        compositions = torch.stack(
            get_compositions_from_numbers(numbers, datamodule.unique_numbers, ptr)
        )
        compositions = torch.cat(
            (torch.ones(len(dataset)).view(-1, 1), compositions), dim=1
        )  # bias
        energies = torch.cat([data.energies.view(1, -1) for data in dataset], dim=0)
        weights = torch.linalg.lstsq(compositions, energies).solution
        composition_layer.weight = torch.nn.Parameter(
            weights[1:].T.contiguous(), requires_grad=True
        )
        composition_layer.bias = torch.nn.Parameter(
            weights[0].contiguous(), requires_grad=True
        )
        print("Composition layer weights are initialized with least squares solution")

    def initialize_combining_matrix(self, model, datamodule):
        assert hasattr(model, "ps_features_layer")
        radial_basis_calculator = (
            model.ps_features_layer.spex_calculator.vector_expansion_calculator.radial_basis_calculator
        )
        n_pseudo_species = radial_basis_calculator.n_pseudo_species
        model.ps_features_layer.spex_calculator.vector_expansion_calculator.radial_basis_calculator.combination_matrix.weight = torch.nn.Parameter(
            get_species_coupling_matrix(
                datamodule.unique_numbers, n_pseudo_species
            ).contiguous(),
            requires_grad=True,
        )
        print("Combinining matrix is initialized manually")

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
        print(predicted_energies.shape, target_energies.shape)
        print(predicted_forces.shape, target_forces.shape)
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

    def on_validation_epoch_end(self):
        num_batches = len(self.trainer.datamodule.val_dataloader())
        val_energies_mae = self.val_energies_mae / num_batches
        val_forces_mae = self.val_forces_mae / num_batches
        print("\n")
        print(f"Energies MAE: Val {val_energies_mae:.3f}")
        print(f"Forces MAE: Val {val_forces_mae:.3f}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
