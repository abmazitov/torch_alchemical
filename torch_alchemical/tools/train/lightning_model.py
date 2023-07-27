import lightning.pytorch as pl
import torch
from torch_alchemical.nn import WeightedMSELoss, MAELoss
from torch_alchemical.utils import (
    get_species_coupling_matrix,
    get_compositions_from_numbers,
    extract_batch_data,
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
        self.automatic_optimization = False

    def initialize_composition_layer_weights(self, model, datamodule):
        assert hasattr(model, "composition_layer")
        dataset = datamodule.train_dataset
        composition_layer = model.composition_layer
        numbers = [data.numbers for data in dataset]
        compositions = torch.stack(
            get_compositions_from_numbers(numbers, datamodule.unique_numbers)
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
        self.epoch_loss = 0.0
        self.train_energies_mae = 0.0
        self.train_forces_mae = 0.0

    def forward(self, batch, training=True):
        positions, cells, numbers, edge_indices, edge_shifts = extract_batch_data(batch)
        predicted_energies = self.model(
            positions, cells, numbers, edge_indices, edge_shifts
        )
        predicted_forces = get_autograd_forces(predicted_energies, positions)
        target_energies = torch.cat(
            [data.energies.view(1, -1) for data in batch], dim=0
        )
        target_forces = torch.cat([data.forces for data in batch], dim=0)
        return predicted_energies, predicted_forces, target_energies, target_forces

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        optimizer.zero_grad()
        (
            predicted_energies,
            predicted_forces,
            target_energies,
            target_forces,
        ) = self.forward(batch)

        loss_fn = WeightedMSELoss(weights=[self.energies_weight, self.forces_weight])
        loss = loss_fn(
            predicted=[predicted_energies, predicted_forces],
            target=[target_energies, target_forces],
        )
        self.manual_backward(loss)
        optimizer.step()
        self.epoch_loss += loss.detach().item()
        predicted_energies = predicted_energies.detach()
        predicted_forces = predicted_forces.detach()
        loss_fn = MAELoss()
        self.train_energies_mae += loss_fn(predicted_energies, target_energies)
        self.train_forces_mae += loss_fn(predicted_forces, target_forces)

    def on_train_epoch_end(self):
        num_batches = len(self.trainer.datamodule.train_dataloader())
        epoch_loss = self.epoch_loss
        train_energies_mae = self.train_energies_mae / num_batches
        train_forces_mae = self.train_forces_mae / num_batches
        print(
            f"Train Energies MAE: {train_energies_mae:.4f}, Train Forces MAE: {train_forces_mae:.4f}"
        )
        self.log("loss", epoch_loss)

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
        energies_mae = loss_fn(predicted_energies, target_energies)
        forces_mae = loss_fn(predicted_forces, target_forces)
        self.val_energies_mae += energies_mae.item()
        self.val_forces_mae += forces_mae.item()

    def on_validation_epoch_end(self):
        num_batches = len(self.trainer.datamodule.val_dataloader())
        val_energies_mae = self.val_energies_mae / num_batches
        val_forces_mae = self.val_forces_mae / num_batches
        self.log("val_energies_mae", val_energies_mae)
        self.log("val_forces_mae", val_forces_mae)
        print(
            f"Val Energies MAE: {val_energies_mae:.4f}, Val Forces MAE: {val_forces_mae:.4f}"
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
