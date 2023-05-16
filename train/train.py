import lightning.pytorch as pl
import torch
from ase.io import read
from torch_geometric.loader import DataLoader

from torch_alchemical.data import AtomisticDataset
from torch_alchemical.models import AlchemicalModel
from torch_alchemical.utils import (
    get_list_of_unique_atomic_numbers,
    get_species_coupling_matrix,
)
from torch_alchemical.transforms import CompositionFeatures, NeighborList
from torch.nn.functional import mse_loss, l1_loss

from typing import Optional

torch.set_default_dtype(torch.float64)

N_PSEUDO = 4
CUTOFF = 5.0
HIDDEN_SIZE = 80
RADIAL_BASIS_CUTOFF = 250.0
PS_BASIS_CUTOFF = 180.0
ENERGIES_WEIGHT = 1.0
FORCES_WEIGHT = 1.0
TRAIN_VAL_TEST = [0.8, 0.1, 0.1]


def train_test_split(dataset, lengths, shuffle=True):
    train_val_test = [length / sum(lengths) for length in lengths]
    if shuffle:
        return torch.utils.data.random_split(dataset, lengths)
    else:
        train_set_indices = range(0, int(train_val_test[0] * len(dataset)))
        train_set = torch.utils.data.Subset(dataset, train_set_indices)
        val_set_indices = range(
            int(train_val_test[0] * len(dataset)), int(sum(lengths[:2]) * len(dataset))
        )
        val_set = torch.utils.data.Subset(dataset, val_set_indices)
        test_set_indices = range(
            int(sum(lengths[:2]) * len(dataset)), int(sum(lengths) * len(dataset))
        )

        test_set = torch.utils.data.Subset(dataset, test_set_indices)
        return train_set, val_set, test_set


class LitModel(pl.LightningModule):
    def __init__(self, model, energies_weight: float, forces_weight: float):
        super().__init__()
        self.model = model
        self.energies_weight = energies_weight
        self.forces_weight = forces_weight
        self.automatic_optimization = False

    def initialize_composition_layer_weights(self, model, datamodule):
        assert hasattr(model, "composition_layer")
        dataset = datamodule.train_dataset
        composition_layer = model.composition_layer
        compositions = torch.cat([data.composition for data in dataset], dim=0)
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
        model.ps_features_layer.spex_calculator.combination_matrix.weight = (
            torch.nn.Parameter(
                get_species_coupling_matrix(
                    datamodule.unique_numbers, N_PSEUDO
                ).contiguous(),
                requires_grad=True,
            )
        )
        print("Combinining matrix is initialized manually")

    def on_train_epoch_start(self):
        self.epoch_loss = 0.0
        self.train_energies_mae = 0.0
        self.train_forces_mae = 0.0

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()

        def closure():
            optimizer.zero_grad()
            predicted_energies, predicted_forces = self.model(batch)
            target_energies = batch.energies
            target_forces = batch.forces
            energies_loss = self.energies_weight * mse_loss(
                predicted_energies.flatten(), target_energies.flatten(), reduction="sum"
            )
            forces_loss = self.forces_weight * mse_loss(
                predicted_forces.flatten(), target_forces.flatten(), reduction="sum"
            )
            loss = energies_loss + forces_loss
            self.manual_backward(loss)
            self.epoch_loss = loss.detach().item()
            self.train_energies_mae = l1_loss(
                predicted_energies.detach().flatten(),
                target_energies.detach().flatten(),
            )
            self.train_forces_mae = l1_loss(
                predicted_forces.flatten(), target_forces.flatten()
            )
            return loss

        optimizer.step(closure)

    # def training_step(self, batch, batch_idx):
    #     predicted_energies, predicted_forces = self.model(batch)
    #     target_energies = batch.energies
    #     target_forces = batch.forces
    #     energies_loss = self.energies_weight * mse_loss(
    #         predicted_energies.flatten(), target_energies, reduction="sum"
    #     )
    #     forces_loss = self.forces_weight * mse_loss(
    #         predicted_forces, target_forces, reduction="sum"
    #     )
    #     loss = energies_loss + forces_loss
    #     self.epoch_loss += loss.detach().item()
    #     predicted_energies = predicted_energies.detach()
    #     predicted_forces = predicted_forces.detach()
    #     self.train_energies_mae += l1_loss(
    #         predicted_energies.flatten(), target_energies
    #     )
    #     self.train_forces_mae += l1_loss(predicted_forces, target_forces)

    #     return loss

    def on_train_epoch_end(self):
        num_batches = len(self.trainer.datamodule.train_dataloader())
        epoch_loss = self.epoch_loss
        train_energies_mae = self.train_energies_mae / num_batches
        train_forces_mae = self.train_forces_mae / num_batches
        print(
            f"Loss: {epoch_loss:.4f}, Train Energies MAE: {train_energies_mae:.4f}, Train Forces MAE: {train_forces_mae:.4f}"
        )
        self.log("loss", epoch_loss)

    def on_validation_epoch_start(self):
        torch.set_grad_enabled(True)
        self.val_energies_mae = 0.0
        self.val_forces_mae = 0.0

    def validation_step(self, batch, batch_idx):
        predicted_energies, predicted_forces = self.model(batch, training=False)
        predicted_energies = predicted_energies.flatten().detach()
        predicted_forces = predicted_forces.detach()
        target_energies = batch.energies
        target_forces = batch.forces
        energies_mae = l1_loss(predicted_energies, target_energies)
        forces_mae = l1_loss(predicted_forces, target_forces)
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

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
    #     scheduler = torch.optim.lr_scheduler.ExponentialLR(
    #         optimizer, gamma=0.99, verbose=True
    #     )
    #     return [optimizer], [scheduler]

    def configure_optimizers(self):
        optimizer = torch.optim.LBFGS(
            self.parameters(), lr=0.05, history_size=128, line_search_fn="strong_wolfe"
        )
        return optimizer


class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        frames_path: str,
        batch_size: int,
        shuffle: bool = True,
        verbose: bool = False,
    ):
        super().__init__()
        self.frames_path = frames_path
        self.batch_size = batch_size
        self.verbose = verbose
        self.shuffle = shuffle

    def prepare_data(self):
        self.frames = read(self.frames_path, ":")[:1250]
        self.unique_numbers = get_list_of_unique_atomic_numbers(self.frames)

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "prepare"):
            transforms = [
                CompositionFeatures(self.unique_numbers),
                NeighborList(cutoff_radius=CUTOFF),
            ]
            dataset = AtomisticDataset(
                self.frames,
                target_properties=["energies", "forces"],
                transforms=transforms,
                verbose=self.verbose,
            )
            (
                self.train_dataset,
                self.val_dataset,
                self.test_dataset,
            ) = train_test_split(dataset, TRAIN_VAL_TEST, shuffle=self.shuffle)

    def train_dataloader(self):
        batch_size = self.batch_size
        if self.batch_size == "len":
            batch_size = len(self.train_dataset)
        dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=self.shuffle
        )
        return dataloader

    def val_dataloader(self):
        batch_size = self.batch_size
        if self.batch_size == "len":
            batch_size = len(self.val_dataset)
        dataloader = DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=self.shuffle
        )
        return dataloader

    def test_dataloader(self):
        batch_size = self.batch_size
        if self.batch_size == "len":
            batch_size = len(self.test_dataset)
        dataloader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=self.shuffle
        )
        return dataloader


if __name__ == "__main__":
    datamodule = LitDataModule(
        "../data/hea_samples_bulk.xyz", batch_size="len", shuffle=False, verbose=True
    )
    datamodule.prepare_data()
    datamodule.setup(stage="prepare")

    model = AlchemicalModel(
        hidden_sizes=HIDDEN_SIZE,
        output_size=1,
        unique_numbers=datamodule.unique_numbers,
        cutoff=CUTOFF,
        basis_cutoff_radial_spectrum=RADIAL_BASIS_CUTOFF,
        basis_cutoff_power_spectrum=PS_BASIS_CUTOFF,
        num_pseudo_species=N_PSEUDO,
    )

    litmodel = LitModel(
        model, energies_weight=ENERGIES_WEIGHT, forces_weight=FORCES_WEIGHT
    )

    callbacks = [
        pl.callbacks.EarlyStopping(
            monitor="loss",
            patience=10,
            mode="min",
        ),
        pl.callbacks.ModelCheckpoint(
            monitor="loss", filename="{epoch}-{loss:.4f}", save_top_k=1, mode="min"
        ),
    ]

    litmodel.initialize_composition_layer_weights(litmodel.model, datamodule)
    litmodel.initialize_combining_matrix(litmodel.model, datamodule)

    trainer = pl.Trainer(max_epochs=1000, callbacks=callbacks)
    trainer.fit(litmodel, datamodule)
