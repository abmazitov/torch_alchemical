import lightning.pytorch as pl
from torch_alchemical.utils import get_list_of_unique_atomic_numbers
from torch_alchemical.data import AtomisticDataset
from torch_alchemical.transforms import NeighborList
from ase.io import read
from typing import Optional
import torch
from torch_geometric.loader import DataLoader


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


class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        frames_path: str,
        train_val_test: list[float],
        batch_size: int,
        neighborlist_cutoff_radius: float,
        target_properties: list[str],
        shuffle: bool = True,
        verbose: bool = False,
    ):
        super().__init__()
        self.frames_path = frames_path
        self.train_val_test = train_val_test
        self.batch_size = batch_size
        self.neighborlist_cutoff_radius = neighborlist_cutoff_radius
        self.target_properties = target_properties
        self.verbose = verbose
        self.shuffle = shuffle

    def prepare_data(self):
        self.frames = read(self.frames_path, ":")
        self.unique_numbers = get_list_of_unique_atomic_numbers(self.frames)

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "prepare"):
            transforms = [
                NeighborList(cutoff_radius=self.neighborlist_cutoff_radius),
            ]
            dataset = AtomisticDataset(
                self.frames,
                target_properties=self.target_properties,
                transforms=transforms,
                verbose=self.verbose,
            )
            (
                self.train_dataset,
                self.val_dataset,
                self.test_dataset,
            ) = train_test_split(dataset, self.train_val_test, shuffle=self.shuffle)

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
        dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        return dataloader

    def test_dataloader(self):
        batch_size = self.batch_size
        if self.batch_size == "len":
            batch_size = len(self.test_dataset)
        dataloader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=self.shuffle
        )
        return dataloader
