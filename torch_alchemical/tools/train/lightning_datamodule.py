from typing import List, Optional

import lightning.pytorch as pl
import torch
from ase.io import read
from torch_geometric.loader import DataLoader

from torch_alchemical.data import AtomisticDataset
from torch_alchemical.transforms import NeighborList
from torch_alchemical.utils import get_list_of_unique_atomic_numbers


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
        train_frames_path: str,
        val_frames_path: str,
        target_properties: List[str],
        neighborlist_cutoff_radius: float,
        test_frames_path: Optional[str] = None,
        batch_size: Optional[int] = 16,
        shuffle: Optional[bool] = True,
        verbose: Optional[bool] = False,
    ):
        super().__init__()
        self.train_frames_path = train_frames_path
        self.val_frames_path = val_frames_path
        self.test_frames_path = test_frames_path
        self.batch_size = batch_size
        self.neighborlist_cutoff_radius = neighborlist_cutoff_radius
        self.target_properties = target_properties
        self.verbose = verbose
        self.shuffle = shuffle

    def prepare_data(self):
        self.train_frames = read(self.train_frames_path, ":")
        self.val_frames = read(self.val_frames_path, ":")
        if self.test_frames_path is not None:
            self.test_frames = read(self.test_frames_path, ":")
        else:
            self.test_frames = []
        self.unique_numbers = get_list_of_unique_atomic_numbers(
            self.train_frames + self.val_frames + self.test_frames
        )

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "prepare"):
            transforms = [
                NeighborList(cutoff_radius=self.neighborlist_cutoff_radius),
            ]
            self.train_dataset = AtomisticDataset(
                self.train_frames,
                target_properties=self.target_properties,
                transforms=transforms,
                verbose=self.verbose,
            )
            self.val_dataset = AtomisticDataset(
                self.val_frames,
                target_properties=self.target_properties,
                transforms=transforms,
                verbose=self.verbose,
            )
            if self.test_frames_path is not None:
                self.test_dataset = AtomisticDataset(
                    self.test_frames,
                    target_properties=self.target_properties,
                    transforms=transforms,
                    verbose=self.verbose,
                )
            else:
                self.test_dataset = []  # type: ignore

    def train_dataloader(self):
        batch_size = self.batch_size
        dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=self.shuffle
        )
        return dataloader

    def val_dataloader(self):
        batch_size = self.batch_size
        dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        return dataloader

    def test_dataloader(self):
        batch_size = self.batch_size
        dataloader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=self.shuffle
        )
        return dataloader
