from typing import List, Optional

import lightning.pytorch as pl
import torch
from ase.io import read
from torch_geometric.loader import DataLoader

from torch_alchemical.data import AtomisticDataset
from torch_alchemical.data.preprocess import (
    NeighborList, 
    get_list_of_unique_atomic_numbers,
    get_compositions_from_numbers,
    )

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

    def setup(self, stage=None):
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

    def prepare_compositions_weights(self):
        train_dataset = self.train_dataset
        unique_numbers = self.unique_numbers
        numbers = torch.cat([data.numbers for data in train_dataset])
        batch = torch.cat(
            [
                torch.repeat_interleave(torch.tensor([i]), data.num_nodes)
                for i, data in enumerate(train_dataset)
            ]
        )
        compositions = torch.stack(
            get_compositions_from_numbers(numbers, unique_numbers, batch)
        ).to(torch.get_default_dtype())

        energies = torch.cat([data.energies.view(1, -1) for data in train_dataset], dim=0)
        weights = torch.linalg.lstsq(compositions, energies).solution
        composition_weights = weights.T
        return composition_weights
