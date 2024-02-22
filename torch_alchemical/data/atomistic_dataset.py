from typing import Optional

import torch
from ase import Atoms
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm

from torch_alchemical.utils import get_target_properties

AVAILABLE_TARGET_PROPERTIES = ["energies", "forces"]


class AtomisticDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        frames: list[Atoms],
        target_properties: list[str],
        transforms: Optional[list[BaseTransform]] = None,
        verbose: Optional[bool] = False,
    ):
        super().__init__()
        assert set(target_properties).issubset(AVAILABLE_TARGET_PROPERTIES)
        self.dataset = []
        self.target_properties = target_properties
        self.frames = frames
        self.transforms = transforms
        self.verbose = verbose
        self.process()

    def process(self):
        positions_requires_grad = True if "forces" in self.target_properties else False
        cell_requires_grad = True if "stresses" in self.target_properties else False
        if self.verbose:
            frames = tqdm(self.frames, total=len(self.frames), desc="Processing data")
        else:
            frames = self.frames
        for frame in frames:
            target_properties = get_target_properties(frame, self.target_properties)
            positions = torch.tensor(
                frame.positions,
                requires_grad=positions_requires_grad,
                dtype=torch.get_default_dtype(),
            )
            cell = torch.tensor(
                frame.cell.array,
                requires_grad=cell_requires_grad,
                dtype=torch.get_default_dtype(),
            )
            numbers = torch.tensor(frame.numbers, dtype=torch.long)
            pbc = torch.tensor(frame.pbc, dtype=torch.bool)

            data = Data(
                numbers=numbers,
                pos=positions,
                cell=cell,
                pbc=pbc,
                **target_properties,
            )
            if self.transforms is not None:
                for transform in self.transforms:
                    data = transform(data)
            self.dataset.append(data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __setitem__(self, idx, item):
        self.dataset[idx] = item

    def __str__(self):
        return self.__class__.__name__ + f"({len(self)})"

    def __repr__(self):
        return str(self)
