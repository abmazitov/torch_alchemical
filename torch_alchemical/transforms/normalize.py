import torch
from torch_alchemical.data import AtomisticDataset
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_alchemical.utils import get_compositions_from_numbers, get_target_properties
from typing import Union
from ase import Atoms


class TargetPropertiesNormalizer(BaseTransform):
    def __init__(
        self,
        unique_numbers: list[int],
        train_frames: list[Atoms] = None,
        target_properties: list[str] = ["energies"],
    ):
        self.unique_numbers = unique_numbers
        self.target_properties = target_properties
        self.normalization_params = None
        if train_frames is not None:
            self.fit(train_frames)

    def __call__(self, data: Data) -> Data:
        self._normalize_item(data)
        return data

    def fit(self, train_frames: list[Atoms]):
        self.normalization_params = dict.fromkeys(self.target_properties)
        target_properties_data = [
            get_target_properties(atoms, self.target_properties)
            for atoms in train_frames
        ]
        target_properties_data = {
            key: torch.cat([item[key].view(-1) for item in target_properties_data])
            for key in self.target_properties
        }
        for target_property in self.target_properties:
            normalization_params = {}
            if target_property == "energies":
                numbers = torch.cat(
                    [torch.tensor(atoms.numbers) for atoms in train_frames]
                )
                ptr = torch.cumsum(
                    torch.tensor([0] + [len(atoms) for atoms in train_frames]), dim=0
                )
                compositions = torch.stack(
                    get_compositions_from_numbers(numbers, self.unique_numbers, ptr)
                )
                energies = target_properties_data[target_property]
                atomic_energies = torch.linalg.lstsq(compositions, energies).solution
                normalization_params.update(
                    {
                        "atomic_energies": atomic_energies,
                    }
                )
                property_values = energies - compositions @ atomic_energies
            else:
                property_values = target_properties_data[target_property]

            mean = property_values.mean()
            std = property_values.std()
            normalization_params.update(
                {
                    "mean": mean,
                    "std": std,
                }
            )
            self.normalization_params[target_property] = normalization_params

    def _normalize_item(self, data: Data):
        assert self.normalization_params is not None
        for target_property in self.target_properties:
            if target_property == "energies":
                atomic_energies = self.normalization_params[target_property][
                    "atomic_energies"
                ]
                if hasattr(data, "batch_size"):
                    numbers = data.numbers
                    ptr = data.ptr
                    compositions = torch.stack(
                        get_compositions_from_numbers(numbers, self.unique_numbers, ptr)
                    )
                    atomic_energy_shift = (compositions @ atomic_energies).squeeze()
                else:
                    numbers = [data.numbers]
                    compositions = torch.stack(
                        get_compositions_from_numbers(numbers, self.unique_numbers)
                    )
                    atomic_energy_shift = (compositions @ atomic_energies).item()
                setattr(
                    data,
                    target_property,
                    getattr(data, target_property) - atomic_energy_shift,
                )
            mean = self.normalization_params[target_property]["mean"]
            std = self.normalization_params[target_property]["std"]
            setattr(
                data, target_property, (getattr(data, target_property) - mean) / std
            )

    def normalize(self, data: Union[Data, list[Data], AtomisticDataset]):
        if isinstance(data, Data):
            self._normalize_item(data)
        else:
            for item in data:
                self._normalize_item(item)

    def _denormalize_item(self, data: Data):
        assert self.normalization_params is not None
        for target_property in self.target_properties:
            mean = self.normalization_params[target_property]["mean"]
            std = self.normalization_params[target_property]["std"]
            setattr(data, target_property, getattr(data, target_property) * std + mean)
            if target_property == "energies":
                atomic_energies = self.normalization_params[target_property][
                    "atomic_energies"
                ]
                if hasattr(data, "batch_size"):
                    numbers = data.numbers
                    ptr = data.ptr
                    compositions = torch.stack(
                        get_compositions_from_numbers(numbers, self.unique_numbers, ptr)
                    )
                    atomic_energy_shift = (compositions @ atomic_energies).squeeze()
                else:
                    numbers = [data.numbers]
                    compositions = torch.stack(
                        get_compositions_from_numbers(numbers, self.unique_numbers)
                    )
                    atomic_energy_shift = (compositions @ atomic_energies).item()
                setattr(
                    data,
                    target_property,
                    getattr(data, target_property) + atomic_energy_shift,
                )

    def denormalize(self, data: Union[Data, list[Data], AtomisticDataset]):
        if isinstance(data, Data):
            self._denormalize_item(data)
        else:
            for item in data:
                self._denormalize_item(item)
