from typing import Union

import numpy as np
import torch
from ase import Atoms


def get_list_of_unique_elements(frames: list[Atoms]) -> np.ndarray[str]:
    species = []
    for frame in frames:
        species.extend(frame.get_chemical_symbols())
    return np.unique(species)


def get_list_of_unique_atomic_numbers(
    frames: list[Union[Atoms, torch.ScriptObject]]
) -> np.ndarray[int]:
    unique_numbers = []
    for frame in frames:
        if isinstance(frame, Atoms):
            unique_numbers.extend(frame.get_atomic_numbers())
        elif isinstance(frame, torch.ScriptObject):
            unique_numbers.extend(frame.species)
        else:
            raise TypeError(f"Unknown frame type: {type(frame)}")
    return np.unique(unique_numbers)
