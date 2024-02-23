import numpy as np
from ase import Atoms


def get_list_of_unique_elements(frames: list[Atoms]) -> np.ndarray:
    species = []
    for frame in frames:
        species.extend(frame.get_chemical_symbols())
    return np.unique(species)


def get_list_of_unique_atomic_numbers(frames: list[Atoms]) -> np.ndarray:
    unique_numbers = []
    for frame in frames:
        unique_numbers.extend(frame.get_atomic_numbers())
    return np.unique(unique_numbers).tolist()
