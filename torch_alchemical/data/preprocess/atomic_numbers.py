from typing import List

import numpy as np
from ase import Atoms


def get_list_of_unique_atomic_numbers(frames: List[Atoms]) -> np.ndarray:
    unique_numbers = []
    for frame in frames:
        unique_numbers.extend(frame.get_atomic_numbers())
    return np.unique(unique_numbers).tolist()
