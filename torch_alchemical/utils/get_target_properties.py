import torch
from ase import Atoms


def get_target_properties(frame: Atoms, properties: list[str]):
    """Get target properties from Atoms object.

    Parameters
    ----------
    frame : ase.Atoms
        ASE Atoms object.
    properties : list[str]
        List of properties to extract from frames.

    Returns
    -------
    dict
        Dictionary of target properties.

    """
    target_properties = {}
    for prop in properties:
        if prop == "energies":
            target_properties[prop] = torch.tensor(
                frame.get_potential_energy(),
                dtype=torch.get_default_dtype(),
            )
        elif prop == "forces":
            target_properties[prop] = torch.tensor(
                frame.get_forces(),
                dtype=torch.get_default_dtype(),
            )
        elif prop == "stresses":
            target_properties[prop] = torch.tensor(
                frame.get_stress(),
                dtype=torch.get_default_dtype(),
            )
        else:
            raise ValueError(f"Unknown property {prop}")
    return target_properties
