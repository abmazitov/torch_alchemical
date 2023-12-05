import torch
from ase import Atoms


def get_property(frame: Atoms, property_name: str):
    if property_name == "energies":
        if hasattr(frame, "calc"):
            prop = frame.get_potential_energy()
        else:
            prop = frame.info["energy"]
    elif property_name == "forces":
        if hasattr(frame, "calc"):
            prop = frame.get_forces()
        else:
            prop = frame.info["forces"]
    elif property_name == "stresses":
        if hasattr(frame, "calc"):
            prop = frame.get_stress()
        else:
            prop = frame.info["stresses"]
    else:
        raise ValueError(f"Unknown property {property_name}")
    return torch.tensor(prop, dtype=torch.get_default_dtype())


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
        target_properties[prop] = get_property(frame, prop)
    return target_properties
