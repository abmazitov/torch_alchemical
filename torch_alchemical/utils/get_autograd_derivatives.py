import torch


def get_autograd_forces(energies: torch.Tensor, positions: list[torch.Tensor]):
    gradients = torch.autograd.grad(
        energies,
        positions,
        grad_outputs=torch.ones_like(energies),
        create_graph=True,
        retain_graph=True,
    )
    forces = [-g for g in gradients]
    return forces
