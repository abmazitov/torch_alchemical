import wandb
import torch


def get_wandb_datatables(
    predicted_energies, predicted_forces, target_energies, target_forces
):
    target_energies = target_energies.detach().flatten().cpu().numpy()
    predicted_energies = predicted_energies.detach().flatten().cpu().numpy()
    data_energies = [[x, y] for x, y in zip(target_energies, predicted_energies)]
    table_energies = wandb.Table(
        data=data_energies, columns=["Target energies", "Predicted energies"]
    )

    target_forces = target_forces.detach().flatten().cpu().numpy()
    predicted_forces = predicted_forces.detach().flatten().cpu().numpy()
    data_forces = [[x, y] for x, y in zip(target_forces, predicted_forces)]
    table_forces = wandb.Table(
        data=data_forces, columns=["Target forces", "Predicted forces"]
    )
    return table_energies, table_forces


def log_wandb_data(
    predicted_energies,
    predicted_forces,
    target_energies,
    target_forces,
    energies_mae,
    forces_mae,
):
    table_energies, table_forces = get_wandb_datatables(
        torch.cat(predicted_energies, dim=0),
        torch.cat(predicted_forces, dim=0),
        torch.cat(target_energies, dim=0),
        torch.cat(target_forces, dim=0),
    )
    wandb.log(
        {
            "val_energies_pairplot": wandb.plot.scatter(
                table_energies,
                "Target energies",
                "Predicted energies",
                title=f"Energies MAE = {energies_mae:.3f} eV",
            ),
            "val_forces_pairplot": wandb.plot.scatter(
                table_forces,
                "Target forces",
                "Predicted forces",
                title=f"Forces MAE = {forces_mae:.3f} eV/A",
            ),
        }
    )
