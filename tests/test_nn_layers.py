import metatensor
import torch
from metatensor.torch import Labels

from torch_alchemical import nn

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


def evaluate_layer(layer, ps, ref_ps):
    with torch.no_grad():
        layer_ps = layer(ps)
    assert metatensor.operations.allclose(layer_ps, ref_ps, atol=1e-4)


ps = metatensor.torch.load("./tests/data/ps_test_data.npz")
unique_numbers = ps.keys.values.flatten().tolist()
emb_ps = metatensor.torch.load("./tests/data/emb_ps_test_data.npz")
ps_input_size = ps.block(0).values.shape[-1]
contraction_layer = torch.load("./tests/data/contraction_layer.pt")
num_channels = 4


def test_alchemical_embedding():
    torch.manual_seed(0)
    layer = nn.AlchemicalEmbedding(unique_numbers, contraction_layer)
    ref_ps = metatensor.torch.load(
        "./tests/data/emb_ps_test_data.npz",
    )
    evaluate_layer(layer, ps, ref_ps)


def test_linear():
    torch.manual_seed(0)
    layer = nn.Linear(ps_input_size, 1)
    ref_ps = metatensor.torch.load(
        "./tests/data/linear_ps_test_data.npz",
    )
    evaluate_layer(layer, ps, ref_ps)


def test_linearmap():
    torch.manual_seed(0)
    linear_layer_keys = Labels(
        names=["a_i"], values=torch.tensor(unique_numbers).view(-1, 1)
    )
    layer = nn.LinearMap(linear_layer_keys, ps_input_size, 1)
    ref_ps = metatensor.torch.load(
        "./tests/data/linearmap_ps_test_data.npz",
    )
    evaluate_layer(layer, ps, ref_ps)


def test_layer_norm():
    torch.manual_seed(0)
    layer = nn.LayerNorm(ps_input_size)
    ref_ps = metatensor.torch.load("./tests/data/norm_ps_test_data.npz")
    evaluate_layer(layer, ps, ref_ps)


def test_relu():
    layer = nn.ReLU()
    ref_ps = metatensor.torch.load("./tests/data/relu_ps_test_data.npz")
    evaluate_layer(layer, ps, ref_ps)


def test_silu():
    layer = nn.SiLU()
    ref_ps = metatensor.torch.load("./tests/data/silu_ps_test_data.npz")
    evaluate_layer(layer, ps, ref_ps)


def test_selu():
    layer = nn.SELU()
    ref_ps = metatensor.torch.load("./tests/data/selu_ps_test_data.npz")
    evaluate_layer(layer, ps, ref_ps)


def test_loss_functions():
    ref_energies = torch.load("./tests/data/hea_bulk_test_ps_energies.pt")
    ref_forces = torch.load("./tests/data/hea_bulk_test_ps_forces.pt")
    loss_fns = [
        nn.MAELoss(),
        nn.MSELoss(),
        nn.HuberLoss(),
        nn.WeightedMAELoss(energies_weight=1.0, forces_weight=1.0),
        nn.WeightedMSELoss(energies_weight=1.0, forces_weight=1.0),
        nn.WeightedHuberLoss(energies_weight=1.0, forces_weight=1.0),
    ]
    for loss_fn in loss_fns:
        loss = loss_fn(
            predicted_energies=ref_energies,
            predicted_forces=ref_forces,
            target_energies=ref_energies,
            target_forces=ref_forces,
        )
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-4)
