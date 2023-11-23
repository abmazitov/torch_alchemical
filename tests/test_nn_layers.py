import torch
import metatensor
from torch_alchemical import nn


torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


def evaluate_layer(layer, ps, ref_ps):
    with torch.no_grad():
        layer_ps = layer(ps)
    assert metatensor.operations.allclose(layer_ps, ref_ps, atol=1e-5, rtol=1e-5)


class TestNNLayers:
    ps = metatensor.torch.load("./tests/data/ps_test_data.npz")
    unique_numbers = ps.keys.values.flatten().tolist()
    emb_ps = metatensor.torch.load("./tests/data/emb_ps_test_data.npz")
    ps_input_size = ps.block(0).values.shape[-1]
    contraction_layer = torch.load("./tests/data/contraction_layer.pt")
    num_channels = 4

    def test_alchemical_embedding(self):
        torch.manual_seed(0)
        layer = nn.AlchemicalEmbedding(
            self.unique_numbers, self.num_channels, self.contraction_layer
        )
        ref_ps = metatensor.torch.load(
            "./tests/data/emb_ps_test_data.npz",
        )
        evaluate_layer(layer, self.ps, ref_ps)

    def test_multi_channel_linear(self):
        torch.manual_seed(0)
        layer = nn.MultiChannelLinear(
            self.ps_input_size,
            1,
            self.num_channels,
        )
        ref_ps = metatensor.torch.load(
            "./tests/data/mclinear_ps_test_data.npz",
        )
        evaluate_layer(layer, self.emb_ps, ref_ps)

    def test_linear(self):
        torch.manual_seed(0)
        layer = nn.Linear(self.ps_input_size, 1)
        ref_ps = metatensor.torch.load(
            "./tests/data/linear_ps_test_data.npz",
        )
        evaluate_layer(layer, self.ps, ref_ps)

    def test_linearmap(self):
        torch.manual_seed(0)
        layer = nn.LinearMap(
            self.ps.keys.values.flatten().tolist(), self.ps_input_size, 1
        )
        ref_ps = metatensor.torch.load(
            "./tests/data/linearmap_ps_test_data.npz",
        )
        evaluate_layer(layer, self.ps, ref_ps)

    def test_layer_norm(self):
        torch.manual_seed(0)
        layer = nn.LayerNorm(self.ps_input_size)
        ref_ps = metatensor.torch.load("./tests/data/norm_ps_test_data.npz")
        evaluate_layer(layer, self.ps, ref_ps)

    def test_relu(self):
        layer = nn.ReLU()
        ref_ps = metatensor.torch.load("./tests/data/relu_ps_test_data.npz")
        evaluate_layer(layer, self.ps, ref_ps)

    def test_silu(self):
        layer = nn.SiLU()
        ref_ps = metatensor.torch.load("./tests/data/silu_ps_test_data.npz")
        evaluate_layer(layer, self.ps, ref_ps)

    def test_selu(self):
        layer = nn.SELU()
        ref_ps = metatensor.torch.load("./tests/data/selu_ps_test_data.npz")
        evaluate_layer(layer, self.ps, ref_ps)

    def test_loss_functions(self):
        ref_energies = torch.load("./tests/data/hea_bulk_test_ps_energies.pt")
        ref_forces = torch.load("./tests/data/hea_bulk_test_ps_forces.pt")
        loss_fns = [
            nn.MAELoss(),
            nn.MSELoss(),
            nn.WeightedMAELoss(energies_weight=1.0, forces_weight=1.0),
            nn.WeightedMSELoss(energies_weight=1.0, forces_weight=1.0),
        ]
        for loss_fn in loss_fns:
            loss = loss_fn(
                predicted_energies=ref_energies,
                predicted_forces=ref_forces,
                target_energies=ref_energies,
                target_forces=ref_forces,
            )
            assert torch.allclose(loss, torch.tensor(0.0))
