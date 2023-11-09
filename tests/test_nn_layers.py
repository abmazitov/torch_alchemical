import torch
import metatensor
from torch_alchemical import nn


torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


class TestNNLayers:
    ps = metatensor.torch.load("./tests/data/ps_test_data.npz")
    ps_input_size = ps.block(0).values.shape[-1]

    def test_linear(self):
        torch.manual_seed(0)
        linear = nn.Linear(self.ps_input_size, 1)
        with torch.no_grad():
            linear_ps = linear(self.ps)
        ref_linear_ps = metatensor.torch.load(
            "./tests/data/linear_ps_test_data.npz",
        )
        assert metatensor.operations.allclose(
            linear_ps, ref_linear_ps, atol=1e-5, rtol=1e-5
        )

    def test_linearmap(self):
        torch.manual_seed(0)
        linear = nn.LinearMap(
            self.ps.keys.values.flatten().tolist(), self.ps_input_size, 1
        )
        with torch.no_grad():
            linear_ps = linear(self.ps)
        ref_linear_ps = metatensor.torch.load(
            "./tests/data/linearmap_ps_test_data.npz",
        )
        assert metatensor.operations.allclose(
            linear_ps, ref_linear_ps, atol=1e-5, rtol=1e-5
        )

    def test_layer_norm(self):
        torch.manual_seed(0)
        norm = nn.LayerNorm(self.ps_input_size)
        with torch.no_grad():
            norm_ps = norm(self.ps)
        ref_norm_ps = metatensor.torch.load("./tests/data/norm_ps_test_data.npz")
        assert metatensor.operations.allclose(
            norm_ps, ref_norm_ps, atol=1e-5, rtol=1e-5
        )

    def test_relu(self):
        relu = nn.ReLU()
        ps_relu = relu(self.ps)
        ref_ps_relu = metatensor.torch.load("./tests/data/relu_ps_test_data.npz")
        assert metatensor.operations.allclose(
            ps_relu, ref_ps_relu, atol=1e-5, rtol=1e-5
        )

    def test_silu(self):
        silu = nn.SiLU()
        ps_silu = silu(self.ps)
        ref_ps_silu = metatensor.torch.load("./tests/data/silu_ps_test_data.npz")
        assert metatensor.operations.allclose(
            ps_silu, ref_ps_silu, atol=1e-5, rtol=1e-5
        )

    def test_selu(self):
        selu = nn.SELU()
        ps_selu = selu(self.ps)
        ref_ps_selu = metatensor.torch.load("./tests/data/selu_ps_test_data.npz")
        assert metatensor.operations.allclose(
            ps_selu, ref_ps_selu, atol=1e-5, rtol=1e-5
        )

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
