import json

import metatensor
import numpy as np
import torch
from ase.io import read
from metatensor.torch import Labels, TensorBlock, TensorMap
from torch.utils.data import DataLoader as SpexDataLoader
from torch_geometric.loader import DataLoader
from torch_spex.forces import compute_forces
from torch_spex.normalize import normalize_true as normalize_func
from torch_spex.spherical_expansions import SphericalExpansion
from torch_spex.structures import InMemoryDataset, TransformerNeighborList, collate_nl

from torch_alchemical.data import AtomisticDataset
from torch_alchemical.models import AlchemicalModel
from torch_alchemical.nn import LayerNorm
from torch_alchemical.nn.power_spectrum import PowerSpectrum
from torch_alchemical.transforms import NeighborList
from torch_alchemical.utils import get_torch_spex_dict

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


def normalize_ps(ps):
    new_keys = []
    new_blocks = []
    for key, block in ps.items():
        new_keys.append(key.values)
        values = block.values
        mean = torch.mean(values, dim=-1, keepdim=True)
        centered_values = values - mean
        variance = torch.mean(centered_values**2, dim=-1, keepdim=True)
        new_values = centered_values / torch.sqrt(variance)
        new_blocks.append(
            TensorBlock(
                values=new_values,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
        )
    return TensorMap(
        keys=Labels(
            names=("a_i",),
            values=torch.tensor(new_keys, device=new_blocks[0].values.device).reshape(
                -1, 1
            ),
        ),
        blocks=new_blocks,
    )


class TorchSpexAlchemicalModel(torch.nn.Module):
    def __init__(
        self,
        hypers,
        all_species,
        n_pseudo,
        average_number_of_atoms,
        num_hidden=64,
        do_forces=False,
    ) -> None:
        super().__init__()
        self.all_species = all_species
        self.spherical_expansion_calculator = SphericalExpansion(hypers, all_species)
        vex_calculator = self.spherical_expansion_calculator.vector_expansion_calculator
        n_max = vex_calculator.radial_basis_calculator.n_max_l
        l_max = len(n_max) - 1
        n_feat = sum(
            [n_max[l] ** 2 * n_pseudo**2 for l in range(l_max + 1)]  # noqa E471
        )
        self.ps_calculator = PowerSpectrum(l_max, all_species)
        self.combination_matrix = (
            vex_calculator.radial_basis_calculator.combination_matrix
        )
        self.all_species_labels = metatensor.torch.Labels(
            names=["a_i"],
            values=torch.tensor(all_species).reshape(-1, 1),
        )
        self.n_pseudo = n_pseudo
        self.average_number_of_atoms = average_number_of_atoms
        self.nu2_model = torch.nn.ModuleDict(
            {
                str(alpha_i): torch.nn.Sequential(
                    normalize_func(
                        "linear_no_bias",
                        torch.nn.Linear(n_feat, num_hidden, bias=False),
                    ),
                    normalize_func("activation", torch.nn.SiLU()),
                    normalize_func(
                        "linear_no_bias",
                        torch.nn.Linear(num_hidden, num_hidden, bias=False),
                    ),
                    normalize_func("activation", torch.nn.SiLU()),
                    normalize_func(
                        "linear_no_bias",
                        torch.nn.Linear(num_hidden, num_hidden, bias=False),
                    ),
                    normalize_func("activation", torch.nn.SiLU()),
                    normalize_func(
                        "linear_no_bias", torch.nn.Linear(num_hidden, 1, bias=False)
                    ),
                )
                for alpha_i in range(n_pseudo)
            }
        )
        # """
        self.do_forces = do_forces
        # self.zero_body_energies = torch.nn.Parameter(torch.zeros(len(all_species)))

    def forward(self, structures, is_training=True):

        n_structures = len(structures["cells"])
        energies = torch.zeros((n_structures,), dtype=torch.get_default_dtype())

        if self.do_forces:
            structures["positions"].requires_grad = True

        # print("Calculating spherical expansion")
        spherical_expansion = self.spherical_expansion_calculator(**structures)
        ps = self.ps_calculator(spherical_expansion)
        ps = normalize_ps(ps)

        # print("Calculating energies")
        embedded_features, block = self._calculate_embedded_features(ps)
        self._apply_layer(energies, embedded_features, block, self.nu2_model)
        energies = energies / self.average_number_of_atoms
        # print("Final", torch.mean(energies), get_2_mom(energies))
        # energies += comp @ self.zero_body_energies

        # print("Computing forces by backpropagation")
        if self.do_forces:
            forces = compute_forces(
                energies, structures["positions"], is_training=is_training
            )
        else:
            forces = None  # Or zero-dimensional tensor?

        return energies, forces

    def _calculate_embedded_features(self, tmap):
        tmap = tmap.keys_to_samples("a_i")
        block = tmap.block()
        # print(block.values)
        samples = block.samples
        one_hot_ai = torch.tensor(
            metatensor.torch.one_hot(samples, self.all_species_labels),
            dtype=torch.get_default_dtype(),
            device=block.values.device,
        )
        pseudo_species_weights = self.combination_matrix(one_hot_ai)
        features = block.values.squeeze(dim=1)
        # print("features", torch.mean(features), get_2_mom(features))
        embedded_features = features[:, :, None] * pseudo_species_weights[:, None, :]
        return embedded_features, block

    def _apply_layer(self, energies, embedded_features, block, layer):
        atomic_energies = []
        structure_indices = []
        # print(tmap.block(0).values)

        atomic_energies = torch.zeros(
            (block.values.shape[0],),
            dtype=torch.get_default_dtype(),
            device=block.values.device,
        )
        for alpha_i in range(self.n_pseudo):
            atomic_energies += layer[str(alpha_i)](
                embedded_features[:, :, alpha_i]
            ).squeeze(dim=-1)
        atomic_energies = atomic_energies / np.sqrt(self.n_pseudo)
        # print("total", torch.mean(atomic_energies), get_2_mom(atomic_energies))
        structure_indices = block.samples["structure"]
        energies.index_add_(dim=0, index=structure_indices, source=atomic_energies)
        # print("in", torch.mean(energies), get_2_mom(energies))
        # THIS IN-PLACE MODIFICATION HAS TO CHANGE!


def compare_dicts(data_dict, spex_batch):
    assert data_dict.keys() == spex_batch.keys()
    for key in data_dict.keys():
        obj = data_dict[key]
        target = spex_batch[key]
        equal = torch.equal(obj, target)
        dtype_equal = obj.dtype == target.dtype
        assert equal and dtype_equal


class TestTorchSpexCompatibility:
    device = "cpu"
    frames = read("./tests/data/hea_bulk_test_sample.xyz", index=":")
    all_species = np.unique(np.hstack([frame.numbers for frame in frames]))
    with open("./tests/configs/default_hypers_alchemical.json", "r") as f:
        hypers_spex = json.load(f)
    transforms = [NeighborList(cutoff_radius=hypers_spex["cutoff radius"])]
    dataset = AtomisticDataset(
        frames, target_properties=["energies"], transforms=transforms
    )
    dataloader = DataLoader(dataset, batch_size=len(frames), shuffle=False)
    batch = next(iter(dataloader))

    spex_transformers = [TransformerNeighborList(cutoff=hypers_spex["cutoff radius"])]
    spex_dataset = InMemoryDataset(frames, spex_transformers)
    spex_loader = SpexDataLoader(
        spex_dataset, batch_size=len(frames), collate_fn=collate_nl
    )
    spex_batch = next(iter(spex_loader))

    def test_batch_to_torch_spex_dict_conversion(self):
        data_dict = get_torch_spex_dict(
            positions=self.batch.pos,
            cells=self.batch.cell,
            numbers=self.batch.numbers,
            edge_indices=self.batch.edge_index,
            edge_offsets=self.batch.edge_offsets,
            batch=self.batch.batch,
        )
        compare_dicts(data_dict, self.spex_batch)

    def test_spherical_expansion_coefficients(self):
        data_dict = get_torch_spex_dict(
            positions=self.batch.pos,
            cells=self.batch.cell,
            numbers=self.batch.numbers,
            edge_indices=self.batch.edge_index,
            edge_offsets=self.batch.edge_offsets,
            batch=self.batch.batch,
        )
        calculator = SphericalExpansion(
            hypers=self.hypers_spex, all_species=self.all_species
        )
        with torch.no_grad():
            data_tm = calculator.forward(**data_dict)
            spex_tm = calculator.forward(**self.spex_batch)
        assert metatensor.operations.allclose(data_tm, spex_tm, atol=1e-4)


class TestAlchemicalModelCompatibility:
    device = "cpu"
    frames = read("./tests/data/hea_bulk_test_sample.xyz", index=":")
    all_species = np.unique(np.hstack([frame.numbers for frame in frames])).tolist()
    with open("./tests/configs/default_hypers_alchemical.json", "r") as f:
        hypers_spex = json.load(f)
        hypers_spex["normalize"] = 1.0
        hidden_sizes = [64, 64, 64]
    transforms = [NeighborList(cutoff_radius=hypers_spex["cutoff radius"])]
    dataset = AtomisticDataset(
        frames, target_properties=["energies"], transforms=transforms
    )
    dataloader = DataLoader(dataset, batch_size=len(frames), shuffle=False)
    batch = next(iter(dataloader))

    spex_transformers = [TransformerNeighborList(cutoff=hypers_spex["cutoff radius"])]
    spex_dataset = InMemoryDataset(frames, spex_transformers)
    spex_loader = SpexDataLoader(
        spex_dataset, batch_size=len(frames), collate_fn=collate_nl
    )
    spex_batch = next(iter(spex_loader))

    torch.manual_seed(0)
    spex_model = TorchSpexAlchemicalModel(
        hypers_spex,
        all_species,
        num_hidden=64,
        n_pseudo=hypers_spex["alchemical"],
        average_number_of_atoms=1.0,
        do_forces=False,
    )

    torch.manual_seed(0)
    model = AlchemicalModel(
        hidden_sizes=hidden_sizes,
        output_size=1,
        num_pseudo_species=hypers_spex["alchemical"],
        normalize=True,
        unique_numbers=all_species,
        cutoff=hypers_spex["cutoff radius"],
        contract_center_species=True,
        trainable_basis=hypers_spex["radial basis"]["mlp"],
        radial_basis_type=hypers_spex["radial basis"]["type"],
        basis_cutoff_power_spectrum=hypers_spex["radial basis"]["E_max"],
        basis_scale=hypers_spex["radial basis"]["scale"],
    )

    vex_calculator = (
        spex_model.spherical_expansion_calculator.vector_expansion_calculator
    )
    rb_calculator = vex_calculator.radial_basis_calculator
    contraction_layer = vex_calculator.radial_basis_calculator.combination_matrix
    model.ps_features_layer.spex_calculator.vector_expansion_calculator = vex_calculator
    model.embedding.contraction_layer = contraction_layer

    layer_norm = LayerNorm(model.ps_features_layer.num_features, eps=0.0)

    for i in range(model.num_pseudo_species):
        for j in range(len(model.nn)):
            if j % 2 == 0:
                layer = model.nn[j].linear[str(i)]
                spex_layer = spex_model.nu2_model[str(i)][j]
                layer.weight.data = (
                    spex_layer.linear_layer.weight.data
                    * spex_layer.normalization_factor
                )
            else:
                layer = model.nn[j]
                spex_layer = spex_model.nu2_model[str(i)][j]
                spex_layer.normalization_factor = layer.scaling

    def test_power_spectrum_compatibility(self):
        spherical_expansion = self.spex_model.spherical_expansion_calculator(
            **self.spex_batch
        )
        spex_ps = self.spex_model.ps_calculator(spherical_expansion)
        ps = self.model.ps_features_layer(
            positions=self.batch.pos,
            numbers=self.batch.numbers,
            cells=self.batch.cell,
            edge_indices=self.batch.edge_index,
            edge_offsets=self.batch.edge_offsets,
            batch=self.batch.batch,
        )
        assert metatensor.torch.allclose(spex_ps, ps, atol=1e-10)

    def test_normalized_ps(self):
        ps = self.model.ps_features_layer(
            positions=self.batch.pos,
            numbers=self.batch.numbers,
            cells=self.batch.cell,
            edge_indices=self.batch.edge_index,
            edge_offsets=self.batch.edge_offsets,
            batch=self.batch.batch,
        )
        spex_normalized_ps = normalize_ps(ps)
        normalized_ps = self.layer_norm(ps)
        model_normalized_ps = self.model.layer_norm(ps)
        assert metatensor.torch.allclose(
            spex_normalized_ps, model_normalized_ps, atol=1e-8
        )
        assert metatensor.torch.allclose(spex_normalized_ps, normalized_ps, atol=1e-10)

    def test_embedded_features(self):
        ps = self.model.ps_features_layer(
            positions=self.batch.pos,
            numbers=self.batch.numbers,
            cells=self.batch.cell,
            edge_indices=self.batch.edge_index,
            edge_offsets=self.batch.edge_offsets,
            batch=self.batch.batch,
        )
        emb_ps = self.model.embedding(ps)
        emb_ps = emb_ps.keys_to_samples(["a_i"])
        emb_features = torch.stack([block.values for block in emb_ps], dim=-1)
        spex_emb_features, _ = self.spex_model._calculate_embedded_features(ps)
        assert torch.allclose(spex_emb_features, emb_features, atol=1e-10)

    def test_nn_layers(self):
        ps = self.model.ps_features_layer(
            positions=self.batch.pos,
            numbers=self.batch.numbers,
            cells=self.batch.cell,
            edge_indices=self.batch.edge_index,
            edge_offsets=self.batch.edge_offsets,
            batch=self.batch.batch,
        )
        emb_ps = self.model.embedding(ps)
        emb_ps = emb_ps.keys_to_samples("a_i")
        emb_ps = metatensor.torch.sum_over_samples(emb_ps, "a_i")

        spex_embedded_features, _ = self.spex_model._calculate_embedded_features(ps)

        for alpha_i in range(self.spex_model.n_pseudo):
            for j in range(1, len(self.model.nn) + 1):
                layers = self.model.nn[:j]
                nn_ps = emb_ps.copy()
                for layer in layers:
                    nn_ps = layer(nn_ps)
                spex_layers = self.spex_model.nu2_model[str(alpha_i)][:j]
                output = nn_ps.block(alpha_i).values
                spex_output = spex_layers(spex_embedded_features[:, :, alpha_i])
                assert torch.allclose(output, spex_output)

        nn_ps = emb_ps.copy()
        for layer in self.model.nn:
            nn_ps = layer(nn_ps)

        for alpha_i in range(self.spex_model.n_pseudo):
            spex_output = self.spex_model.nu2_model[str(alpha_i)](
                spex_embedded_features[:, :, alpha_i]
            )
            output = nn_ps.block(alpha_i).values
            assert torch.allclose(output, spex_output)

    def test_full_forward_pass(self):
        energies = self.model(
            positions=self.batch.pos,
            cells=self.batch.cell,
            numbers=self.batch.numbers,
            edge_indices=self.batch.edge_index,
            edge_offsets=self.batch.edge_offsets,
            batch=self.batch.batch,
        )
        spex_energies, _ = self.spex_model(
            structures=self.spex_batch, is_training=False
        )
        assert torch.allclose(energies.squeeze(dim=-1), spex_energies, atol=1e-10)
