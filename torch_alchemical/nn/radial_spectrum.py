from typing import Union
import numpy as np
import torch
from torch_alchemical.utils import get_torch_spex_dict
from metatensor.torch import TensorBlock, TensorMap, Labels
from torch_spex.radial_basis import RadialBasis
from torch_spex.spherical_expansions import get_cartesian_vectors
import copy


class RadialSpectrumFeatures(torch.nn.Module):
    def __init__(
        self,
        all_species: Union[list, np.ndarray],
        cutoff_radius: float,
        basis_cutoff: float,
        radial_basis_type: str = "le",
        basis_normalization_factor: float = None,
        trainable_basis: bool = True,
        device: torch.device = None,
    ):
        super().__init__()
        self.all_species = all_species
        self.cutoff_radius = cutoff_radius
        self.basis_cutoff = basis_cutoff
        self.radial_basis_type = radial_basis_type
        self.basis_normalization_factor = basis_normalization_factor
        self.trainable_basis = trainable_basis
        self.device = device
        hypers = {
            "cutoff radius": self.cutoff_radius,
            "radial basis": {
                "type": self.radial_basis_type,
                "E_max": self.basis_cutoff,
                "mlp": self.trainable_basis,
                "scale": 3.0,
                "cost_trade_off": False,
            },
        }
        if self.basis_normalization_factor:
            hypers["normalize"] = self.basis_normalization_factor
        self.rex_calculator = RadialExpansion(
            hypers=hypers,
            all_species=self.all_species,
            device=self.device,
        )

    def forward(
        self,
        positions: list[torch.Tensor],
        cells: list[torch.Tensor],
        numbers: list[torch.Tensor],
        edge_indices: list[torch.Tensor],
        edge_shifts: list[torch.Tensor],
    ):
        batch_dict = get_torch_spex_dict(
            positions, cells, numbers, edge_indices, edge_shifts
        )
        rex = self.rex_calculator(**batch_dict)
        return rex

    @property
    def num_features(self):
        vex_calculator = self.rex_calculator.vector_expansion_calculator
        n_max = vex_calculator.radial_basis_calculator.n_max_l
        l_max = len(n_max) - 1
        n_feat = sum([n_max[l] * len(self.all_species) for l in range(l_max + 1)])
        return n_feat


class VectorExpansion(torch.nn.Module):
    def __init__(self, hypers: dict, all_species, device: str = "cpu") -> None:
        super().__init__()

        self.hypers = hypers
        self.normalize = True if "normalize" in hypers else False
        # radial basis needs to know cutoff so we pass it,
        # as well as whether to normalize or not
        hypers_radial_basis = copy.deepcopy(hypers["radial basis"])
        hypers_radial_basis["r_cut"] = hypers["cutoff radius"]
        hypers_radial_basis["normalize"] = self.normalize
        self.radial_basis_calculator = RadialBasis(
            hypers_radial_basis, all_species, device=device
        )
        self.l_max = self.radial_basis_calculator.l_max

    def forward(
        self,
        positions: list[torch.Tensor],
        cells: list[torch.Tensor],
        species: torch.Tensor,
        cell_shifts: torch.Tensor,
        centers: torch.Tensor,
        pairs: torch.Tensor,
        structure_centers: torch.Tensor,
        structure_pairs: torch.Tensor,
        structure_offsets: torch.Tensor,
    ) -> TensorMap:
        positions = torch.concatenate(positions)
        cells = torch.stack(cells)
        cartesian_vectors = get_cartesian_vectors(
            positions,
            cells,
            species,
            cell_shifts,
            centers,
            pairs,
            structure_centers,
            structure_pairs,
            structure_offsets,
        )

        bare_cartesian_vectors = cartesian_vectors.values.squeeze(dim=-1)
        r = torch.sqrt((bare_cartesian_vectors**2).sum(dim=-1))
        samples_metadata = (
            cartesian_vectors.samples
        )  # This can be needed by the radial basis to do alchemical contractions
        radial_basis = self.radial_basis_calculator(r, samples_metadata)

        # Use broadcasting semantics to get the products in equistore shape
        vector_expansion_blocks = []
        for l, radial_basis_l in enumerate(radial_basis):
            vector_expansion_l = radial_basis_l[:, None, :]
            n_max_l = vector_expansion_l.shape[2]
            properties = Labels.range("n", n_max_l)
            vector_expansion_blocks.append(
                TensorBlock(
                    values=vector_expansion_l.reshape(vector_expansion_l.shape[0], -1),
                    samples=cartesian_vectors.samples,
                    components=[],
                    properties=properties,
                )
            )

        l_max = len(vector_expansion_blocks) - 1
        vector_expansion_tmap = TensorMap(
            keys=Labels(
                names=("l",),
                values=np.arange(0, l_max + 1, dtype=np.int32).reshape(l_max + 1, 1),
            ),
            blocks=vector_expansion_blocks,
        )

        return vector_expansion_tmap


class RadialExpansion(torch.nn.Module):
    """
    The radial expansion coefficients summed over all neighbours.

    """

    def __init__(
        self, hypers: dict, all_species: list[int], device: str = "cpu"
    ) -> None:
        super().__init__()

        self.hypers = hypers
        self.normalize = True if "normalize" in hypers else False
        if self.normalize:
            avg_num_neighbors = hypers["normalize"]
            self.normalization_factor = 1.0 / np.sqrt(avg_num_neighbors)
            self.normalization_factor_0 = 1.0 / avg_num_neighbors ** (3 / 4)
        self.all_species = np.array(
            all_species, dtype=np.int32
        )  # convert potential list to np.array
        self.vector_expansion_calculator = VectorExpansion(
            hypers, self.all_species, device=device
        )
        self.device = device

    def forward(
        self,
        positions: list[torch.Tensor],
        cells: list[torch.Tensor],
        species: torch.Tensor,
        cell_shifts: torch.Tensor,
        centers: torch.Tensor,
        pairs: torch.Tensor,
        structure_centers: torch.Tensor,
        structure_pairs: torch.Tensor,
        structure_offsets: torch.Tensor,
    ) -> TensorMap:
        expanded_vectors = self.vector_expansion_calculator(
            positions,
            cells,
            species,
            cell_shifts,
            centers,
            pairs,
            structure_centers,
            structure_pairs,
            structure_offsets,
        )

        samples_metadata = expanded_vectors.block(l=0).samples
        n_species = len(self.all_species)
        species_to_index = {
            atomic_number: i_species
            for i_species, atomic_number in enumerate(self.all_species)
        }

        unique_s_i_indices = torch.stack(
            (structure_centers.cpu(), centers.cpu()), dim=1
        )
        s_i_metadata_to_unique = structure_offsets[structure_pairs] + pairs[:, 0]

        l_max = self.vector_expansion_calculator.l_max
        n_centers = len(centers)  # total number of atoms in this batch of structures

        densities = []
        aj_metadata = samples_metadata["species_neighbor"]
        aj_shifts = torch.LongTensor(
            [species_to_index[aj_index] for aj_index in aj_metadata]
        )
        density_indices = torch.LongTensor(
            s_i_metadata_to_unique.cpu() * n_species + aj_shifts
        )

        for l in range(l_max + 1):
            expanded_vectors_l = expanded_vectors.block(l=l).values
            densities_l = torch.zeros(
                (
                    n_centers * n_species,
                    expanded_vectors_l.shape[1],
                ),
                dtype=expanded_vectors_l.dtype,
                device=expanded_vectors_l.device,
            )
            densities_l.index_add_(
                dim=0,
                index=density_indices.to(expanded_vectors_l.device),
                source=expanded_vectors_l,
            )
            densities_l = (
                densities_l.reshape((n_centers, n_species, -1))
                .swapaxes(1, 2)
                .reshape((n_centers, -1))
            )  # need to swap n, a indices which are in the wrong order
            densities.append(densities_l)
        unique_species = self.all_species

        # constructs the TensorMap object
        labels = []
        blocks = []
        for l in range(l_max + 1):
            densities_l = densities[l]
            vectors_l_block = expanded_vectors.block(l=l)
            vectors_l_block_components = vectors_l_block.components
            vectors_l_block_n = np.arange(
                len(np.unique(vectors_l_block.properties["n"]))
            )  # Need to be smarter to optimize
            for a_i in self.all_species:
                where_ai = torch.where(species == a_i)[0]
                densities_ai_l = torch.index_select(densities_l, 0, where_ai)
                if self.normalize:
                    if l == 0:
                        # Very high correlations for l = 0: use a stronger normalization
                        densities_ai_l *= self.normalization_factor_0
                    else:
                        densities_ai_l *= self.normalization_factor
                labels.append([a_i, l])
                blocks.append(
                    TensorBlock(
                        values=densities_ai_l,
                        samples=Labels(
                            names=["structure", "center"],
                            values=unique_s_i_indices.numpy()[where_ai.cpu().numpy()],
                        ),
                        components=vectors_l_block_components,
                        properties=Labels(
                            names=["a1", "n1"],
                            values=np.stack(
                                [
                                    np.repeat(
                                        unique_species, vectors_l_block_n.shape[0]
                                    ),
                                    np.tile(vectors_l_block_n, unique_species.shape[0]),
                                ],
                                axis=1,
                            ),
                        ),
                    )
                )

        radial_expansion = TensorMap(
            keys=Labels(names=["a_i", "l1"], values=np.array(labels, dtype=np.int32)),
            blocks=blocks,
        )

        radial_expansion = radial_expansion.keys_to_properties("l1")

        return radial_expansion
