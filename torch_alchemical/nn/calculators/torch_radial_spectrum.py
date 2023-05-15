import torch
import numpy as np
from torch_spex.radial_basis import RadialBasis
from equistore import TensorMap, TensorBlock, Labels
from torch_alchemical.utils import get_cartesian_vectors
from torch_geometric.data import Batch
from typing import Union


class RadialSpectrumCalculator(torch.nn.Module):
    def __init__(
        self,
        all_species: Union[list, np.ndarray],
        cutoff_radius: float,
        basis_cutoff: float,
        device: torch.device = None,
    ):
        super().__init__()
        self.cutoff_radius = cutoff_radius
        self.basis_cutoff = basis_cutoff
        self.all_species = all_species
        if device is None:
            device = torch.device("cpu")
        self.radial_basis_calculator = RadialBasis(
            {"r_cut": cutoff_radius, "E_max": basis_cutoff}, device=device
        )
        self.l_max = self.radial_basis_calculator.l_max

    def forward(self, batch: Batch) -> TensorMap:
        cartesian_vectors = get_cartesian_vectors(batch)
        edge_vec = cartesian_vectors.values.squeeze(dim=-1)
        r = torch.sum(edge_vec**2, axis=1) ** 0.5
        radial_basis = self.radial_basis_calculator(r)
        radial_expansion_blocks = []
        for l, radial_basis_l in enumerate(radial_basis):
            n_max_l = radial_basis_l.shape[1]
            radial_expansion_blocks.append(
                TensorBlock(
                    values=radial_basis_l,
                    samples=cartesian_vectors.samples,
                    components=[],
                    properties=Labels(
                        names=("n",),
                        values=np.arange(0, n_max_l, dtype=np.int32).reshape(
                            n_max_l, 1
                        ),
                    ),
                )
            )

        l_max = len(radial_expansion_blocks) - 1
        radial_expansion_tmap = TensorMap(
            keys=Labels(
                names=("l",),
                values=np.arange(0, l_max + 1, dtype=np.int32).reshape(l_max + 1, 1),
            ),
            blocks=radial_expansion_blocks,
        )

        samples_metadata = radial_expansion_tmap.block(l=0).samples

        s_metadata = samples_metadata["structure"]
        i_metadata = samples_metadata["center"]
        ai_metadata = samples_metadata["species_center"]
        aj_metadata = samples_metadata["species_neighbor"]

        n_species = len(self.all_species)
        species_to_index = {
            atomic_number: i_species
            for i_species, atomic_number in enumerate(self.all_species)
        }

        s_i_metadata = np.stack([s_metadata, i_metadata], axis=-1)
        (
            unique_s_i_indices,
            s_i_unique_to_metadata,
            s_i_metadata_to_unique,
        ) = np.unique(s_i_metadata, axis=0, return_index=True, return_inverse=True)

        aj_shifts = np.array([species_to_index[aj_index] for aj_index in aj_metadata])
        density_indices = torch.LongTensor(
            s_i_metadata_to_unique * n_species + aj_shifts
        )

        n_centers = len(unique_s_i_indices)

        densities = []
        for l in range(l_max + 1):
            expanded_vectors_l = radial_expansion_tmap.block(l=l).values
            densities_l = torch.zeros(
                (
                    n_centers * n_species,
                    expanded_vectors_l.shape[1],
                ),
                dtype=torch.get_default_dtype(),
                device=expanded_vectors_l.device,
            )
            densities_l.index_add_(
                dim=0,
                index=density_indices.to(expanded_vectors_l.device),
                source=expanded_vectors_l,
            )
            densities_l = densities_l.reshape((n_centers, n_species, -1)).reshape(
                (n_centers, -1)
            )
            densities.append(densities_l)

        ai_new_indices = torch.tensor(ai_metadata[s_i_unique_to_metadata])
        labels = []
        blocks = []
        for l in range(l_max + 1):
            densities_l = densities[l]
            vectors_l_block = radial_expansion_tmap.block(l=l)
            vectors_l_block_n = vectors_l_block.properties["n"]
            for a_i in self.all_species:
                where_ai = torch.LongTensor(np.where(ai_new_indices == a_i)[0]).to(
                    densities_l.device
                )
                densities_ai_l = torch.index_select(densities_l, 0, where_ai)
                labels.append([a_i, l])
                blocks.append(
                    TensorBlock(
                        values=densities_ai_l,
                        samples=Labels(
                            names=["structure", "center"],
                            values=unique_s_i_indices[where_ai.cpu().numpy()],
                        ),
                        components=[],
                        properties=Labels(
                            names=["a1", "n1"],
                            values=np.stack(
                                [
                                    np.repeat(
                                        self.all_species, vectors_l_block_n.shape[0]
                                    ),
                                    np.tile(
                                        vectors_l_block_n, self.all_species.shape[0]
                                    ),
                                ],
                                axis=1,
                            ),
                        ),
                    )
                )

        radial_expansion = TensorMap(
            keys=Labels(names=["a_i", "l"], values=np.array(labels, dtype=np.int32)),
            blocks=blocks,
        )
        radial_expansion = radial_expansion.keys_to_properties("l")

        return radial_expansion
