from typing import Union

import equistore
import numpy as np
import torch
from equistore import Labels, TensorBlock, TensorMap
from torch_geometric.data import Batch

from .torch_vector_expansion import VectorExpansionCalculator


class SphericalExpansionCalculator(torch.nn.Module):
    def __init__(
        self,
        all_species: Union[list, np.ndarray],
        cutoff_radius: float,
        basis_cutoff: float,
        num_pseudo_species: int = None,
        device: torch.device = None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self.all_species = np.array(all_species, dtype=np.int32)
        self.num_pseudo_species = num_pseudo_species
        self.vector_expansion_calculator = VectorExpansionCalculator(
            cutoff_radius, basis_cutoff, device=device
        )

        if num_pseudo_species is not None:
            self.is_alchemical = True
            self.all_species_labels = Labels(
                names=["species_neighbor"], values=all_species[:, None]
            )
            self.combination_matrix = torch.nn.Linear(
                all_species.shape[0], self.num_pseudo_species, bias=False
            )
        else:
            self.is_alchemical = False

    def forward(self, data: Batch) -> TensorMap:
        expanded_vectors = self.vector_expansion_calculator(data)
        samples_metadata = expanded_vectors.block(l=0).samples

        if self.is_alchemical:
            s_metadata = samples_metadata["structure"]
            i_metadata = samples_metadata["center"]
            ai_metadata = samples_metadata["species_center"]
            one_hot_aj = torch.tensor(
                equistore.one_hot(samples_metadata, self.all_species_labels),
                dtype=torch.get_default_dtype(),
                device=expanded_vectors.block(l=0).values.device,
            )
            pseudo_species_weights = self.combination_matrix(one_hot_aj)

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

            l_max = self.vector_expansion_calculator.l_max
            n_centers = len(unique_s_i_indices)
            density_indices = torch.LongTensor(s_i_metadata_to_unique)
            densities = []
            for l in range(l_max + 1):
                expanded_vectors_l = expanded_vectors.block(l=l).values
                expanded_vectors_l_pseudo = torch.einsum(
                    "abc, ad -> abcd", expanded_vectors_l, pseudo_species_weights
                )
                densities_l = torch.zeros(
                    (
                        n_centers,
                        expanded_vectors_l.shape[1],
                        expanded_vectors_l.shape[2],
                        self.num_pseudo_species,
                    ),
                    dtype=torch.get_default_dtype(),
                    device=expanded_vectors_l.device,
                )
                densities_l.index_add_(
                    dim=0,
                    index=density_indices.to(expanded_vectors_l.device),
                    source=expanded_vectors_l_pseudo,
                )
                densities_l = (
                    densities_l.reshape(
                        (n_centers, 2 * l + 1, -1, self.num_pseudo_species)
                    )
                    .swapaxes(2, 3)
                    .reshape((n_centers, 2 * l + 1, -1))
                )
                densities.append(densities_l)

            ai_new_indices = torch.tensor(ai_metadata[s_i_unique_to_metadata])
            labels = []
            blocks = []
            for l in range(l_max + 1):
                densities_l = densities[l]
                vectors_l_block = expanded_vectors.block(l=l)
                vectors_l_block_components = vectors_l_block.components
                vectors_l_block_n = vectors_l_block.properties["n"]
                for a_i in self.all_species:
                    where_ai = torch.LongTensor(np.where(ai_new_indices == a_i)[0]).to(
                        densities_l.device
                    )
                    densities_ai_l = torch.index_select(densities_l, 0, where_ai)
                    labels.append([a_i, l, 1])
                    blocks.append(
                        TensorBlock(
                            values=densities_ai_l,
                            samples=Labels(
                                names=["structure", "center"],
                                values=unique_s_i_indices[where_ai.cpu().numpy()],
                            ),
                            components=vectors_l_block_components,
                            properties=Labels(
                                names=["a1", "n1", "l1"],
                                values=np.stack(
                                    [
                                        np.repeat(
                                            -np.arange(self.num_pseudo_species),
                                            vectors_l_block_n.shape[0],
                                        ),
                                        np.tile(
                                            vectors_l_block_n, self.num_pseudo_species
                                        ),
                                        l
                                        * np.ones(
                                            (densities_ai_l.shape[2],), dtype=np.int32
                                        ),
                                    ],
                                    axis=1,
                                ),
                            ),
                        )
                    )

        else:
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

            aj_shifts = np.array(
                [species_to_index[aj_index] for aj_index in aj_metadata]
            )
            density_indices = torch.LongTensor(
                s_i_metadata_to_unique * n_species + aj_shifts
            )

            l_max = self.vector_expansion_calculator.l_max
            n_centers = len(unique_s_i_indices)
            densities = []
            for l in range(l_max + 1):
                expanded_vectors_l = expanded_vectors.block(l=l).values
                densities_l = torch.zeros(
                    (
                        n_centers * n_species,
                        expanded_vectors_l.shape[1],
                        expanded_vectors_l.shape[2],
                    ),
                    dtype=torch.get_default_dtype(),
                    device=expanded_vectors_l.device,
                )
                densities_l.index_add_(
                    dim=0,
                    index=density_indices.to(expanded_vectors_l.device),
                    source=expanded_vectors_l,
                )
                densities_l = (
                    densities_l.reshape((n_centers, n_species, 2 * l + 1, -1))
                    .swapaxes(1, 2)
                    .reshape((n_centers, 2 * l + 1, -1))
                )
                densities.append(densities_l)

            ai_new_indices = torch.tensor(ai_metadata[s_i_unique_to_metadata])
            labels = []
            blocks = []
            for l in range(l_max + 1):
                densities_l = densities[l]
                vectors_l_block = expanded_vectors.block(l=l)
                vectors_l_block_components = vectors_l_block.components
                vectors_l_block_n = vectors_l_block.properties["n"]
                for a_i in self.all_species:
                    where_ai = torch.LongTensor(np.where(ai_new_indices == a_i)[0]).to(
                        densities_l.device
                    )
                    densities_ai_l = torch.index_select(densities_l, 0, where_ai)
                    labels.append([a_i, l, 1])
                    blocks.append(
                        TensorBlock(
                            values=densities_ai_l,
                            samples=Labels(
                                names=["structure", "center"],
                                values=unique_s_i_indices[where_ai.cpu().numpy()],
                            ),
                            components=vectors_l_block_components,
                            properties=Labels(
                                names=["a1", "n1", "l1"],
                                values=np.stack(
                                    [
                                        np.repeat(
                                            self.all_species, vectors_l_block_n.shape[0]
                                        ),
                                        np.tile(
                                            vectors_l_block_n, self.all_species.shape[0]
                                        ),
                                        l
                                        * np.ones(
                                            (densities_ai_l.shape[2],), dtype=np.int32
                                        ),
                                    ],
                                    axis=1,
                                ),
                            ),
                        )
                    )

        spherical_expansion = TensorMap(
            keys=Labels(
                names=["a_i", "lam", "sigma"], values=np.array(labels, dtype=np.int32)
            ),
            blocks=blocks,
        )

        return spherical_expansion
