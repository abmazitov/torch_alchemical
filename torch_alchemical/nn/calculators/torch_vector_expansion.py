import numpy as np
import torch
from equistore import Labels, TensorBlock, TensorMap
from torch_geometric.data import Batch
from torch_spex.angular_basis import AngularBasis
from torch_spex.radial_basis import RadialBasis

from torch_alchemical.utils import get_cartesian_vectors


class VectorExpansionCalculator(torch.nn.Module):
    def __init__(
        self, cutoff_radius: float, basis_cutoff: float, device: torch.device = None
    ):
        super().__init__()
        self.cutoff_radius = cutoff_radius
        self.cutoff_energy = basis_cutoff
        if device is None:
            device = torch.device("cpu")
        self.radial_basis_calculator = RadialBasis(
            {"r_cut": cutoff_radius, "E_max": basis_cutoff}, device=device
        )
        self.l_max = self.radial_basis_calculator.l_max
        self.spherical_harmonics_calculator = AngularBasis(self.l_max)

    def forward(self, data: Batch) -> TensorMap:
        cartesian_vectors = get_cartesian_vectors(data)
        bare_cartesian_vectors = cartesian_vectors.values.squeeze(dim=-1)

        r = torch.sqrt((bare_cartesian_vectors**2).sum(dim=-1))
        radial_basis = self.radial_basis_calculator(r)

        spherical_harmonics = self.spherical_harmonics_calculator(bare_cartesian_vectors)

        # Use broadcasting semantics to get the products in equistore shape
        vector_expansion_blocks = []
        for l, (radial_basis_l, spherical_harmonics_l) in enumerate(
            zip(radial_basis, spherical_harmonics)
        ):
            vector_expansion_l = radial_basis_l.unsqueeze(
                dim=1
            ) * spherical_harmonics_l.unsqueeze(dim=2)
            n_max_l = vector_expansion_l.shape[2]
            vector_expansion_blocks.append(
                TensorBlock(
                    values=vector_expansion_l,
                    samples=cartesian_vectors.samples,
                    components=[
                        Labels(
                            names=("m",),
                            values=np.arange(-l, l + 1, dtype=np.int32).reshape(
                                2 * l + 1, 1
                            ),
                        )
                    ],
                    properties=Labels(
                        names=("n",),
                        values=np.arange(0, n_max_l, dtype=np.int32).reshape(
                            n_max_l, 1
                        ),
                    ),
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
