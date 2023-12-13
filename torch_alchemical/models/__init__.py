from .power_spectrum_model import PowerSpectrumModel
from .bpps_model import BPPSModel
from .alchemical_model import AlchemicalModel

# from .alchemical_gat import AlchemicalGAT
from .alchemical_gcn import AlchemicalGCN

__all__ = [
    "PowerSpectrumModel",
    "BPPSModel",
    "AlchemicalModel",
    # "AlchemicalGAT",
    "AlchemicalGCN",
]
