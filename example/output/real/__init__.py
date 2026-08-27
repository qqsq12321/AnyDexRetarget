"""Real-hand output drivers, one script per hand type."""

from .drivers_gaia import GaiaHand20Output
from .drivers_inspire import InspireSerialOutput
from .drivers_l20 import LinkerL20V10SerialOutput
from .drivers_linker_l20 import LinkerL20SerialOutput
from .drivers_linker_l20_can import LinkerL20CanOutput
from .drivers_shadow import ShadowTCPOutput
from .drivers_wuji import WujiOutput

__all__ = [
    "GaiaHand20Output",
    "InspireSerialOutput",
    "LinkerL20CanOutput",
    "LinkerL20SerialOutput",
    "LinkerL20V10SerialOutput",
    "ShadowTCPOutput",
    "WujiOutput",
]
