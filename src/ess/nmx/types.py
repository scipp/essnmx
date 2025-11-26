import enum
from typing import NewType

import scipp as sc


class Compression(enum.StrEnum):
    """Compression type of the output file.

    These options are written as enum for future extensibility.
    """

    NONE = 'NONE'
    BITSHUFFLE_LZ4 = 'BITSHUFFLE_LZ4'


NMXCrystalRotation = NewType("NMXCrystalRotation", sc.Variable)
"""Crystal rotation of the sample."""

TofSimulationMinWavelength = NewType("TofSimulationMinWavelength", sc.Variable)
"""Minimum wavelength for tof simulation to calculate look up table."""

TofSimulationMaxWavelength = NewType("TofSimulationMaxWavelength", sc.Variable)
"""Maximum wavelength for tof simulation to calculate look up table."""
