# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E402, F401

import importlib.metadata

try:
    __version__ = importlib.metadata.version("essnmx")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

from .data import small_mcstas_3_sample
from .reduction import NMXReducedData
from .types import MaximumCounts, NMXRawData

default_parameters = {MaximumCounts: 10_000}

del MaximumCounts

__all__ = [
    "NMXRawData",
    "NMXReducedData",
    "default_parameters",
    "small_mcstas_3_sample",
]
