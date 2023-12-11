# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from logging import Logger
from typing import NewType

NMXLogger = NewType("NMXLogger", Logger)


def get_logger() -> NMXLogger:
    """Return scipp logger for logging nmx data reduction."""
    from scipp.logging import get_logger

    return NMXLogger(get_logger())
