# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from .detector import default_params as detector_default_params
from .loader import DefaultMaximumProbability, MaximumProbability
from .loader import providers as loader_providers
from .logging import logging_providers
from .reduction import providers as reduction_providers

providers = (*logging_providers, *loader_providers, *reduction_providers)


def collect_default_parameters() -> dict:
    return {
        **detector_default_params,
        MaximumProbability: DefaultMaximumProbability,
    }
