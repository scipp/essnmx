# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType, Optional

import sciline as sl

from .detector import default_params as detector_default_params
from .loader import DefaultMaximumPropability, InputFileName, MaximumPropability
from .loader import providers as loader_providers
from .logging import logging_providers
from .reduction import TimeBinStep
from .reduction import providers as reduction_providers

NMXWorkflow = NewType("NMXWorkflow", sl.Pipeline)
NMXProviders = NewType("NMXProviders", list)
NMXParams = NewType("NMXParams", dict)

providers = (*logging_providers, *loader_providers, *reduction_providers)


def collect_default_parameters() -> NMXParams:
    return NMXParams(
        {
            **detector_default_params,
            **{MaximumPropability: DefaultMaximumPropability},
        }
    )


def build_workflow(
    input_file_name: InputFileName,
    maximum_propability: MaximumPropability = DefaultMaximumPropability,
    time_bin_step: Optional[TimeBinStep] = None,
) -> NMXWorkflow:
    """Build workflow with input_file_name included in the params."""
    combined_params = {
        **collect_default_parameters(),
        **{MaximumPropability: maximum_propability, InputFileName: input_file_name},
    }
    if time_bin_step is not None:
        combined_params[TimeBinStep] = time_bin_step

    return NMXWorkflow(sl.Pipeline(list(providers), params=combined_params))
