# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)


def McStasWorkflow():
    import sciline as sl

    from ess.nmx.reduction import bin_time_of_arrival, format_nmx_reduced_data
    from .reduction import event_counts_from_probability, maximum_probability
    from .load import providers as loader_providers
    from .xml import read_mcstas_geometry_xml

    return sl.Pipeline(
        (
            *loader_providers,
            read_mcstas_geometry_xml,
            bin_time_of_arrival,
            format_nmx_reduced_data,
            event_counts_from_probability,
            maximum_probability,
        )
    )
