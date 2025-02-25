# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)


def McStasWorkflow():
    import sciline as sl

    from ess.nmx.reduction import format_nmx_reduced_data, reduce_raw_event_counts

    from .load import providers as loader_providers
    from .xml import read_mcstas_geometry_xml

    return sl.Pipeline(
        (
            *loader_providers,
            read_mcstas_geometry_xml,
            reduce_raw_event_counts,
            format_nmx_reduced_data,
        )
    )
