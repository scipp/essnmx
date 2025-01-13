# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)


def McStasWorkflow():
    import sciline as sl

    from ess.nmx.reduction import bin_time_of_arrival

    from .load import providers as loader_providers
    from .geometry import read_mcstas_geometry_xml

    return sl.Pipeline(
        (*loader_providers, read_mcstas_geometry_xml, bin_time_of_arrival)
    )


def McStas35Workflow():
    import sciline as sl

    from ess.nmx.reduction import bin_time_of_arrival

    from .load import providers as loader_providers
    from .load import (
        load_raw_event_data_mcstas_35,
        load_raw_event_data,
        load_crystal_rotation_mcstas_35,
        load_crystal_rotation,
    )
    from .geometry import load_mcstas_geometry

    loader_providers = list(loader_providers)
    loader_providers.remove(load_raw_event_data)
    loader_providers.remove(load_crystal_rotation)

    return sl.Pipeline(
        (
            *loader_providers,
            load_mcstas_geometry,
            bin_time_of_arrival,
            load_raw_event_data_mcstas_35,
            load_crystal_rotation_mcstas_35,
        )
    )
