# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from dataclasses import dataclass
from typing import Generic, NewType, Optional, TypeVar

import sciline as sl
import scipp as sc

from .logging import NMXLogger

InputFileName = NewType("InputFileName", str)

FileType = TypeVar("FileType")
FileTypeNMX = NewType("FileTypeNMX", str)
FileTypeMcStas = NewType("FileTypeMcStas", str)

# McStas Configurations
MaximumProbability = NewType("MaximumProbability", int)
DefaultMaximumProbability = MaximumProbability(100000)


class McStasEventDataSchema(str):
    """McStas event data schema, suffix of the event data path."""

    ...


DefaultMcStasEventDataSchema = McStasEventDataSchema("p_x_y_n_id_t")
McStasEventDataSchemaWithWaveLength = McStasEventDataSchema("p_x_y_n_id_t_L_L")


def get_file_type_nmx() -> FileTypeNMX:
    return FileTypeNMX("NMX_nexus")


def get_file_type_mcstas() -> FileTypeMcStas:
    return FileTypeMcStas("mcstas")


@dataclass
class LoadedData(sl.Scope[FileType, sc.DataGroup], sc.DataGroup):
    ...


@dataclass
class XList(sl.Scope[FileType, sc.Variable], sc.Variable):
    """List of x."""

    ...


@dataclass
class YList(sl.Scope[FileType, sc.Variable], sc.Variable):
    """List of y."""

    ...


@dataclass
class Weights(sl.Scope[FileType, sc.Variable], sc.Variable):
    """Weights."""

    ...


@dataclass
class TList(sl.Scope[FileType, sc.Variable], sc.Variable):
    """List of time."""

    ...


@dataclass
class IDList(sl.Scope[FileType, sc.Variable], sc.Variable):
    """List of IDs."""

    ...


@dataclass
class Events(sl.Scope[FileType, sc.DataArray], sc.DataArray):
    """Event data."""

    ...


@dataclass
class DataPath(Generic[FileType]):
    entry_path: str
    event_path: str
    file_type: type[FileType]


def get_data_path_mcstas(
    data_path_suffix: Optional[McStasEventDataSchema] = None,
) -> DataPath[FileTypeMcStas]:
    if data_path_suffix is None:
        data_path_suffix = DefaultMcStasEventDataSchema

    return DataPath(
        f"entry1/data/bank01_events_dat_list_{data_path_suffix}",
        "events",
        FileTypeMcStas,
    )


def get_data_path() -> DataPath[FileTypeNMX]:
    return DataPath(
        "/entry/instrument/detector_panel_%d/event_data",
        "event_time_offset",
        FileTypeNMX,
    )


def read_h5file(
    file_name: InputFileName, data_path: DataPath[FileTypeNMX]
) -> LoadedData[FileTypeNMX]:
    ...


def _copy_partial_var(
    var: sc.Variable, dim: str, idx: int, unit: str, dtype: Optional[str] = None
) -> sc.Variable:
    """Retrieve property from variable."""
    original_var = var[dim, idx]
    var = original_var.copy().astype(dtype) if dtype else original_var.copy()
    var.unit = sc.Unit(unit)
    return var


def read_h5file_mcstas(
    file_name: InputFileName, data_path: DataPath[FileTypeMcStas]
) -> LoadedData[FileTypeMcStas]:
    import scippnexus as snx

    with snx.File(file_name) as file:
        var = file[data_path.entry_path][data_path.event_path][()].rename_dims(
            {'dim_0': 'event', 'dim_1': 'property'}
        )

        property_recipes = {
            'weights': {"idx": 0, "unit": 'counts'},
            'x_list': {"idx": 1, "unit": 'm'},
            'y_list': {"idx": 2, "unit": 'm'},
            'id_list': {"idx": 4, "unit": 'dimensionless', "dtype": "int64"},
            't_list': {"idx": 5, "unit": 's'},
        }

        return LoadedData(
            sc.DataGroup(
                **{
                    name: _copy_partial_var(var, dim='property', **recipe)
                    for name, recipe in property_recipes.items()
                }
            )
        )


def get_weights(t_list: TList[FileTypeNMX]) -> Weights[FileTypeNMX]:
    """Get weights of measurement data."""
    return Weights(sc.ones_like(t_list.value))


def get_weights_mcstas(
    loaded_data: LoadedData[FileTypeMcStas], max_prop: MaximumPropability
) -> Weights[FileTypeMcStas]:
    """Get weights of McStas data."""
    weights: Weights = loaded_data['weights']

    return Weights((max_prop / weights.max()) * weights)


def get_t_list(loaded_data: LoadedData[FileTypeNMX]) -> TList[FileTypeNMX]:
    ...


def get_t_list_mcstas(
    loaded_data: LoadedData[FileTypeMcStas],
    logger: Optional[NMXLogger] = None,
) -> TList[FileTypeMcStas]:
    t_list = loaded_data['t_list']

    if logger:
        logger.info("T list range: [ %s, %s ]", t_list.min().value, t_list.max().value)

    return TList(t_list)


def get_id_list(loaded_data: LoadedData[FileTypeNMX]) -> IDList[FileTypeNMX]:
    ...


def get_id_list_mcstas(
    loaded_data: LoadedData[FileTypeMcStas],
) -> IDList[FileTypeMcStas]:
    id_list = loaded_data['id_list']

    return IDList(id_list)


def get_x_list_mcstas(
    loaded_data: LoadedData[FileTypeMcStas],
) -> XList[FileTypeMcStas]:
    return XList(loaded_data['x_list'])


def get_y_list_mcstas(
    loaded_data: LoadedData[FileTypeMcStas],
) -> YList[FileTypeMcStas]:
    return YList(loaded_data['y_list'])


def get_events(
    weights: Weights[FileType], t_list: TList[FileType], id_list: IDList[FileType]
) -> Events[FileType]:
    """get event list depending on dataformart and return dataset"""
    return Events(sc.DataArray(data=weights, coords={'t': t_list, 'id': id_list}))


providers = (
    get_file_type_nmx,
    get_file_type_mcstas,
    read_h5file,
    read_h5file_mcstas,
    get_data_path,
    get_data_path_mcstas,
    get_weights_mcstas,
    get_id_list_mcstas,
    get_x_list_mcstas,
    get_y_list_mcstas,
    get_t_list,
    get_t_list_mcstas,
    get_events,
)
