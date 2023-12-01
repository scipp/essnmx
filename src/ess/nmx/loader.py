# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from dataclasses import dataclass
from typing import Generic, NewType, TypeVar

import sciline as sl
import scipp as sc

MaximumPropability = NewType("MaximumPropability", int)
DefaultMaximumPropability = MaximumPropability(100000)
InputFileName = NewType("FileNameToBeRepacked", str)
PreprocessedFileName = NewType("RepackedFileName", str)
ChunkNeeded = NewType("ChunkNeeded", bool)

FileTypeMcStas = NewType("FileTypeMcStas", str)
FileTypeMcStasL = NewType("FileTypeMcStasL", str)
FileTypeNMX = NewType("FileTypeNMX", str)


FileType = TypeVar("FileType")
FileTypeMcStasT = TypeVar("FileTypeMcStasT")


def get_file_type_nmx() -> FileTypeNMX:
    return FileTypeNMX("NMX_nexus")


def get_file_type_mcstas() -> FileTypeMcStas:
    return FileTypeMcStas("mcstas")


def get_file_type_mcstas_l() -> FileTypeMcStasL:
    return FileTypeMcStasL("mcstas_L")


@dataclass
class LoadedData(Generic[FileType]):
    value: sc.DataGroup


@dataclass
class XList(Generic[FileType]):
    """List of x."""

    value: sc.Variable


@dataclass
class YList(Generic[FileType]):
    """List of y."""

    value: sc.Variable


@dataclass
class Weights(Generic[FileType]):
    """Weights."""

    value: sc.Variable


@dataclass
class TList(Generic[FileType]):
    """List of time."""

    value: sc.Variable


@dataclass
class IDList(Generic[FileType]):
    """List of IDs."""

    value: sc.Variable


@dataclass
class Events(sl.domain.Generic[FileType]):
    """Events"""

    value: sc.DataArray


def preprocess_file(
    file_name: InputFileName, chunk_needed: ChunkNeeded
) -> PreprocessedFileName:
    # TODO: Is it always needed?
    # If so, we can do sth like...
    # import subprocess
    # if chunk_needed:
    #     suffix = "-rechunk.h5"
    #     chunk_size = "1024x6"
    # else:
    #     suffix = "-nochunk.h5"
    #     chucnk_size = "NONE"

    # processed_file_name = PreprocessedFileName(file_name.replace(".h5", suffix))

    # subprocess.run(
    #   ["h5repack", "-l", f"CHUNK={chunk_size}", file_name, processed_file_name]
    # )
    ...


@dataclass
class DataPath(Generic[FileType]):
    entry_path: str
    event_path: str
    file_type: type[FileType]


def get_data_path_mcstas() -> DataPath[FileTypeMcStas]:
    return DataPath(
        "entry1/data/bank01_events_dat_list_p_x_y_n_id_t", "events", FileTypeMcStas
    )


def get_l_data_path_mcstas() -> DataPath[FileTypeMcStasL]:
    return DataPath(
        "entry1/data/bank01_events_dat_list_p_x_y_n_id_t_L_L", "events", FileTypeMcStasL
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


def read_h5file_mcstas(
    file_name: InputFileName, data_path: DataPath[FileTypeMcStasT]
) -> LoadedData[FileTypeMcStasT]:
    import scippnexus as snx

    with snx.File(file_name) as file:
        var = file[data_path.entry_path][data_path.event_path][()].rename_dims(
            {'dim_0': 'x', 'dim_1': 'property'}
        )
        weights = var['property', 0].copy()
        weights.unit = sc.Unit('counts')
        x_list = var['property', 1].copy()
        x_list.unit = sc.Unit('m')
        y_list = var['property', 2].copy()
        y_list.unit = sc.Unit('m')
        # TODO: What is in dim_1, 3?
        id_list = var['property', 4].copy().astype('int64')
        id_list.unit = sc.Unit('dimensionless')
        t_list = var['property', 5].copy()
        t_list.unit = sc.Unit('s')

        return LoadedData(
            sc.DataGroup(
                weights=weights,
                t_list=t_list,
                id_list=id_list,
                x_list=x_list,
                y_list=y_list,
            )
        )


def get_weights(t_list: TList[FileTypeNMX]) -> Weights[FileTypeNMX]:
    """get weight."""
    return Weights(sc.ones_like(t_list.value))


def get_weights_mcstas(
    loaded_data: LoadedData[FileTypeMcStasT], max_prop: MaximumPropability
) -> Weights[FileTypeMcStasT]:
    """get weight."""
    # delete for actual data  TODO: Check if this means file type.
    weights = loaded_data.value['weights']

    return Weights((max_prop / weights.max()) * weights)


def get_t_list(loaded_data: LoadedData[FileTypeNMX]) -> TList[FileTypeNMX]:
    ...


def get_t_list_mcstas(
    loaded_data: LoadedData[FileTypeMcStasT],
) -> TList[FileTypeMcStasT]:
    # print("tlist range",t_list.min(), t_list.max()) TODO: log range of t_list
    t_list = loaded_data.value['t_list']

    return TList(t_list)


def get_id_list(loaded_data: LoadedData[FileTypeNMX]) -> IDList[FileTypeNMX]:
    ...


def get_id_list_mcstas(
    loaded_data: LoadedData[FileTypeMcStasT],
) -> IDList[FileTypeMcStasT]:
    # Ensure all IDs are recognized
    # print("id min",id_list.values.min())
    # print("id max",id_list.values.max()) TODO: log min and max of id_list
    id_list = loaded_data.value['id_list']

    return IDList(id_list)


def get_x_list_mcstas(
    loaded_data: LoadedData[FileTypeMcStasT],
) -> XList[FileTypeMcStasT]:
    return XList(loaded_data.value['x_list'])


def get_y_list_mcstas(
    loaded_data: LoadedData[FileTypeMcStasT],
) -> YList[FileTypeMcStasT]:
    return YList(loaded_data.value['y_list'])


def get_events(
    weights: Weights[FileType], t_list: TList[FileType], id_list: IDList[FileType]
) -> Events[FileType]:
    """get event list depending on dataformart and return dataset"""
    return Events(
        sc.DataArray(
            data=weights.value, coords={'t': t_list.value, 'id': id_list.value}
        )
    )


providers = [
    get_file_type_nmx,
    get_file_type_mcstas,
    get_file_type_mcstas_l,
    read_h5file,
    read_h5file_mcstas,
    get_data_path,
    get_data_path_mcstas,
    get_l_data_path_mcstas,
    get_weights_mcstas,
    get_id_list_mcstas,
    get_x_list_mcstas,
    get_y_list_mcstas,
    get_t_list,
    get_t_list_mcstas,
    get_events,
]
