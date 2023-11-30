from dataclasses import dataclass
from typing import NewType, TypeVar

import sciline as sl
import scipp as sc

MaximumPropability = NewType("MaximumPropability", int)
DefaultMaximumPropability = MaximumPropability(100000)
InputFileName = NewType("FileNameToBeRepacked", str)
PreprocessedFileName = NewType("RepackedFileName", str)
ChunkNeeded = NewType("ChunkNeeded", bool)
FileTypeMcStas = NewType("FileTypeMcStas", str)
FileTypeNMX = NewType("FileTypeNMX", str)


FileType = TypeVar("FileType")


@dataclass
class LoadedData(sl.domain.Generic[FileType]):
    value: sc.Variable


@dataclass
class Transposed(sl.domain.Generic[FileType]):
    value: sc.Variable


@dataclass
class Weights(sl.domain.Generic[FileType]):
    """Weights for each event"""

    value: sc.Variable


@dataclass
class TList(sl.domain.Generic[FileType]):
    """List of time for each event"""

    value: sc.Variable


@dataclass
class IDList(sl.domain.Generic[FileType]):
    """List of IDs for each event"""

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


def get_data_path(file: FileType) -> LoadedData[FileType]:
    """try to find data in file"""
    # TODO: Are they conventions?
    # file_type = "none"
    # try:
    #     a = file['entry1/data']['bank01_events_dat_list_p_x_y_n_id_t']['events'][...]
    #     file_type = 'mcstas'
    # except:
    #     try:
    #         a = file['entry1/data']['bank01_events_dat_list_p_x_y_n_id_t_L_L'][
    #             'events'
    #         ][...]
    #         file_type = 'mcstas_L'
    #     except:
    #         try:
    #             a = file['/entry/instrument/detector_panel_0/event_data/'][
    #                 'event_time_offset'
    #             ][...]
    #             file_type = 'NMX_nexus'
    #         except:
    #             print("do data found in file")
    #             file_type = 'none'
    # return a, file_type
    ...


def get_transposed(loaded_data: LoadedData[FileType]) -> Transposed[FileType]:
    """get transposed data"""
    # TODO: How should they be transposed?
    # i.e. which dimensions should be transposed?
    # print("shape of event list (p_x_y_n_id_t)", d.shape)
    # TODO: log shape of event list
    return Transposed(sc.transpose(loaded_data.value))


def get_weights(t_list: TList[FileTypeNMX]) -> Weights[FileTypeNMX]:
    """get weight for each event"""
    return Weights(sc.ones_like(t_list.value))


def get_mcstas_weights(
    transposed_d: Transposed[FileTypeMcStas], max_prop: MaximumPropability
) -> Weights[FileTypeMcStas]:
    """get weight for each event"""
    # delete for actual data  TODO: Check if this means file type.
    weights = sc.array(
        dims=['x'], unit='counts', values=transposed_d.value[0]
    )  # change to integer for measured data
    return Weights(weights * (max_prop / weights.max()))


def get_t_list(loaded_data: LoadedData[FileTypeNMX]) -> TList[FileTypeNMX]:
    ...


def get_t_list_mcstas(
    transposed_d: Transposed[FileTypeMcStas],
) -> TList[FileTypeMcStas]:
    # print("tlist range",t_list.min(), t_list.max()) TODO: log range of t_list
    return TList(sc.array(dims=['x'], unit='s', values=transposed_d.value[5]))


def get_id_list(loaded_data: LoadedData[FileTypeNMX]) -> IDList[FileTypeNMX]:
    ...


def get_id_list_mcstas(
    transposed_d: Transposed[FileTypeMcStas],
) -> IDList[FileTypeMcStas]:
    # Ensure all IDs are recognized
    # print("id min",id_list.values.min())
    # print("id max",id_list.values.max()) TODO: log min and max of id_list
    return IDList(
        sc.array(dims=['x'], unit=None, values=transposed_d.value[4], dtype='int64')
    )


def get_events(
    weights: Weights[FileType], t_list: TList[FileType], id_list: IDList[FileType]
) -> Events[FileType]:
    """get event list depending on dataformart and return dataset"""
    return Events(
        sc.DataArray(
            data=weights.value, coords={'t': t_list.value, 'id': id_list.value}
        )
    )
