# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from dataclasses import dataclass
from typing import NewType

import sciline as sl
import scipp as sc

from .loader import Events, FileType, FileTypeMcStas, FileTypeNMX

TimeBinStep = NewType("TimeBinStep", int)


@dataclass
class PixelIDIntervals(sl.domain.Generic[FileType]):
    """Pixel IDs for each detector"""

    interval_1: tuple
    interval_2: tuple
    interval_3: tuple


class PixelIDs(sl.Scope[FileType, sc.Variable], sc.Variable):
    """Pixel IDs for each detector"""

    ...


class GroupedByPixelID(sl.Scope[FileType, sc.DataArray], sc.DataArray):
    """Grouped events"""

    ...


class TimeBinned(sl.Scope[FileType, sc.DataArray], sc.DataArray):
    """Time binned events"""

    ...


def get_intervals() -> PixelIDIntervals[FileTypeNMX]:
    """Pixel ID intervals for each detector"""
    return PixelIDIntervals[FileTypeNMX](
        (1, 1638401), (1638401, 3276802), (3276802, 4915203)
    )


def get_intervals_mcstas(
    file_type: FileTypeMcStas,
) -> PixelIDIntervals[FileTypeMcStas]:
    """Pixel IDs intervals for each detector"""
    if file_type not in ('mcstas', 'mcstas_L'):
        raise ValueError(f"file_type must be mcstas or mcstas_L, got {file_type}")

    return PixelIDIntervals[FileTypeMcStas](  # Check McStas pixel number conuting
        (1, 1638401), (2000001, 3638401), (4000001, 5638401)
    )


def get_ids(pixel_id_intervals: PixelIDIntervals[FileType]) -> PixelIDs[FileType]:
    """pixel IDs for each detector"""
    intervals = [
        pixel_id_intervals.interval_1,
        pixel_id_intervals.interval_2,
        pixel_id_intervals.interval_3,
    ]
    ids = [sc.arange('id', start, stop) for start, stop in intervals]
    return PixelIDs[FileType](sc.concat(ids, 'id'))


def get_grouped_by_pixel_id(
    da: Events[FileType], ids: PixelIDs[FileType]
) -> GroupedByPixelID[FileType]:
    """group events by pixel ID"""
    return GroupedByPixelID[FileType](da.group(ids))


def get_time_binned(
    da: GroupedByPixelID[FileType], time_bin_step: TimeBinStep
) -> TimeBinned[FileType]:
    """histogram events by time"""
    return TimeBinned[FileType](da.hist(t=time_bin_step))


providers = (
    get_intervals_mcstas,
    get_intervals,
    get_ids,
    get_grouped_by_pixel_id,
    get_time_binned,
)
