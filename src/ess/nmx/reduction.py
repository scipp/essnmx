# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from typing import NewType, Optional

import scipp as sc

from .loader import Events

TimeBinStep = NewType("TimeBinStep", int)


@dataclass
class PixelIDIntervals:
    """Pixel IDs for each detector"""

    interval_1: tuple
    interval_2: tuple
    interval_3: tuple


PixelIDs = NewType("PixelIDs", sc.Variable)
GroupedByPixelID = NewType("GroupedByPixelID", sc.DataArray)
TimeBinned = NewType("TimeBinned", sc.DataArray)


def get_intervals_mcstas() -> PixelIDIntervals:
    """Pixel IDs intervals for each detector"""

    return PixelIDIntervals(  # Check McStas pixel number conuting
        (1, 1638401), (2000001, 3638401), (4000001, 5638401)
    )


def get_ids(pixel_id_intervals: Optional[PixelIDIntervals] = None) -> PixelIDs:
    """pixel IDs for each detector"""
    if pixel_id_intervals is None:
        intervals = [(1, 1638401), (1638401, 3276802), (3276802, 4915203)]
    else:
        intervals = [
            pixel_id_intervals.interval_1,
            pixel_id_intervals.interval_2,
            pixel_id_intervals.interval_3,
        ]

    ids = [sc.arange('id', start, stop) for start, stop in intervals]
    return PixelIDs(sc.concat(ids, 'id'))


def get_grouped_by_pixel_id(da: Events, ids: PixelIDs) -> GroupedByPixelID:
    """group events by pixel ID"""
    return GroupedByPixelID(da.group(ids))


def get_time_binned(da: GroupedByPixelID, time_bin_step: TimeBinStep) -> TimeBinned:
    """histogram events by time"""
    return TimeBinned(da.hist(t=time_bin_step))


reduction_providers = (
    get_ids,
    get_grouped_by_pixel_id,
    get_time_binned,
)
