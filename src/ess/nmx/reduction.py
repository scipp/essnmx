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


# all this is McStas specific, move to mcstas loader module


def get_intervals_mcstas() -> PixelIDIntervals:
    """Pixel IDs intervals for each detector"""

    return PixelIDIntervals(  # Check McStas pixel number conuting
        (1, 1638401), (2000001, 3638401), (4000001, 5638401)
    )


def get_ids(pixel_id_intervals: Optional[PixelIDIntervals] = None) -> PixelIDs:
    """pixel IDs for each detector"""
    if pixel_id_intervals is None:
        # TODO check with Justin why panels have different pixel counts,
        # should be the same?
        intervals = [(1, 1638401), (1638401, 3276802), (3276802, 4915203)]
    else:
        intervals = [
            pixel_id_intervals.interval_1,
            pixel_id_intervals.interval_2,
            pixel_id_intervals.interval_3,
        ]

    ids = [sc.arange('pixel', start, stop) for start, stop in intervals]
    # TODO We'd like panel as extra dim, but works only if all panels have same
    # number of pixels.
    return PixelIDs(sc.concat(ids, 'panel'))


def get_grouped_by_pixel_id(da: Events, ids: PixelIDs) -> GroupedByPixelID:
    """group events by pixel ID"""
    return GroupedByPixelID(da.group(ids))


def get_time_binned(da: GroupedByPixelID, time_bin_step: TimeBinStep) -> TimeBinned:
    """histogram events by time"""
    # TODO Do we need time binning? Should avoid this if possible for the actual
    # reduction, e.g., convert to wavelength first, then bin (or hist) before saving
    return TimeBinned(da.hist(t=time_bin_step))


reduction_providers = (
    get_ids,
    get_grouped_by_pixel_id,
    get_time_binned,
)
