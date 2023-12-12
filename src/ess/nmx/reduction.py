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


def get_ids() -> PixelIDs:
    """pixel IDs for each detector"""
    intervals = [(1, 1638401), (2000001, 3638401), (4000001, 5638401)]
    ids = [sc.arange('pixel', start, stop) for start, stop in intervals]
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
