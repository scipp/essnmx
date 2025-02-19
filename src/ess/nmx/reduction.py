# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import NewType

import scipp as sc

from .mcstas.xml import McStasInstrument
from .types import (
    CrystalRotation,
    DetectorName,
    NMXHistogram,
    ProtonCharge,
    RawEventData,
    RawHistogram,
    TimeBinSteps,
)

NMXReducedData = NewType("NMXReducedData", sc.DataGroup)


def bin_time_of_arrival(da: RawEventData, time_bin_step: TimeBinSteps) -> RawHistogram:
    """Bin time of arrival data into ``time_bin_step`` bins."""
    return RawHistogram(da.hist(t=time_bin_step))


def format_nmx_reduced_data(
    counts: NMXHistogram,
    detector_name: DetectorName,
    instrument: McStasInstrument,
    proton_charge: ProtonCharge,
    crystal_rotation: CrystalRotation,
) -> NMXReducedData:
    """Bin time of arrival data into ``time_bin_step`` bins."""
    new_coords = instrument.to_coords(detector_name)
    new_coords.pop('pixel_id')
    return NMXReducedData(
        sc.DataGroup(
            counts=counts,
            proton_charge=proton_charge,
            crystal_rotation=crystal_rotation,
            name=sc.scalar(detector_name),
            **new_coords,
        )
    )


def _concat_or_same(
    obj: list[sc.Variable | sc.DataArray], dim: str
) -> sc.Variable | sc.DataArray:
    first = obj[0]
    # instrument.to_coords in bin_time_of_arrival adds a panel coord to some fields,
    # even if it has only length 1. If this is the case we concat, even if identical.
    # Maybe McStasInstrument.to_coords should be changed to only handle a single
    # panel, and not perform concatenation?
    if all(dim not in o.dims and sc.identical(first, o) for o in obj):
        return first
    return sc.concat(obj, dim)


def merge_panels(*panel: NMXReducedData) -> NMXReducedData:
    """Merge a list of panels by concatenating along the 'panel' dimension."""
    keys = panel[0].keys()
    if not all(p.keys() == keys for p in panel):
        raise ValueError("All panels must have the same keys.")
    return NMXReducedData(
        sc.DataGroup(
            {key: _concat_or_same([p[key] for p in panel], 'panel') for key in keys}
        )
    )
