# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import scipp as sc

from .mcstas.xml import McStasInstrument
from .types import (
    CrystalRotation,
    DetectorName,
    NMXReducedData,
    NMXReducedDataGroup,
    PixelIds,
    ProtonCharge,
    RawEventCounts,
    TimeBinSteps,
)


def reduce_raw_event_counts(
    da: RawEventCounts,
    pixel_ids: PixelIds,
    time_bin_step: TimeBinSteps,
) -> NMXReducedData:
    return NMXReducedData(da.group(pixel_ids).hist(t=time_bin_step))


def format_nmx_reduced_data(
    da: NMXReducedData,
    detector_name: DetectorName,
    instrument: McStasInstrument,
    proton_charge: ProtonCharge,
    crystal_rotation: CrystalRotation,
) -> NMXReducedDataGroup:
    """Bin time of arrival data into ``time_bin_step`` bins."""
    new_coords = instrument.to_coords(detector_name)

    return NMXReducedDataGroup(
        sc.DataGroup(
            counts=da,
            proton_charge=proton_charge,
            crystal_rotation=crystal_rotation,
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


def merge_panels(*panel: NMXReducedDataGroup) -> NMXReducedDataGroup:
    """Merge a list of panels by concatenating along the 'panel' dimension."""
    keys = panel[0].keys()
    if not all(p.keys() == keys for p in panel):
        raise ValueError("All panels must have the same keys.")
    return NMXReducedDataGroup(
        sc.DataGroup(
            {key: _concat_or_same([p[key] for p in panel], 'panel') for key in keys}
        )
    )
