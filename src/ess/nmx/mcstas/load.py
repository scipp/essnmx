# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import re
from collections.abc import Generator

import scipp as sc
import scippnexus as snx

from ..types import (
    CrystalRotation,
    DetectorBankPrefix,
    DetectorIndex,
    DetectorName,
    FilePath,
    NMXRawData,
    ProtonCharge,
    RawEventData,
)
from .xml import McStasInstrument, read_mcstas_geometry_xml


def detector_name_from_index(index: DetectorIndex) -> DetectorName:
    return f'nD_Mantid_{getattr(index, "value", index)}'


def load_event_data_bank_name(
    detector_name: DetectorName, file_path: FilePath
) -> DetectorBankPrefix:
    '''Finds the filename associated with a detector'''
    with snx.File(file_path) as file:
        description = file['entry1/instrument/description'][()]
        for bank_name, det_names in bank_names_to_detector_names(description).items():
            if detector_name in det_names:
                return bank_name.partition('.')[0]


def _exclude_zero_events(data: sc.Variable) -> sc.Variable:
    """Exclude events with zero counts from the data."""
    if (data.values[0] == 0).all():
        data = data["event", 1:]
    else:
        data = data
    return data


def load_raw_event_data(
    file_path: FilePath,
    bank_prefix: DetectorBankPrefix,
    detector_name: DetectorName,
    instrument: McStasInstrument,
) -> RawEventData:
    """Retrieve events from the nexus file."""
    coords = instrument.to_coords(detector_name)
    bank_name = f'{bank_prefix}_dat_list_p_x_y_n_id_t'
    with snx.File(file_path, 'r') as f:
        root = f["entry1/data"]
        (bank_name,) = (name for name in root.keys() if bank_name in name)
        data = root[bank_name]["events"][()].rename_dims({'dim_0': 'event'})
        # McStas can add an extra event line containing 0,0,0,0,0,0
        # This line should not be included so we skip it.
        data = _exclude_zero_events(data)
        return RawEventData(
            sc.DataArray(
                coords={
                    'id': sc.array(
                        dims=['event'],
                        values=data['dim_1', 4].values,
                        dtype='int64',
                        unit=None,
                    ),
                    't': sc.array(
                        dims=['event'], values=data['dim_1', 5].values, unit='s'
                    ),
                },
                data=sc.array(
                    dims=['event'], values=data['dim_1', 0].values, unit='counts'
                ),
            ).group(coords.pop('pixel_id'))
        )


def raw_event_data_chunks(
    file_path: FilePath,
    bank_prefix: DetectorBankPrefix,
    pixel_ids: sc.Variable,
    chunk_size: int,  # Number of rows to read at a time
) -> Generator[RawEventData, None, None]:
    """Chunk events from the nexus file.

    Parameters
    ----------
    file_path:
        Path to the nexus file
    bank_prefix:
        Prefix of the bank name
    coords:
        Coordinates to generate the data array with the events
    chunk_size:
        Number of rows to read at a time
    """
    bank_name = f'{bank_prefix}_dat_list_p_x_y_n_id_t'
    with snx.File(file_path, 'r') as f:
        root = f["entry1/data"]
        num_events = root[bank_name]["events"].shape[0]
        (bank_name,) = (name for name in root.keys() if bank_name in name)

    for start in range(0, num_events, chunk_size):
        with snx.File(file_path, 'r') as f:
            root = f["entry1/data"]
            data = root[bank_name]["events"][
                "dim_0", start : start + chunk_size
            ].rename_dims({'dim_0': 'event'})
        # McStas can add an extra event line containing 0,0,0,0,0,0
        # This line should not be included so we skip it.
        data = _exclude_zero_events(data)
        event_da = sc.DataArray(
            coords={
                'id': sc.array(
                    dims=['event'],
                    values=data['dim_1', 4].values,
                    dtype='int64',
                    unit=None,
                ),
                't': sc.array(dims=['event'], values=data['dim_1', 5].values, unit='s'),
            },
            data=sc.array(
                dims=['event'], values=data['dim_1', 0].values, unit='counts'
            ),
        )
        yield RawEventData(event_da.group(pixel_ids))


def load_crystal_rotation(
    file_path: FilePath, instrument: McStasInstrument
) -> CrystalRotation:
    """Retrieve crystal rotation from the file.

    Raises
    ------
    KeyError
        If the crystal rotation is not found in the file.

    """
    with snx.File(file_path, 'r') as file:
        param_keys = tuple(f"entry1/simulation/Param/XtalPhi{key}" for key in "XYZ")
        if not all(key in file for key in param_keys):
            raise KeyError(
                f"Crystal rotations [{', '.join(param_keys)}] not found in file."
            )
        return CrystalRotation(
            sc.vector(
                value=[file[param_key][...] for param_key in param_keys],
                unit=instrument.simulation_settings.angle_unit,
            )
        )


def proton_charge_from_event_data(da: RawEventData) -> ProtonCharge:
    """Make up the proton charge from the event data array.

    Proton charge is proportional to the number of neutrons,
    which is proportional to the number of events.
    The scale factor is manually chosen based on previous results
    to be convenient for data manipulation in the next steps.
    It is derived this way since
    the protons are not part of McStas simulation,
    and the number of neutrons is not included in the result.

    Parameters
    ----------
    event_da:
        The event data

    """
    # Arbitrary number to scale the proton charge
    return ProtonCharge(sc.scalar(1 / 10_000, unit=None) * da.bins.size().sum().data)


def bank_names_to_detector_names(description: str) -> dict[str, list[str]]:
    """Associates event data names with the names of the detectors
    where the events were detected"""

    detector_component_regex = (
        # Start of the detector component definition, contains the detector name.
        r'^COMPONENT (?P<detector_name>.*) = Monitor_nD\(\n'
        # Some uninteresting lines, we're looking for 'filename'.
        # Make sure no new component begins.
        r'(?:(?!COMPONENT)(?!filename)(?:.|\s))*'
        # The line that defines the filename of the file that stores the
        # events associated with the detector.
        r'(?:filename = \"(?P<bank_name>[^\"]*)\")?'
    )
    matches = re.finditer(detector_component_regex, description, re.MULTILINE)
    bank_names_to_detector_names = {}
    for m in matches:
        bank_names_to_detector_names.setdefault(
            # If filename was not set for the detector the filename for the
            # event data defaults to the name of the detector.
            m.group('bank_name') or m.group('detector_name'),
            [],
        ).append(m.group('detector_name'))
    return bank_names_to_detector_names


def load_mcstas(
    *,
    da: RawEventData,
    proton_charge: ProtonCharge,
    crystal_rotation: CrystalRotation,
    detector_name: DetectorName,
    instrument: McStasInstrument,
) -> NMXRawData:
    coords = instrument.to_coords(detector_name)
    coords.pop('pixel_id')
    return NMXRawData(
        sc.DataGroup(
            weights=da,
            proton_charge=proton_charge,
            crystal_rotation=crystal_rotation,
            name=sc.scalar(detector_name),
            **coords,
        )
    )


providers = (
    read_mcstas_geometry_xml,
    detector_name_from_index,
    load_event_data_bank_name,
    load_raw_event_data,
    proton_charge_from_event_data,
    load_crystal_rotation,
    load_mcstas,
)
