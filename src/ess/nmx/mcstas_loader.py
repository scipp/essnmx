# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from typing import Iterable, NewType, Optional

import scipp as sc
import scippnexus as snx
from defusedxml import ElementTree as ET

PixelIDs = NewType("PixelIDs", sc.Variable)
InputFilepath = NewType("InputFilepath", str)
NMXData = NewType("NMXData", sc.DataArray)

# McStas Configurations
MaximumProbability = NewType("MaximumProbability", int)
DefaultMaximumProbability = MaximumProbability(100_000)


@dataclass
class McStasGeometry:
    ds_l: list
    fast_l: list
    slow_l: list
    sample_pos: list
    source_pos: list


def _retrieve_event_list_name(keys: Iterable[str]) -> str:
    prefix = "bank01_events_dat_list"

    # (weight, x, y, n, pixel id, time of arrival)
    mandatory_fields = 'p_x_y_n_id_t'

    for key in keys:
        if key.startswith(prefix) and mandatory_fields in key:
            return key

    raise ValueError("Can not find event list name.")


def _copy_partial_var(
    var: sc.Variable, idx: int, unit: Optional[str] = None, dtype: Optional[str] = None
) -> sc.Variable:
    """Retrieve a property from a variable."""
    var = var['dim_1', idx].astype(dtype or var.dtype, copy=True)
    if unit:
        var.unit = sc.Unit(unit)
    return var


def _get_mcstas_pixel_ids() -> PixelIDs:
    """pixel IDs for each detector"""
    intervals = [(1, 1638401), (2000001, 3638401), (4000001, 5638401)]
    ids = [sc.arange('id', start, stop, unit=None) for start, stop in intervals]
    return PixelIDs(sc.concat(ids, 'id'))


def _read_mcstas_geometry_xml(file_path: InputFilepath) -> bytes:
    """Retrieve geometry parameters from mcstas file"""
    import h5py

    instrument_xml_path = 'entry1/instrument/instrument_xml/data'
    with h5py.File(file_path) as file:
        return file[instrument_xml_path][...][0]


def _collect_components(xml_str: bytes) -> list:
    """Collect components from xml."""
    from defusedxml.ElementTree import fromstring

    tree = fromstring(xml_str)
    return [branch for branch in tree if branch.tag == 'component']


def _select_by_type_prefix(
    components: list[ET.Element], prefix: str
) -> list[ET.Element]:
    """Select components by type prefix."""
    return [comp for comp in components if comp.attrib['type'].startswith(prefix)]


def _check_if_only_one(xml_items: list[ET.Element], name: str) -> None:
    """Check if there is only one element with ``name``."""
    if len(xml_items) > 1:
        raise ValueError(f"Multiple {name}s found.")
    elif len(xml_items) == 0:
        raise ValueError(f"No {name} found.")


def _retrieve_attribs(component: ET.Element, *args: str) -> list[float]:
    """Retrieve ``args`` from xml."""

    return [float(component.attrib[key]) for key in args]


def _retrieve_xyz(component: ET.Element) -> list[float]:
    """Retrieve xyz from xml."""
    location = component.find('location')
    if location is None:
        raise ValueError("No location found in component ", component.find('name'))

    return _retrieve_attribs(location, 'x', 'y', 'z')


def _retrieve_rxyz(component: ET.Element) -> list[float]:
    """Retrieve rotation angle, x, y, z axes from xml."""
    location = component.find('location')
    if location is None:
        raise ValueError("No location found in component ", component.find('name'))

    return _retrieve_attribs(location, 'rot', 'axis-x', 'axis-y', 'axis-z')


def load_mcstas_geometry(file_path: InputFilepath) -> McStasGeometry:
    """Retrieve geometry parameters from mcstas file"""
    import numpy as np

    xml = _read_mcstas_geometry_xml(file_path)
    components = _collect_components(xml)
    detectors = _select_by_type_prefix(components, 'MonNDtype')

    sources = _select_by_type_prefix(components, 'sourceMantid-type')
    samples = _select_by_type_prefix(components, 'sampleMantid-type')
    _check_if_only_one(sources, 'source')
    _check_if_only_one(samples, 'sample')
    sample_pos = np.array(_retrieve_xyz(samples.pop())) * [-1, -1, -1]
    source_pos = np.array(_retrieve_xyz(sources.pop()))

    fast_l = []
    slow_l = []
    vec_f = np.array([-1.0, 0.0, -0.0])
    vec_s = np.array([0.0, -1.0, 0.0])
    ds_l = []

    for detector in detectors:
        det_pos = np.array(_retrieve_xyz(detector)) * [-1, -1, -1]
        rel_pos = np.round(det_pos - sample_pos, 2)
        # rel_pos = np.round(twoP_to_vec(sample_pos, det_pos), 2)
        ds_l.append(rel_pos)

        rot_pos = np.array(_retrieve_rxyz(detector))
        theta = np.radians(-rot_pos[0])
        v = -(rot_pos[1:] / np.linalg.norm(rot_pos[1:])) * np.sin(theta / 2)
        q = np.array([-np.cos(theta / 2)] + v.tolist())
        w, x, y, z = q
        rot_matrix = np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
            ]
        )
        fast_l.append(np.round(np.dot(rot_matrix, vec_f), 2))
        slow_l.append(np.dot(rot_matrix, vec_s))

    return ds_l, fast_l, slow_l, sample_pos, source_pos


def load_mcstas_nexus(
    file_path: InputFilepath,
    max_probability: Optional[MaximumProbability] = None,
) -> NMXData:
    """Load McStas simulation result from h5(nexus) file.

    Parameters
    ----------
    file_path:
        File name to load.

    max_probability:
        The maximum probability to scale the weights.

    """

    probability = max_probability or DefaultMaximumProbability

    with snx.File(file_path) as file:
        bank_name = _retrieve_event_list_name(file["entry1/data"].keys())
        var: sc.Variable
        var = file["entry1/data/" + bank_name]["events"][()].rename_dims(
            {'dim_0': 'event'}
        )

        weights = _copy_partial_var(var, idx=0, unit='counts')  # p
        id_list = _copy_partial_var(var, idx=4, dtype='int64')  # id
        t_list = _copy_partial_var(var, idx=5, unit='s')  # t

        weights = (probability / weights.max()) * weights

        loaded = sc.DataArray(data=weights, coords={'t': t_list, 'id': id_list})
        grouped = loaded.group(_get_mcstas_pixel_ids())

        return NMXData(grouped.fold(dim='id', sizes={'panel': 3, 'id': -1}))
