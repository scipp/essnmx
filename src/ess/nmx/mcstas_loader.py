# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from typing import Iterable, NewType, Optional

import numpy as np
import scipp as sc
import scippnexus as snx
from numpy.typing import NDArray

PixelIDs = NewType("PixelIDs", sc.Variable)
InputFilepath = NewType("InputFilepath", str)
NMXData = NewType("NMXData", sc.DataArray)

# McStas Configurations
MaximumProbability = NewType("MaximumProbability", int)
DefaultMaximumProbability = MaximumProbability(100_000)

DetectorGeometry = NewType("DetectorGeometry", sc.Dataset)


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


def _select_by_type_prefix(components: list, prefix: str) -> list:
    """Select components by type prefix."""
    return [comp for comp in components if comp.attrib['type'].startswith(prefix)]


def _check_if_only_one(xml_items: list, name: str) -> None:
    """Check if there is only one element with ``name``."""
    if len(xml_items) > 1:
        raise ValueError(f"Multiple {name}s found.")
    elif len(xml_items) == 0:
        raise ValueError(f"No {name} found.")


def _retrieve_attribs(component, *args: str) -> list[float]:
    """Retrieve ``args`` from xml."""

    return [float(component.attrib[key]) for key in args]


def _retrieve_xyz(component) -> sc.Variable:
    """Retrieve x, y, z position from xml."""
    location = component.find('location')
    if location is None:
        raise ValueError("No location found in component ", component.find('name'))

    return sc.vector(_retrieve_attribs(location, 'x', 'y', 'z'))


def _retrieve_rotation_axis_angle(component) -> list[float]:
    """Retrieve rotation angle(theta), x, y, z axes from xml."""
    location = component.find('location')
    if location is None:
        raise ValueError("No location found in component ", component.find('name'))

    return _retrieve_attribs(location, 'rot', 'axis-x', 'axis-y', 'axis-z')


def axis_angle_to_quaternion(
    x: float, y: float, z: float, theta: sc.Variable
) -> NDArray:
    """Convert axis-angle to queternions, [x, y, z, w].

    Parameters
    ----------
    x:
        X component of axis of rotation.
    y:
        Y component of axis of rotation.
    z:
        Z component of axis of rotation.
    theta:
        Angle of rotation, with unit of ``rad`` or ``deg``.

    Returns
    -------
    :
        A list of (normalized) queternions, [x, y, z, w].

    Notes
    -----
    Axis of rotation (x, y, z) does not need to be normalized,
    but it returns a unit quaternion (x, y, z, w).

    """

    w: sc.Variable = sc.cos(theta.to(unit='rad') / 2)
    xyz: sc.Variable = -sc.sin(theta.to(unit='rad') / 2) * sc.vector([x, y, z])
    q = np.array([*xyz.values, w.value])
    return q / np.linalg.norm(q)


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> sc.Variable:
    """Convert quaternion to rotation matrix.

    Parameters
    ----------
    x:
        x(a) component of quaternion.
    y:
        y(b) component of quaternion.
    z:
        z(c) component of quaternion.
    w:
        w component of quaternion.

    Returns
    -------
    :
        A 3X3 rotation matrix (3 vectors).

    """
    from scipy.spatial.transform import Rotation

    return sc.spatial.rotations_from_rotvecs(
        rotation_vectors=sc.vector(
            Rotation.from_quat([x, y, z, w]).as_rotvec(),
            unit='rad',
        )
    )


def flip_vector(x: sc.Variable) -> sc.Variable:
    """Flip a vector.

    It is just multiplying -1 to each component,
    but this function shows the purpose of multiplying -1.
    It will be easier to find where the vectors are flipped in the process.

    """
    return x * -1


def _retrieve_rotation_matrix_from_detector(detector) -> sc.Variable:
    theta, x, y, z = _retrieve_rotation_axis_angle(detector)
    q = axis_angle_to_quaternion(x, y, z, sc.scalar(theta, unit='deg'))
    return quaternion_to_matrix(*q)


def load_mcstas_geometry(file_path: InputFilepath) -> DetectorGeometry:
    """Retrieve geometry parameters from mcstas file"""
    import numpy as np
    import scipp as sc

    xml = _read_mcstas_geometry_xml(file_path)
    components = _collect_components(xml)
    detectors = _select_by_type_prefix(components, 'MonNDtype')

    sources = _select_by_type_prefix(components, 'sourceMantid-type')
    samples = _select_by_type_prefix(components, 'sampleMantid-type')
    _check_if_only_one(sources, 'source')
    _check_if_only_one(samples, 'sample')

    sample_pos = flip_vector(_retrieve_xyz(samples.pop()))  # Why flipping?
    # source_pos = _retrieve_xyz(sources.pop())  # Why not flipping?

    fast_axis = flip_vector(sc.vector([1.0, 0.0, 0.0]))  # Why flipping? Hardcoded?
    slow_axis = flip_vector(sc.vector([0.0, 1.0, 0.0]))  # Why flipping? Hardcoded?

    detectors_to_samples = [
        sc.vector(
            np.round((flip_vector(_retrieve_xyz(detector)) - sample_pos).values, 2)
        )
        # Why flipping? Why rounding?
        for detector in detectors
    ]
    rotation_matrices = [
        _retrieve_rotation_matrix_from_detector(detector) for detector in detectors
    ]
    slow_axes = [
        rotation_matrix * slow_axis  # Why not rounding?
        for rotation_matrix in rotation_matrices
    ]
    fast_axes = [
        sc.vector(np.round((rotation_matrix * fast_axis).values, 2))  # Why rounding?
        for rotation_matrix in rotation_matrices
    ]

    return DetectorGeometry(
        sc.Dataset(
            coords={
                'id': sc.arange('detector', len(detectors)),
            },
            data={
                'origen': sc.concat(detectors_to_samples, 'detector'),
                'slow_axis': sc.concat(slow_axes, 'detector'),
                'fast_axis': sc.concat(fast_axes, 'detector'),
            },
        )
    )


def _retrieve_detector_geometries(tree):
    """Retrieve detector geometries from mcstas file"""

    components = [branch for branch in tree if branch.tag == 'component']
    detectors = _select_by_type_prefix(components, 'MonNDtype')
    type_list = [branch for branch in tree if branch.tag == 'type']

    def _find_type_desc(elem, types: list):
        for type_ in types:
            if type_.attrib['name'] == elem.attrib['type']:
                return type_

        raise ValueError(f"Can not find type for component {elem.attrib['name']}.")

    detector_type_map = {
        det.attrib['name']: _find_type_desc(det, type_list) for det in detectors
    }
    detector_type_type_map = {
        det_type.attrib['name']: _find_type_desc(det_type, type_list)
        for det_type in detector_type_map.values()
    }
    return detector_type_map, detector_type_type_map


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
    from defusedxml.ElementTree import fromstring

    tree = fromstring(_read_mcstas_geometry_xml(file_path))
    _retrieve_detector_geometries(tree)
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
