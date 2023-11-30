from typing import NewType

import sciline as sl
import scipp as sc

from .loader import FileType, FileTypeMcStas, FileTypeNMX

TimeBinStep = NewType("TimeBinStep", int)
DefaultTimeBinStep = TimeBinStep(1)
Distance = NewType("Distance", sc.Variable)
Vector3D = NewType("Vector3D", sc.Variable)
RotationMatirx = NewType("RotationMatirx", sc.Variable)


class PixelIDs(sl.Scope[sc.Variable, FileType]):
    """Pixel IDs for each detector"""

    ...


def calculate_distance(point_a: Vector3D, point_b: Vector3D) -> Distance:
    """
    Calculate the distance between two points.
    """
    diff = point_b - point_a
    return Distance(sc.sqrt(sc.dot(diff, diff)))


RotationAngle = NewType("RotationAngle", sc.Variable)


def rotation_matrix(axis: Vector3D, theta: RotationAngle) -> RotationMatirx:
    """
    Return the rotation matrix associated with counter-clockwise rotation about
    the given axis by theta radians.

    # TODO: Add reference.
    """

    a = sc.cos(theta / 2.0)
    b, c, d = -(axis / sc.dot(axis, axis)) * sc.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

    return RotationMatirx(
        sc.vectors(
            dims=['row', 'col'],
            values=[
                sc.vector([aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)]),
                sc.vector([2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)]),
                sc.vector([2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]),
            ],
        )
    )


def _get_ids(*intervals) -> PixelIDs:
    """pixel IDs for each detector"""
    ids = [sc.arange('id', start, stop, unit=None) for start, stop in intervals]
    return PixelIDs(sc.concat(ids, 'id'))


def get_ids() -> PixelIDs[FileTypeNMX]:
    """pixel IDs for each detector"""
    # TODO: Why are they different...?
    id1_interval = (1, 1638401)
    id2_interval = (1638401, 3276802)
    id3_interval = (3276802, 4915203)

    return _get_ids(id1_interval, id2_interval, id3_interval)


def get_ids_mcstas() -> PixelIDs[FileTypeMcStas]:
    """pixel IDs for each detector"""
    id1_interval = (1, 1638401)
    id2_interval = (2000001, 3638401)
    id3_interval = (4000001, 5638401)

    return _get_ids(id1_interval, id2_interval, id3_interval)
