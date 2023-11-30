from dataclasses import dataclass
from typing import NewType

import sciline as sl
import scipp as sc

from .loader import FileType, FileTypeMcStas, FileTypeNMX

TimeBinStep = NewType("TimeBinStep", int)
DefaultTimeBinStep = TimeBinStep(1)
Distance = NewType("Distance", sc.Variable)
Vector3D = NewType("Vector3D", sc.Variable)
RotationMatirx = NewType("RotationMatirx", sc.Variable)


@dataclass
class PixelIDIntervals(sl.domain.Generic[FileType]):
    """Pixel IDs for each detector"""

    interval_1: tuple
    interval_2: tuple
    interval_3: tuple


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


def get_intervals() -> PixelIDIntervals[FileTypeNMX]:
    """Pixel ID intervals for each detector"""
    return PixelIDIntervals[FileTypeNMX](
        (1, 1638401), (1638401, 3276802), (3276802, 4915203)
    )


def get_intervals_mcstas() -> PixelIDIntervals[FileTypeMcStas]:
    """Pixel IDs intervals for each detector"""
    return PixelIDIntervals[FileTypeMcStas](
        (1, 1638401), (2000001, 3638401), (4000001, 5638401)
    )


def get_ids(pixel_id_intervals: PixelIDIntervals[FileType]) -> PixelIDs[FileType]:
    """pixel IDs for each detector"""
    intervals = [
        pixel_id_intervals.interval_1,
        pixel_id_intervals.interval_2,
        pixel_id_intervals.interval_3,
    ]
    ids = [sc.arange('id', start, stop, unit=None) for start, stop in intervals]
    return PixelIDs[FileType](sc.concat(ids, 'id'))
