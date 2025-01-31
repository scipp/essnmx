# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import io
import pathlib
from functools import partial
from typing import Any

import h5py
import numpy as np
import scipp as sc


def _create_dataset_from_var(
    *,
    root_entry: h5py.Group,
    var: sc.Variable,
    name: str,
    long_name: str | None = None,
    compression: str | None = None,
    compression_opts: int | None = None,
    dtype: Any = None,
) -> h5py.Dataset:
    compression_options = {}
    if compression is not None:
        compression_options["compression"] = compression
    if compression_opts is not None:
        compression_options["compression_opts"] = compression_opts

    dataset = root_entry.create_dataset(
        name,
        data=var.values if dtype is None else var.values.astype(dtype, copy=False),
        **compression_options,
    )
    dataset.attrs["units"] = str(var.unit)
    if long_name is not None:
        dataset.attrs["long_name"] = long_name
    return dataset


_create_compressed_dataset = partial(
    _create_dataset_from_var,
    compression="gzip",
    compression_opts=4,
)


def _create_root_data_entry(file_obj: h5py.File) -> h5py.Group:
    nx_entry = file_obj.create_group("NMX_data")
    nx_entry.attrs["NX_class"] = "NXentry"
    nx_entry.attrs["default"] = "data"
    nx_entry.attrs["name"] = "NMX"
    nx_entry["name"] = "NMX"
    nx_entry["definition"] = "TOFRAW"
    return nx_entry


def _create_sample_group(data: sc.DataGroup, nx_entry: h5py.Group) -> h5py.Group:
    nx_sample = nx_entry.create_group("NXsample")
    nx_sample["name"] = data['sample_name'].value
    _create_dataset_from_var(
        root_entry=nx_sample,
        var=data['crystal_rotation'],
        name='crystal_rotation',
        long_name='crystal rotation in Phi (XYZ)',
    )
    return nx_sample


def _create_instrument_group(data: sc.DataGroup, nx_entry: h5py.Group) -> h5py.Group:
    nx_instrument = nx_entry.create_group("NXinstrument")
    nx_instrument.create_dataset("proton_charge", data=data['proton_charge'].values)

    nx_detector_1 = nx_instrument.create_group("detector_1")
    # Detector counts
    _create_compressed_dataset(
        root_entry=nx_detector_1,
        name="counts",
        var=data['counts'],
    )
    # Time of arrival bin edges
    _create_dataset_from_var(
        root_entry=nx_detector_1,
        var=data['counts'].coords['t'],
        name="t_bin",
        long_name="t_bin TOF (ms)",
    )
    # Pixel IDs
    _create_compressed_dataset(
        root_entry=nx_detector_1,
        name="pixel_id",
        var=data['counts'].coords['id'],
        long_name="pixel ID",
    )
    return nx_instrument


def _create_detector_group(data: sc.DataGroup, nx_entry: h5py.Group) -> h5py.Group:
    nx_detector = nx_entry.create_group("NXdetector")
    # Position of the first pixel (lowest ID) in the detector
    _create_compressed_dataset(
        root_entry=nx_detector,
        name="origin",
        var=data['origin_position'],
    )
    # Fast axis, along where the pixel ID increases by 1
    _create_dataset_from_var(
        root_entry=nx_detector, var=data['fast_axis'], name="fast_axis"
    )
    # Slow axis, along where the pixel ID increases
    # by the number of pixels in the fast axis
    _create_dataset_from_var(
        root_entry=nx_detector, var=data['slow_axis'], name="slow_axis"
    )
    return nx_detector


def _create_source_group(data: sc.DataGroup, nx_entry: h5py.Group) -> h5py.Group:
    nx_source = nx_entry.create_group("NXsource")
    nx_source["name"] = "European Spallation Source"
    nx_source["short_name"] = "ESS"
    nx_source["type"] = "Spallation Neutron Source"
    nx_source["distance"] = sc.norm(data['source_position']).value
    nx_source["probe"] = "neutron"
    nx_source["target_material"] = "W"
    return nx_source


def export_as_nexus(
    data: sc.DataGroup, output_file: str | pathlib.Path | io.BytesIO
) -> None:
    """Export the reduced data to a NeXus file.

    Currently exporting step is not expected to be part of sciline pipelines.
    """
    import warnings

    warnings.warn(
        DeprecationWarning(
            "Exporting to custom NeXus format will be deprecated in the near future."
            "Please use ``export_as_nxlauetof`` instead."
        ),
        stacklevel=1,
    )
    with h5py.File(output_file, "w") as f:
        f.attrs["default"] = "NMX_data"
        nx_entry = _create_root_data_entry(f)
        _create_sample_group(data, nx_entry)
        _create_instrument_group(data, nx_entry)
        _create_detector_group(data, nx_entry)
        _create_source_group(data, nx_entry)


def _create_lauetof_data_entry(file_obj: h5py.File) -> h5py.Group:
    nx_entry = file_obj.create_group("entry")
    nx_entry.attrs["NX_class"] = "NXentry"
    return nx_entry


def _add_lauetof_definition(nx_entry: h5py.Group) -> None:
    nx_entry["definition"] = "NXlauetof"


def _add_lauetof_instrument(nx_entry: h5py.Group):
    nx_instrument = nx_entry.create_group("instrument")
    nx_instrument.attrs["NX_class"] = "NXinstrument"
    nx_instrument["name"] = "NMX"


def _add_lauetof_detector_group(dg: sc.DataGroup, nx_instrument: h5py.Group) -> None:
    nx_detector = nx_instrument.create_group(dg["name"].value)  # Detector name
    nx_detector.attrs["NX_class"] = "NXdetector"
    # Polar angle
    _create_dataset_from_var(
        name="polar_angle",
        root_entry=nx_detector,
        var=sc.scalar(0, unit='deg'),  # TODO: Add real data
    )
    # Azimuthal angle
    _create_dataset_from_var(
        name="azimuthal_angle",
        root_entry=nx_detector,
        var=sc.scalar(0, unit=''),  # TODO: Add real data
    )
    # Data - shape: [n_x_pixels, n_y_pixels, n_tof_bins]
    # The actual application definition defines it as integer,
    # but we keep the original data type for now
    num_x, num_y = dg["detector_shape"].values[0]  # Probably better way to do this...
    _create_dataset_from_var(
        name="data",
        root_entry=nx_detector,
        var=sc.fold(dg["counts"].data, dim='id', sizes={'x': num_x, 'y': num_y}),
        dtype=np.uint,
    )
    # x_pixel_size
    _create_dataset_from_var(
        name="x_pixel_size", root_entry=nx_detector, var=dg["x_pixel_size"]
    )
    # y_pixel_size
    _create_dataset_from_var(
        name="y_pixel_size", root_entry=nx_detector, var=dg["y_pixel_size"]
    )
    # distance
    _create_dataset_from_var(
        name="distance",
        root_entry=nx_detector,
        var=sc.scalar(0, unit='m'),  # TODO: Add real data
    )
    # time_of_flight - shape: [nTOF]
    _create_dataset_from_var(
        name="time_of_flight",
        root_entry=nx_detector,
        var=sc.midpoints(dg["counts"].coords['t']),
        # It should be actual time of flight values of each bin
        # Not sure if it should be median/mean of the bin or bin edges
    )


def _add_lauetof_sample_group(data: sc.DataGroup, nx_entry: h5py.Group) -> None:
    nx_sample = nx_entry.create_group("sample")
    nx_sample.attrs["NX_class"] = "NXsample"
    nx_sample["name"] = data['sample_name'].value
    _create_dataset_from_var(
        name='orientation_matrix',
        root_entry=nx_sample,
        var=sc.array(
            dims=['i', 'j'],
            values=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            unit="dimensionless",
        ),  # TODO: Add real data, the sample orientation matrix
    )
    _create_dataset_from_var(
        name='unit_cell',
        root_entry=nx_sample,
        var=sc.array(
            dims=['i'],
            values=[1.0, 1.0, 1.0, 90.0, 90.0, 90.0],
            unit="dimensionless",  # TODO: Add real data,
            # a, b, c, alpha, beta, gamma
        ),
    )


def _add_lauetof_monitor_group(data: sc.DataGroup, nx_entry: h5py.Group) -> None:
    nx_sample = nx_entry.create_group("control")
    nx_sample.attrs["NX_class"] = "NXmonitor"
    nx_sample["mode"] = "monitor"
    nx_sample["preset"] = 0.0  # Check if this is the correct value
    _create_dataset_from_var(
        name='data',
        root_entry=nx_sample,
        var=sc.array(
            dims=['tof'], values=[1, 1, 1], unit="counts"
        ),  # TODO: Add real data, bin values
    )
    _create_dataset_from_var(
        name='time_of_flight',
        root_entry=nx_sample,
        var=sc.array(
            dims=['tof'], values=[1, 1, 1], unit="s"
        ),  # TODO: Add real data, bin edges
    )


def export_panel_independent_data_as_nxlauetof(
    data: sc.DataGroup, output_file: str | pathlib.Path | io.BytesIO
) -> None:
    """Export panel independent data to a nxlauetof format.

    It also creates parents of panel dependent datasets/groups.
    Therefore panel dependent data should be added on the same file.
    """
    with h5py.File(output_file, "w") as f:
        f.attrs["NX_class"] = "NXlauetof"
        nx_entry = _create_lauetof_data_entry(f)
        _add_lauetof_definition(nx_entry)
        _add_lauetof_instrument(nx_entry)
        _add_lauetof_sample_group(data, nx_entry)
        # Placeholder for ``monitor`` group
        _add_lauetof_monitor_group(data, nx_entry)
        # Skipping ``name`` field


def export_panel_dependent_data_as_nxlauetof(
    dg: sc.DataGroup,
    output_file: str | pathlib.Path | io.BytesIO,
    append_mode: bool = True,
) -> None:
    mode = "r+" if append_mode else "w"
    with h5py.File(output_file, mode) as f:
        nx_instrument: h5py.Group = f["entry/instrument"]
        _add_lauetof_detector_group(dg, nx_instrument)


def export_as_nxlauetof(
    dg: sc.DataGroup, *dgs: sc.DataGroup, output_file: str | pathlib.Path | io.BytesIO
) -> None:
    """Export the reduced data into a nxlauetof format.

    Exporting step is not expected to be part of sciline pipelines.
    """

    export_panel_independent_data_as_nxlauetof(dg, output_file)
    for single_dg in [dg, *dgs]:
        export_panel_dependent_data_as_nxlauetof(single_dg, output_file=output_file)
