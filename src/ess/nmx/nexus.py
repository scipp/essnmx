# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import io
import pathlib
import warnings
from functools import partial
from typing import Any

import h5py
import numpy as np
import scipp as sc
import scippnexus as snx

from .types import NMXDetectorMetadata, NMXExperimentMetadata, NMXReducedDataGroup

from .types import NMXDetectorMetadata, NMXExperimentMetadata, NMXReducedDataGroup


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
    if var.unit is not None:
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
    warnings.warn(
        DeprecationWarning(
            "Exporting to custom NeXus format will be deprecated in the near future."
            "Please use ``export_as_nxlauetof`` instead."
        ),
        stacklevel=2,
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


def _add_lauetof_instrument(nx_entry: h5py.Group) -> h5py.Group:
    nx_instrument = nx_entry.create_group("instrument")
    nx_instrument.attrs["NX_class"] = "NXinstrument"
    nx_instrument["name"] = "NMX"
    return nx_instrument


def _add_lauetof_detector_group(dg: sc.DataGroup, nx_instrument: h5py.Group) -> None:
    nx_detector = nx_instrument.create_group(dg["detector_name"].value)  # Detector name
    nx_detector.attrs["NX_class"] = "NXdetector"
    _create_dataset_from_var(
        name="polar_angle",
        root_entry=nx_detector,
        var=sc.scalar(0, unit='deg'),  # TODO: Add real data
    )
    _create_dataset_from_var(
        name="azimuthal_angle",
        root_entry=nx_detector,
        var=sc.scalar(0, unit='deg'),  # TODO: Add real data
    )
    _create_dataset_from_var(
        name="x_pixel_size", root_entry=nx_detector, var=dg["x_pixel_size"]
    )
    _create_dataset_from_var(
        name="y_pixel_size", root_entry=nx_detector, var=dg["y_pixel_size"]
    )
    _create_dataset_from_var(
        name="distance",
        root_entry=nx_detector,
        var=sc.scalar(0, unit='m'),  # TODO: Add real data
    )
    # Legacy geometry information until we have a better way to store it
    _create_dataset_from_var(
        name="origin", root_entry=nx_detector, var=dg['origin_position']
    )
    # Fast axis, along where the pixel ID increases by 1
    _create_dataset_from_var(
        root_entry=nx_detector, var=dg['fast_axis'], name="fast_axis"
    )
    # Slow axis, along where the pixel ID increases
    # by the number of pixels in the fast axis
    _create_dataset_from_var(
        root_entry=nx_detector, var=dg['slow_axis'], name="slow_axis"
    )


def _add_lauetof_sample_group(dg: NMXExperimentMetadata, nx_entry: h5py.Group) -> None:
    nx_sample = nx_entry.create_group("sample")
    nx_sample.attrs["NX_class"] = "NXsample"
    nx_sample["name"] = (
        dg['sample_name'].value
        if isinstance(dg['sample_name'], sc.Variable)
        else dg['sample_name']
    )
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
    nx_monitor = nx_entry.create_group("control")
    nx_monitor.attrs["NX_class"] = "NXmonitor"
    nx_monitor["mode"] = "monitor"
    nx_monitor["preset"] = 0.0  # Check if this is the correct value
    data_dset = _create_dataset_from_var(
        name='data',
        root_entry=nx_monitor,
        var=sc.array(
            dims=['tof'], values=[1, 1, 1], unit="counts"
        ),  # TODO: Add real data, bin values
    )
    data_dset.attrs["signal"] = 1
    data_dset.attrs["primary"] = 1
    _create_dataset_from_var(
        name='time_of_flight',
        root_entry=nx_monitor,
        var=sc.array(
            dims=['tof'], values=[1, 1, 1], unit="s"
        ),  # TODO: Add real data, bin edges
    )


def _add_arbitrary_metadata(
    nx_entry: h5py.Group, **arbitrary_metadata: sc.Variable
) -> None:
    if not arbitrary_metadata:
        return

    metadata_group = nx_entry.create_group("metadata")
    for key, value in arbitrary_metadata.items():
        if not isinstance(value, sc.Variable):
            import warnings

            msg = f"Skipping metadata key '{key}' as it is not a scipp.Variable."
            warnings.warn(UserWarning(msg), stacklevel=2)
            continue
        else:
            _create_dataset_from_var(
                name=key,
                root_entry=metadata_group,
                var=value,
            )


def export_metadata_as_nxlauetof(
    *detector_metadatas: NMXDetectorMetadata,
    experiment_metadata: NMXExperimentMetadata,
    output_file: str | pathlib.Path | io.BytesIO,
    **arbitrary_metadata: sc.Variable,
) -> None:
    """Export the metadata to a NeXus file with the LAUE_TOF application definition.

    ``Metadata`` in this context refers to the information
    that is not part of the reduced detector counts itself,
    but is necessary for the interpretation of the reduced data.
    Since NMX can have arbitrary number of detectors,
    this function can take multiple detector metadata objects.

    Parameters
    ----------
    detector_metadatas:
        Detector metadata objects.
    experiment_metadata:
        Experiment metadata object.
    output_file:
        Output file path.
    arbitrary_metadata:
        Arbitrary metadata that does not fit into the existing metadata objects.

    """
    with h5py.File(output_file, "w") as f:
        f.attrs["NX_class"] = "NXlauetof"
        nx_entry = _create_lauetof_data_entry(f)
        _add_lauetof_definition(nx_entry)
        nx_instrument = _add_lauetof_instrument(nx_entry)
        _add_lauetof_sample_group(experiment_metadata, nx_entry)
        # Placeholder for ``monitor`` group
        _add_lauetof_monitor_group(experiment_metadata, nx_entry)
        # Skipping ``NXdata``(name) field with data link
        # Add detector group metadata
        for detector_metadata in detector_metadatas:
            _add_lauetof_detector_group(detector_metadata, nx_instrument)
        # Add arbitrary metadata
        _add_arbitrary_metadata(nx_entry, **arbitrary_metadata)


def _validate_existing_metadata(
    dg: NMXReducedDataGroup,
    detector_group: snx.Group,
    sample_group: snx.Group,
    safety_checks: bool = True,
) -> None:
    flag = True
    # check pixel size
    flag = flag and sc.identical(dg["x_pixel_size"], detector_group["x_pixel_size"])
    flag = flag and sc.identical(dg["y_pixel_size"], detector_group["y_pixel_size"])
    # check sample name
    flag = flag and dg["sample_name"].value == sample_group["name"]

    if not flag and safety_checks:
        raise ValueError(
            f"Metadata for detector '{dg['detector_name'].value}' in the file "
            "does not match the provided data."
        )
    elif not flag and not safety_checks:
        warnings.warn(
            UserWarning(
                "Metadata for detector in the file does not match the provided data."
                "This may lead to unexpected results."
                "However, the operation will proceed as requested "
                "since safety checks are disabled."
            ),
            stacklevel=2,
        )


def export_reduced_data_as_nxlauetof(
    dg: NMXReducedDataGroup,
    output_file: str | pathlib.Path | io.BytesIO,
    append_mode: bool = True,
    safety_checks: bool = True,
) -> None:
    """Export the reduced data to a NeXus file with the LAUE_TOF application definition.

    Even though this function only exports
    reduced data(detector counts and its coordinates),
    the input should contain all the necessary metadata
    for minimum sanity check.

    Parameters
    ----------
    dg:
        Reduced data and metadata.
    output_file:
        Output file path.
    append_mode:
        If ``True``, the file is opened in append mode.
        If ``False``, the file is opened in None-append mode.
        > None-append mode is not supported for now.
        > Only append mode is supported for now.

    """
    if not append_mode:
        raise NotImplementedError("Only append mode is supported for now.")
    detector_group_path = f"entry/instrument/{dg['detector_name'].value}"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        # Userwarning is expected here as histogram data is not yet saved.
        with snx.File(output_file, "r") as f:
            _validate_existing_metadata(
                dg=dg,
                detector_group=f[detector_group_path][()],
                sample_group=f["entry/sample"][()],
                safety_checks=safety_checks,
            )

    with h5py.File(output_file, "r+") as f:
        nx_detector: h5py.Group = f[detector_group_path]

    if not append_mode:
        raise NotImplementedError("Only append mode is supported for now.")

    with h5py.File(output_file, "r+") as f:
        nx_detector: h5py.Group = f[f"entry/instrument/{dg['detector_name'].value}"]
        # Data - shape: [n_x_pixels, n_y_pixels, n_tof_bins]
        # The actual application definition defines it as integer,
        # but we keep the original data type for now
        num_x, num_y = dg["detector_shape"].value  # Probably better way to do this
        data_dset = _create_dataset_from_var(
            name="data",
            root_entry=nx_detector,
            var=sc.fold(dg['counts'].data, dim='id', sizes={'x': num_x, 'y': num_y}),
            dtype=np.uint,
        )
        data_dset.attrs["signal"] = 1
        _create_dataset_from_var(
            name='time_of_flight',
            root_entry=nx_detector,
            var=sc.midpoints(dg['counts'].coords['t'], dim='t'),
        )
