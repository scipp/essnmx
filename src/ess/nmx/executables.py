# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import argparse
import logging
import pathlib
from collections.abc import Callable
from dataclasses import dataclass

import scipp as sc
import scippnexus as snx

from .nexus import (
    _compute_positions,
    _export_detector_metadata_as_nxlauetof,
    _export_reduced_data_as_nxlauetof,
    _export_static_metadata_as_nxlauetof,
)
from .streaming import _validate_chunk_size
from .types import NMXDetectorMetadata, NMXExperimentMetadata


def _retrieve_source_position(file: snx.File) -> sc.Variable:
    da = file['entry/instrument/source'][()]
    return _compute_positions(da, auto_fix_transformations=True)['position']


def _retrieve_sample_position(file: snx.File) -> sc.Variable:
    da = file['entry/sample'][()]
    return _compute_positions(da, auto_fix_transformations=True)['position']


def _decide_fast_axis(da: sc.DataArray) -> str:
    x_slice = da['x_pixel_offset', 0]
    y_slice = da['y_pixel_offset', 0]
    if (x_slice.max() < y_slice.max()).value:
        return 'y'
    elif x_slice.max() > y_slice.max().value:
        return 'x'
    else:
        raise ValueError(
            "Cannot decide fast axis based on pixel offsets. "
            "Please specify the fast axis explicitly."
        )


def _decide_step(offsets: sc.Variable) -> sc.Variable:
    """Decide the step size based on the offsets assuming at least 2 values."""
    sorted_offsets = sc.sort(offsets, key=offsets.dim, order='ascending')
    return sorted_offsets[1] - sorted_offsets[0]


@dataclass
class DetectorDesc:
    """Detector information extracted from McStas instrument xml description."""

    name: str
    id_start: int  # 'idstart'
    num_x: int  # 'xpixels'
    num_y: int  # 'ypixels'
    step_x: sc.Variable  # 'xstep'
    step_y: sc.Variable  # 'ystep'
    start_x: float  # 'xstart'
    start_y: float  # 'ystart'
    position: sc.Variable  # <location> 'x', 'y', 'z'
    # Calculated fields
    rotation_matrix: sc.Variable
    fast_axis_name: str
    slow_axis_name: str
    fast_axis: sc.Variable
    slow_axis: sc.Variable


def build_detector_desc(name: str, dg: sc.DataGroup) -> DetectorDesc:
    da: sc.DataArray = dg['data']
    fast_axis = _decide_fast_axis(da)
    transformation_matrix = dg['transform_matrix']
    t_unit = transformation_matrix.unit
    fast_axis_vector = (
        sc.vector([1, 0, 0], unit=t_unit)
        if fast_axis == 'x'
        else sc.vector([0, 1, 0], unit=t_unit)
    )
    slow_axis_vector = (
        sc.vector([0, 1, 0], unit=t_unit)
        if fast_axis == 'x'
        else sc.vector([1, 0, 0], unit=t_unit)
    )
    return DetectorDesc(
        name=name,
        id_start=da.coords['detector_number'].min().value,
        num_x=da.sizes['x_pixel_offset'],
        num_y=da.sizes['y_pixel_offset'],
        start_x=da.coords['x_pixel_offset'].min().value,
        start_y=da.coords['y_pixel_offset'].min().value,
        position=dg['position'],
        rotation_matrix=dg['transform_matrix'],
        fast_axis_name=fast_axis,
        slow_axis_name='x' if fast_axis == 'y' else 'y',
        fast_axis=fast_axis_vector,
        slow_axis=slow_axis_vector,
        step_x=_decide_step(da.coords['x_pixel_offset']),
        step_y=_decide_step(da.coords['y_pixel_offset']),
    )


def calculate_number_of_chunks(detector_gr: snx.Group, *, chunk_size: int = 0) -> int:
    _validate_chunk_size(chunk_size)
    event_time_zero_size = detector_gr.sizes['event_time_zero']
    if chunk_size == -1:
        return 1  # Read all at once
    else:
        return event_time_zero_size // chunk_size + int(
            event_time_zero_size % chunk_size != 0
        )


def build_toa_bin_edges(
    toa_bin_edges: sc.Variable | int = 250,
) -> sc.Variable:
    if isinstance(toa_bin_edges, sc.Variable):
        return toa_bin_edges
    elif isinstance(toa_bin_edges, int):
        return sc.linspace(
            dim='event_time_offset',
            start=0,
            stop=1 / 14,
            unit='s',
            num=toa_bin_edges + 1,
        )


def reduction(
    *,
    input_file: pathlib.Path,
    output_file: pathlib.Path,
    chunk_size: int = 1_000,
    detector_ids: list[int | str],
    compression: bool = False,
    logger: logging.Logger | None = None,
    toa_bin_edges: sc.Variable | int = 250,
    display: Callable | None = None,  # For Jupyter notebook display
) -> sc.DataGroup:
    import ipywidgets as widgets
    import scippnexus as snx

    if logger is None:
        logger = logging.getLogger(__name__)
    if display is None:
        display = logger.info

    with snx.File(input_file) as f:
        intrument_group = f['entry/instrument']
        dets = intrument_group[snx.NXdetector]
        detector_group_keys = list(dets.keys())
        display(widgets.Label(f"Found NXdetectors: {detector_group_keys}"))
        detector_id_map = {
            det_name: dets[det_name]
            for i, det_name in enumerate(detector_group_keys)
            if i in detector_ids or det_name in detector_ids
        }
        if len(detector_id_map) != len(detector_ids):
            raise ValueError(
                f"Requested detector ids {detector_ids} not found in the file.\n"
                f"Found {detector_group_keys}\n"
                f"Try using integer indices instead of names."
            )
        display(widgets.Label(f"Selected detectors: {list(detector_id_map.keys())}"))
        source_position = _retrieve_source_position(f)
        sample_position = _retrieve_sample_position(f)
        experiment_metadata = NMXExperimentMetadata(
            sc.DataGroup(
                {
                    # Placeholder for crystal rotation
                    'crystal_rotation': sc.vector([0, 0, 0], unit='deg'),
                    'sample_position': sample_position,
                    'source_position': source_position,
                    'sample_name': sc.scalar(f['entry/sample/name'][()]),
                }
            )
        )
        display(experiment_metadata)
        display("Experiment metadata component:")
        for name, component in experiment_metadata.items():
            display(f"{name}: {component}")

        _export_static_metadata_as_nxlauetof(
            experiment_metadata=experiment_metadata,
            output_file=output_file,
        )
        detector_grs = {}
        for det_name, det_group in detector_id_map.items():
            display(f"Processing {det_name}")
            if chunk_size <= 0:
                dg = det_group[()]
            else:
                # Slice the first chunk for metadata extraction
                dg = det_group['event_time_zero', 0:chunk_size]

            display(dg := _compute_positions(dg, auto_fix_transformations=True))
            detector = build_detector_desc(det_name, dg)
            detector_meta = sc.DataGroup(
                {
                    'fast_axis': detector.fast_axis,
                    'slow_axis': detector.slow_axis,
                    'origin_position': sc.vector([0, 0, 0], unit='m'),
                    'position': detector.position,
                    'detector_shape': sc.scalar(
                        (
                            dg['data'].sizes['x_pixel_offset'],
                            dg['data'].sizes['y_pixel_offset'],
                        )
                    ),
                    'x_pixel_size': detector.step_x,
                    'y_pixel_size': detector.step_y,
                    'detector_name': sc.scalar(detector.name),
                }
            )
            _export_detector_metadata_as_nxlauetof(
                NMXDetectorMetadata(detector_meta), output_file=output_file
            )

            da: sc.DataArray = dg['data']
            if chunk_size <= 0:
                counts = da.hist(event_time_offset=toa_bin_edges).rename_dims(
                    x_pixel_offset='x', y_pixel_offset='y', event_time_offset='t'
                )
                counts.coords['t'] = counts.coords['event_time_offset']

            else:
                num_chunks = calculate_number_of_chunks(
                    det_group, chunk_size=chunk_size
                )
                display(f"Number of chunks: {num_chunks}")
                counts = da.hist(event_time_offset=toa_bin_edges).rename_dims(
                    x_pixel_offset='x', y_pixel_offset='y', event_time_offset='t'
                )
                counts.coords['t'] = counts.coords['event_time_offset']
                for chunk_index in range(1, num_chunks):
                    cur_chunk = det_group[
                        'event_time_zero',
                        chunk_index * chunk_size : (chunk_index + 1) * chunk_size,
                    ]
                    display(f"Processing chunk {chunk_index + 1} of {num_chunks}")
                    cur_chunk = _compute_positions(
                        cur_chunk, auto_fix_transformations=True
                    )
                    cur_counts = (
                        cur_chunk['data']
                        .hist(event_time_offset=toa_bin_edges)
                        .rename_dims(
                            x_pixel_offset='x',
                            y_pixel_offset='y',
                            event_time_offset='t',
                        )
                    )
                    cur_counts.coords['t'] = cur_counts.coords['event_time_offset']
                    counts += cur_counts

            dg = sc.DataGroup(
                counts=counts,
                detector_shape=detector_meta['detector_shape'],
                detector_name=detector_meta['detector_name'],
            )
            display("Final data group:")
            display(dg)
            display("Saving reduced data to Nexus file...")
            _export_reduced_data_as_nxlauetof(
                dg, output_file=output_file, compress_counts=compression
            )
            detector_grs[det_name] = dg

    display("Reduction completed successfully.")
    return sc.DataGroup(detector_grs)


def _add_ess_reduction_args(arg: argparse.ArgumentParser) -> None:
    argument_group = arg.add_argument_group("ESS Reduction Options")
    argument_group.add_argument(
        "--chunk_size",
        type=int,
        default=1_000,
        help="Chunk size for processing (number of pulses per chunk).",
    )


def main() -> None:
    from ._executable_helper import build_logger, build_reduction_arg_parser

    parser = build_reduction_arg_parser()
    _add_ess_reduction_args(parser)
    args = parser.parse_args()

    input_file = pathlib.Path(args.input_file).resolve()
    output_file = pathlib.Path(args.output_file).resolve()
    logger = build_logger(args)

    logger.info("Input file: %s", input_file)
    logger.info("Output file: %s", output_file)

    reduction(
        input_file=input_file,
        output_file=output_file,
        chunk_size=args.chunk_size,
        detector_ids=args.detector_ids,
        compression=args.compression,
        toa_bin_edges=args.nbins,
        logger=logger,
    )
