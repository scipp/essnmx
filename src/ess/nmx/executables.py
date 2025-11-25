# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import logging
import pathlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import sciline
import scipp as sc
import scippnexus as snx

from ess.reduce.nexus.types import (
    Filename,
    NeXusComponent,
    Position,
    SampleRun,
)
from ess.reduce.time_of_flight import (
    TimeOfFlightLookupTableFilename,
    TofDetector,
)

from ._executable_helper import (
    ReductionConfig,
    build_logger,
    build_reduction_argument_parser,
    collect_matching_input_files,
    reduction_config_from_args,
)
from .configuration import InputConfig, WorkflowConfig
from .nexus import (
    _export_static_metadata_as_nxlauetof,
)
from .streaming import _validate_chunk_size
from .types import (
    NMXExperimentMetadata,
    TofSimulationMaxWavelength,
    TofSimulationMinWavelength,
)
from .workflows import (
    NMXWorkflow,
    assert_equal_mapped,
    map_detector_names,
    map_file_names,
    select_detector_names,
)


def _retrieve_crystal_rotation(
    file_path: Filename[SampleRun],
) -> Position[snx.NXcrystal, SampleRun]:
    with snx.File(file_path) as file:
        if 'crystal_rotation' not in file['entry/sample']:
            import warnings

            warnings.warn(
                "No crystal rotation found in the Nexus file under "
                "'entry/sample/crystal_rotation'. Returning zero rotation.",
                RuntimeWarning,
                stacklevel=2,
            )
            return Position[snx.NXcrystal, SampleRun](sc.vector([0, 0, 0], unit='deg'))

        # Temporary way of storing crystal rotation.
        # streaming-sample-mcstas module writes crystal rotation under
        # 'entry/sample/crystal_rotation' as an array of three values.
        return Position[snx.NXcrystal, SampleRun](
            file['entry/sample/crystal_rotation'][()]
        )


def retrieve_crystal_rotation(input_files: list[str]) -> sc.Variable:
    wf = sciline.Pipeline(providers=[_retrieve_crystal_rotation])
    wf = map_file_names(
        wf=wf,
        file_names=input_files,
        mapped_type=Position[snx.NXcrystal, SampleRun],
        reduce_func=assert_equal_mapped,
    )
    return wf.compute(Position[snx.NXcrystal, SampleRun])


def _decide_fast_axis(da: sc.DataArray) -> str:
    x_slice = da['x_pixel_offset', 0].coords['detector_number']
    y_slice = da['y_pixel_offset', 0].coords['detector_number']

    if (x_slice.max() < y_slice.max()).value:
        return 'y'
    elif (x_slice.max() > y_slice.max()).value:
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


def build_detector_desc(
    name: str, dg: sc.DataGroup, *, fast_axis: Literal['x', 'y'] | None = None
) -> DetectorDesc:
    da: sc.DataArray = dg['data']
    _fast_axis = fast_axis if fast_axis is not None else _decide_fast_axis(da)
    transformation_matrix = dg['transform_matrix']
    t_unit = transformation_matrix.unit
    fast_axis_vector = (
        sc.vector([1, 0, 0], unit=t_unit)
        if _fast_axis == 'x'
        else sc.vector([0, 1, 0], unit=t_unit)
    )
    slow_axis_vector = (
        sc.vector([0, 1, 0], unit=t_unit)
        if _fast_axis == 'x'
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
        fast_axis_name=_fast_axis,
        slow_axis_name='x' if _fast_axis == 'y' else 'y',
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


def _build_toa_bin_edges(
    *,
    min_toa: sc.Variable | int = 0,
    max_toa: sc.Variable | int = int((1 / 14) * 1_000),  # Default for ESS NMX
    toa_bin_edges: sc.Variable | int = 250,
) -> sc.Variable:
    if isinstance(toa_bin_edges, sc.Variable):
        return toa_bin_edges
    elif isinstance(toa_bin_edges, int):
        min_toa = sc.scalar(min_toa, unit='ms') if isinstance(min_toa, int) else min_toa
        max_toa = sc.scalar(max_toa, unit='ms') if isinstance(max_toa, int) else max_toa
        return sc.linspace(
            dim='event_time_offset',
            start=min_toa.value,
            stop=max_toa.to(unit=min_toa.unit).value,
            unit=min_toa.unit,
            num=toa_bin_edges + 1,
        )


def _build_time_bins(*, workflow_config: WorkflowConfig) -> sc.Variable | int:
    if workflow_config.min_time_bin is None or workflow_config.max_time_bin is None:
        return workflow_config.nbins

    min_toa = sc.scalar(
        workflow_config.min_time_bin, unit=workflow_config.time_bin_unit
    )
    max_toa = sc.scalar(
        workflow_config.max_time_bin, unit=workflow_config.time_bin_unit
    )
    return sc.linspace(
        dim=workflow_config.time_bin_coordinate,
        start=min_toa,
        stop=max_toa,
        num=workflow_config.nbins + 1,
    )


def _retrieve_input_file(input_file: list[pathlib.Path] | pathlib.Path) -> pathlib.Path:
    """Temporary helper to retrieve a single input file from the list
    Until multiple input file support is implemented.
    """
    if isinstance(input_file, list) and len(input_file) != 1:
        raise NotImplementedError(
            "Currently, only a single input file is supported for reduction."
        )
    elif isinstance(input_file, list):
        input_file_path = input_file[0]
    else:
        input_file_path = input_file

    return input_file_path


def _retrieve_display(
    logger: logging.Logger | None, display: Callable | None
) -> Callable:
    if display is not None:
        return display
    elif logger is not None:
        return logger.info
    else:
        return logging.getLogger(__name__).info


def _build_workflow(wf_config: WorkflowConfig) -> sciline.Pipeline:
    if wf_config.tof_lookup_table_file_path is None:
        workflow = NMXWorkflow(tof_simulation=True)
        workflow[TimeOfFlightLookupTableFilename] = wf_config.tof_lookup_table_file_path
        return workflow
    else:
        workflow = NMXWorkflow(tof_simulation=False)
        wmax = sc.scalar(wf_config.tof_simulation_max_wavelength, unit='angstrom')
        wmin = sc.scalar(wf_config.tof_simulation_min_wavelength, unit='angstrom')
        workflow[TofSimulationMaxWavelength] = wmax
        workflow[TofSimulationMinWavelength] = wmin
        return workflow


def _map_workflow(
    *, wf: sciline.Pipeline, input_config: InputConfig
) -> sciline.Pipeline:
    detector_names = select_detector_names(
        input_files=input_config.input_file, detector_ids=input_config.detector_ids
    )
    wf = map_file_names(wf=wf, file_names=input_config.input_file)
    static_metadata_types = (
        Position[snx.NXsource, SampleRun],
        Position[snx.NXsample, SampleRun],
        NeXusComponent[snx.NXsample, SampleRun],
    )
    for static_type in static_metadata_types:
        wf[static_type] = wf[static_type].reduce(func=assert_equal_mapped)

    wf = map_detector_names(wf=wf, detector_names=detector_names)
    return wf


def reduction(
    *,
    config: ReductionConfig,
    logger: logging.Logger | None = None,
    display: Callable | None = None,
) -> sc.DataGroup:
    """Reduce NMX data from a Nexus file and export to NXLauetof(ESS NMX specific) file.

    This workflow is written as a flatten function without using sciline Pipeline.
    It is because the first part of NMX reduction only requires
    a few steps of processing and it is overkill to use a Pipeline or GenericWorkflow.

    We also do not apply frame unwrapping or pulse skipping here,
    as it is not expected from NMX experiments.

    Frame unwrapping may be applied later on the result of this function if needed
    however, then the whole range of `event_time_offset` should have been histogrammed
    so that the unwrapping can be applied.
    i.e. `min_toa` should be 0 and `max_toa` should be 1/14 seconds
    for 14 Hz pulse frequency.
    TODO: Implement tof/wavelength workflow for NMX.

    Parameters
    ----------
    config:
        Reduction configuration.

        Data reduction parameters are taken from this config
        instead of passing them directly as keyword arguments.
        They can be either built from command-line arguments
        using `ReductionConfig.from_args()` or constructed manually.

        If the reduced data is successfully written to the output file
        the configuration is also saved there for future reference.
    logger:
        Logger to use for logging messages. If None, a default logger is created.
    display:
        Callable for displaying messages, useful in Jupyter notebooks. If None,
        defaults to logger.info.

    Returns
    -------
    sc.DataGroup:
        A DataGroup containing the reduced data for each selected detector.

    """
    import scippnexus as snx

    display = _retrieve_display(logger, display)
    # Temporary way of retrieving crystal rotation from the input file.
    crystal_rotation = retrieve_crystal_rotation(config.inputs.input_file)
    workflow = _build_workflow(wf_config=config.workflow)
    workflow = _map_workflow(wf=workflow, input_config=config.inputs)
    if config.output.verbose:
        display(workflow)
        display(workflow.visualize(compact=True))

    toa_bin_edges = _build_time_bins(workflow_config=config.workflow)
    display(toa_bin_edges)
    results = workflow.compute(
        (
            TofDetector[SampleRun],
            Position[snx.NXsource, SampleRun],
            Position[snx.NXsample, SampleRun],
            NeXusComponent[snx.NXsample, SampleRun],
        )
    )
    experiment_metadata = NMXExperimentMetadata(
        sc.DataGroup(
            {
                'crystal_rotation': crystal_rotation,
                'sample_position': results[Position[snx.NXsample, SampleRun]],
                'source_position': results[Position[snx.NXsource, SampleRun]],
                'sample_name': results[NeXusComponent[snx.NXsample, SampleRun]]['name'],
            }
        )
    )
    display(experiment_metadata)

    _export_static_metadata_as_nxlauetof(
        experiment_metadata=experiment_metadata,
        output_file=config.output.output_file,
    )
    return results[TofDetector[SampleRun]]
    #     display(experiment_metadata)
    #     display("Experiment metadata component:")
    #     for name, component in experiment_metadata.items():
    #         display(f"{name}: {component}")

    #     _export_static_metadata_as_nxlauetof(
    #         experiment_metadata=experiment_metadata,
    #         output_file=output_file,
    #     )
    #     detector_grs = {}
    #     for det_name, det_group in detector_id_map.items():
    #         display(f"Processing {det_name}")
    #         if chunk_size <= 0:
    #             dg = det_group[()]
    #         else:
    #             # Slice the first chunk for metadata extraction
    #             dg = det_group['event_time_zero', 0:chunk_size]

    #         display("Computing detector positions...")
    #         display(dg := _compute_positions(dg, auto_fix_transformations=True))
    #         detector = build_detector_desc(det_name, dg, fast_axis=fast_axis)
    #         detector_meta = sc.DataGroup(
    #             {
    #                 'fast_axis': detector.fast_axis,
    #                 'slow_axis': detector.slow_axis,
    #                 'origin_position': sc.vector([0, 0, 0], unit='m'),
    #                 'position': detector.position,
    #                 'detector_shape': sc.scalar(
    #                     (
    #                         dg['data'].sizes['x_pixel_offset'],
    #                         dg['data'].sizes['y_pixel_offset'],
    #                     )
    #                 ),
    #                 'x_pixel_size': detector.step_x,
    #                 'y_pixel_size': detector.step_y,
    #                 'detector_name': sc.scalar(detector.name),
    #             }
    #         )
    #         _export_detector_metadata_as_nxlauetof(
    #             NMXDetectorMetadata(detector_meta), output_file=output_file
    #         )

    #         da: sc.DataArray = dg['data']
    #         event_time_offset_unit = da.bins.coords['event_time_offset'].bins.unit
    #         display("Event time offset unit: %s", event_time_offset_unit)
    #         toa_bin_edges = toa_bin_edges.to(unit=event_time_offset_unit, copy=False)
    #         if chunk_size <= 0:
    #             counts = da.hist(event_time_offset=toa_bin_edges).rename_dims(
    #                 x_pixel_offset='x', y_pixel_offset='y', event_time_offset='t'
    #             )
    #             counts.coords['t'] = counts.coords['event_time_offset']

    #         else:
    #             num_chunks = calculate_number_of_chunks(
    #                 det_group, chunk_size=chunk_size
    #             )
    #             display(f"Number of chunks: {num_chunks}")
    #             counts = da.hist(event_time_offset=toa_bin_edges).rename_dims(
    #                 x_pixel_offset='x', y_pixel_offset='y', event_time_offset='t'
    #             )
    #             counts.coords['t'] = counts.coords['event_time_offset']
    #             for chunk_index in range(1, num_chunks):
    #                 cur_chunk = det_group[
    #                     'event_time_zero',
    #                     chunk_index * chunk_size : (chunk_index + 1) * chunk_size,
    #                 ]
    #                 display(f"Processing chunk {chunk_index + 1} of {num_chunks}")
    #                 cur_chunk = _compute_positions(
    #                     cur_chunk, auto_fix_transformations=True
    #                 )
    #                 cur_counts = (
    #                     cur_chunk['data']
    #                     .hist(event_time_offset=toa_bin_edges)
    #                     .rename_dims(
    #                         x_pixel_offset='x',
    #                         y_pixel_offset='y',
    #                         event_time_offset='t',
    #                     )
    #                 )
    #                 cur_counts.coords['t'] = cur_counts.coords['event_time_offset']
    #                 counts += cur_counts
    #                 display("Accumulated counts:")
    #                 display(counts.sum().data)

    #         dg = sc.DataGroup(
    #             counts=counts,
    #             detector_shape=detector_meta['detector_shape'],
    #             detector_name=detector_meta['detector_name'],
    #         )
    #         display("Final data group:")
    #         display(dg)
    #         display("Saving reduced data to Nexus file...")
    #         _export_reduced_data_as_nxlauetof(
    #             dg,
    #             output_file=output_file,
    #             compress_counts=(compression == Compression.BITSHUFFLE_LZ4),
    #         )
    #         detector_grs[det_name] = dg

    # display("Reduction completed successfully.")
    # return sc.DataGroup(detector_grs)


def main() -> None:
    parser = build_reduction_argument_parser()
    config: ReductionConfig = reduction_config_from_args(parser.parse_args())

    input_file = collect_matching_input_files(*config.inputs.input_file)
    output_file = pathlib.Path(config.output.output_file).resolve()
    logger = build_logger(config.output)

    logger.info("Input file: %s", input_file)
    logger.info("Output file: %s", output_file)

    reduction(config=config, logger=logger)
