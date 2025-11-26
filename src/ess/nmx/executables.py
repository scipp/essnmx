# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import logging
import pathlib
from collections.abc import Callable, Iterable

import sciline
import scipp as sc
import scippnexus as snx

from ess.reduce.nexus.types import (
    EmptyDetector,
    Filename,
    NeXusComponent,
    NeXusName,
    NeXusTransformation,
    Position,
    SampleRun,
)
from ess.reduce.time_of_flight import (
    DetectorLtotal,
    TimeOfFlightLookupTable,
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
from ._io import (
    NMXDetectorMetadata,
    NMXMonitorMetadata,
    NMXSampleMetadata,
    NMXSourceMetadata,
    export_detector_metadata_as_nxlauetof,
    export_monitor_metadata_as_nxlauetof,
    export_reduced_data_as_nxlauetof,
    export_static_metadata_as_nxlauetof,
)
from .configuration import TimeBinCoordinate, WorkflowConfig
from .streaming import _validate_chunk_size
from .types import (
    NMXCrystalRotation,
    TofSimulationMaxWavelength,
    TofSimulationMinWavelength,
)
from .workflows import (
    NMXWorkflow,
    _concatenate_events_bins,
    _merge_panels,
    _patch_workflow_lookup_table_steps,
    assemble_detector_metadata,
    assert_equal_mapped,
    map_detector_names,
    map_file_paths,
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


def retrieve_crystal_rotation(file_paths: Iterable[pathlib.Path]) -> sc.Variable:
    wf = sciline.Pipeline(providers=[_retrieve_crystal_rotation])
    wf = map_file_paths(
        wf=wf,
        file_paths=file_paths,
        mapped_type=Position[snx.NXcrystal, SampleRun],
        reduce_func=assert_equal_mapped,
    )
    return wf.compute(Position[snx.NXcrystal, SampleRun])


def calculate_number_of_chunks(detector_gr: snx.Group, *, chunk_size: int = 0) -> int:
    _validate_chunk_size(chunk_size)
    event_time_zero_size = detector_gr.sizes['event_time_zero']
    if chunk_size == -1:
        return 1  # Read all at once
    else:
        return event_time_zero_size // chunk_size + int(
            event_time_zero_size % chunk_size != 0
        )


def _try_building_tof_bin_edges(
    *,
    workflow_config: WorkflowConfig,
    lookup_table: TimeOfFlightLookupTable,
) -> sc.Variable | int | None:
    if workflow_config.min_time_bin is None and workflow_config.max_time_bin is None:
        return workflow_config.nbins
    elif workflow_config.min_time_bin is None or workflow_config.max_time_bin is None:
        return

    # We have both min and max time bins specified
    min_t = sc.scalar(workflow_config.min_time_bin, unit=workflow_config.time_bin_unit)
    max_t = sc.scalar(workflow_config.max_time_bin, unit=workflow_config.time_bin_unit)
    if workflow_config.time_bin_coordinate == TimeBinCoordinate.event_time_offset:
        min_tof = sc.lookup(lookup_table.min(dim='distance'))[min_t]
        max_tof = sc.lookup(lookup_table.max(dim='distance'))[max_t]
    else:  # time_of_flight
        min_tof = min_t
        max_tof = max_t

    return sc.linspace(
        dim='tof', start=min_tof, stop=max_tof, num=workflow_config.nbins + 1
    )


def _decide_edge(
    tof_bin_edges: sc.Variable,
    config_num: int | None,
    config: WorkflowConfig,
    lookup_table: TimeOfFlightLookupTable,
    method: Callable,
) -> sc.Variable:
    if config_num is not None:
        t = sc.scalar(config_num, unit=config.time_bin_unit)
        if config.time_bin_coordinate == TimeBinCoordinate.event_time_offset:
            tof_edge = sc.lookup(method(lookup_table, dim='distance'))[t]
        else:
            tof_edge = t
    else:
        tof_edge = method(tof_bin_edges)
    return tof_edge


def _finalize_tof_bin_edges(
    *,
    base_bin_edges: sc.Variable | int | None,
    tof_das: sc.DataGroup,
    config: WorkflowConfig,
    lookup_table: TimeOfFlightLookupTable,
) -> sc.Variable:
    if isinstance(base_bin_edges, sc.Variable):
        return base_bin_edges
    tof_bin_edges = sc.concat(
        tuple(tof_da.coords['tof'] for tof_da in tof_das.values()), dim='tof'
    )
    min_tof = _decide_edge(
        tof_bin_edges, config.min_time_bin, config, lookup_table, method=sc.min
    )
    max_tof = _decide_edge(
        tof_bin_edges, config.max_time_bin, config, lookup_table, method=sc.max
    )
    return sc.linspace(dim='tof', start=min_tof, stop=max_tof, num=config.nbins + 1)


def _retrieve_display(
    logger: logging.Logger | None, display: Callable | None
) -> Callable:
    if display is not None:
        return display
    elif logger is not None:
        return logger.info
    else:
        return logging.getLogger(__name__).info


def _map_static_metadata_workflow(
    *, wf: sciline.Pipeline, input_files: Iterable[pathlib.Path]
) -> sciline.Pipeline:
    wf = wf.copy()
    static_metadata_types = (
        Position[snx.NXsource, SampleRun],
        Position[snx.NXsample, SampleRun],
        NeXusComponent[snx.NXsample, SampleRun],
    )
    for static_type in static_metadata_types:
        wf = map_file_paths(
            wf=wf,
            file_paths=input_files,
            mapped_type=static_type,
            reduce_func=assert_equal_mapped,
        )

    return wf


def _compute_lookup_table(
    *,
    base_wf: sciline.Pipeline,
    workflow_config: WorkflowConfig,
    file_paths: Iterable[pathlib.Path],
    detector_names: Iterable[str],
) -> sc.DataArray:
    wf = base_wf.copy()
    if workflow_config.tof_lookup_table_file_path is not None:
        wf[TimeOfFlightLookupTableFilename] = workflow_config.tof_lookup_table_file_path
    else:
        wf = _patch_workflow_lookup_table_steps(wf=wf)
        wmax = sc.scalar(workflow_config.tof_simulation_max_wavelength, unit='angstrom')
        wmin = sc.scalar(workflow_config.tof_simulation_min_wavelength, unit='angstrom')
        wf[TofSimulationMaxWavelength] = wmax
        wf[TofSimulationMinWavelength] = wmin
        wf = map_file_paths(
            wf=wf,
            file_paths=file_paths,
            mapped_type=DetectorLtotal[SampleRun],
            reduce_func=_concatenate_events_bins,
        )
        wf = map_detector_names(
            wf=wf,
            detector_names=detector_names,
            mapped_type=DetectorLtotal[SampleRun],
            reduce_func=_merge_panels,
        )

    return wf.compute(TimeOfFlightLookupTable)


def _validate_static_info_and_cache(
    *, wf: sciline.Pipeline, input_files: Iterable[pathlib.Path]
) -> sciline.Pipeline:
    """Check that static metadata matches across input files and cache them.

    **Note**: ``base_wf`` is modified in-place and also returned.
    """
    # Static metadata extraction
    # Temporary way of retrieving crystal rotation from the input file.
    crystal_rotation = retrieve_crystal_rotation(input_files)
    static_wf = _map_static_metadata_workflow(wf=wf, input_files=input_files)
    results = static_wf.compute(
        (
            Position[snx.NXsource, SampleRun],
            Position[snx.NXsample, SampleRun],
            NeXusComponent[snx.NXsample, SampleRun],
        )
    )
    # Update base workflow with static metadata results.
    # We can do this safely as these values are confirmed to be equal across files.
    for key, value in results.items():
        wf[key] = value
    wf[NMXCrystalRotation] = crystal_rotation

    return wf


def _compute_and_cache_lookup_table(
    *,
    wf: sciline.Pipeline,
    workflow_config: WorkflowConfig,
    input_files: Iterable[pathlib.Path],
    detector_names: Iterable[str],
    display: Callable,
) -> sciline.Pipeline:
    """Compute and cache the TOF lookup table in the workflow.

    **Note**: ``base_wf`` is modified in-place and also returned.
    """
    # We compute one lookup table that covers all range
    # to avoid multiple tof simulations.
    if workflow_config.tof_lookup_table_file_path is None:
        display("Computing TOF lookup table from simulation...")
    else:
        display("Loading TOF lookup table from file...")

    lookup_table = _compute_lookup_table(
        base_wf=wf,
        workflow_config=workflow_config,
        file_paths=input_files,
        detector_names=detector_names,
    )
    wf[TimeOfFlightLookupTable] = lookup_table
    return wf


def _map_tof_detector_workflow(
    *, wf: sciline.Pipeline, input_files: Iterable[pathlib.Path]
) -> sciline.Pipeline:
    cur_wf = map_file_paths(
        wf=wf,
        file_paths=input_files,
        mapped_type=TofDetector[SampleRun],
        reduce_func=_concatenate_events_bins,
    )
    cur_wf = map_file_paths(
        wf=cur_wf,
        file_paths=input_files,
        mapped_type=NeXusTransformation[snx.NXdetector, SampleRun],
        reduce_func=assert_equal_mapped,
    )
    cur_wf[NeXusComponent[snx.NXdetector, SampleRun]] = cur_wf[
        NeXusComponent[snx.NXdetector, SampleRun]
    ].reduce(func=assert_equal_mapped)
    cur_wf[EmptyDetector[SampleRun]] = cur_wf[EmptyDetector[SampleRun]].reduce(
        func=assert_equal_mapped
    )
    return cur_wf


def reduction(
    *,
    config: ReductionConfig,
    logger: logging.Logger | None = None,
    display: Callable | None = None,
) -> sc.DataGroup:
    """Reduce NMX data from a Nexus file and export to NXLauetof(ESS NMX specific) file.

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
    input_files = collect_matching_input_files(*config.inputs.input_file)
    detector_names = select_detector_names(
        input_files=input_files, detector_ids=config.inputs.detector_ids
    )
    base_wf = NMXWorkflow(tof_simulation=False)
    base_wf = _validate_static_info_and_cache(wf=base_wf, input_files=input_files)
    base_wf = _compute_and_cache_lookup_table(
        wf=base_wf,
        workflow_config=config.workflow,
        input_files=input_files,
        detector_names=detector_names,
        display=display,
    )
    export_static_metadata_as_nxlauetof(
        sample_metadata=base_wf.compute(NMXSampleMetadata),
        source_metadata=base_wf.compute(NMXSourceMetadata),
        output_file=config.output.output_file,
    )

    lookup = base_wf.compute(TimeOfFlightLookupTable)
    tof_bin_edges = _try_building_tof_bin_edges(
        workflow_config=config.workflow, lookup_table=lookup
    )
    tof_das = sc.DataGroup()
    detector_metas = sc.DataGroup()
    for detector_name in detector_names:
        cur_wf = base_wf.copy()
        cur_wf[NeXusName[snx.NXdetector]] = detector_name
        cur_wf = _map_tof_detector_workflow(wf=cur_wf, input_files=input_files)
        results = cur_wf.compute(
            (
                TofDetector[SampleRun],
                NeXusComponent[snx.NXdetector, SampleRun],
                NeXusTransformation[snx.NXdetector, SampleRun],
                Position[snx.NXsource, SampleRun],
                EmptyDetector[SampleRun],
            )
        )
        detector_metadata = assemble_detector_metadata(
            detector_component=results[NeXusComponent[snx.NXdetector, SampleRun]],
            transformation=results[NeXusTransformation[snx.NXdetector, SampleRun]],
            source_position=results[Position[snx.NXsource, SampleRun]],
            empty_detector=results[EmptyDetector[SampleRun]],
        )
        export_detector_metadata_as_nxlauetof(
            detector_metadata=detector_metadata, output_file=config.output.output_file
        )
        # If tof_bin_edges is not a Variable,
        # tof dimension should be rebinned after collecting all detectors' data.
        tof_bins = tof_bin_edges if isinstance(tof_bin_edges, sc.Variable) else 2
        detector_metas[detector_name] = detector_metadata
        tof_das[detector_name] = results[TofDetector[SampleRun]].bin(tof=tof_bins)

    final_tof_bin_edges = _finalize_tof_bin_edges(
        base_bin_edges=tof_bin_edges,
        tof_das=tof_das,
        config=config.workflow,
        lookup_table=lookup,
    )
    monitor_metadata = NMXMonitorMetadata(
        tof_bin_coord='tof',
        # TODO: Use real monitor data
        # Currently NMX simulations or experiments do not have monitors
        monitor_histogram=sc.DataArray(
            coords={'tof': final_tof_bin_edges},
            data=sc.ones_like(final_tof_bin_edges[:-1]),
        ),
    )
    export_monitor_metadata_as_nxlauetof(
        monitor_metadata=monitor_metadata, output_file=config.output.output_file
    )

    # Histogram detector counts
    tof_histograms = sc.DataGroup()
    for detector_name, tof_da in tof_das.items():
        det_meta: NMXDetectorMetadata = detector_metas[detector_name]
        histogram = tof_da.hist(tof=final_tof_bin_edges)
        tof_histograms[detector_name] = histogram
        export_reduced_data_as_nxlauetof(
            detector_name=det_meta.detector_name,
            da=histogram,
            output_file=config.output.output_file,
            compress_mode=config.output.compression,
        )

    return sc.DataGroup(
        metadata=detector_metas, histogram=tof_histograms, lookup_table=lookup
    )


def main() -> None:
    parser = build_reduction_argument_parser()
    config: ReductionConfig = reduction_config_from_args(parser.parse_args())

    input_file = collect_matching_input_files(*config.inputs.input_file)
    output_file = pathlib.Path(config.output.output_file).resolve()
    logger = build_logger(config.output)

    logger.info("Input file: %s", input_file)
    logger.info("Output file: %s", output_file)

    reduction(config=config, logger=logger)
