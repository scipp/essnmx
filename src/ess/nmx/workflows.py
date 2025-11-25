# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from collections.abc import Iterable

import pandas as pd
import sciline
import scipp as sc
import scippnexus as snx
import tof

from ess.reduce.nexus.types import (
    Component,
    Filename,
    NeXusComponent,
    NeXusName,
    NeXusTransformationChain,
    RunType,
    SampleRun,
)
from ess.reduce.time_of_flight import (
    DetectorLtotal,
    GenericTofWorkflow,
    LtotalRange,
    NumberOfSimulatedNeutrons,
    SimulationResults,
    SimulationSeed,
    TofDetector,
    TofLookupTableWorkflow,
)
from ess.reduce.workflow import register_workflow

from .types import (
    TofSimulationMaxWavelength,
    TofSimulationMinWavelength,
)

default_parameters = {
    TofSimulationMaxWavelength: sc.scalar(3.6, unit='angstrom'),
    TofSimulationMinWavelength: sc.scalar(1.8, unit='angstrom'),
}


def _validate_mergable_workflow(wf: sciline.Pipeline):
    if wf.indices:
        raise NotImplementedError("Only flat workflow can be merged.")


def _merge_workflows(
    base_wf: sciline.Pipeline, merged_wf: sciline.Pipeline
) -> sciline.Pipeline:
    _validate_mergable_workflow(base_wf)
    _validate_mergable_workflow(merged_wf)

    for key, spec in merged_wf.underlying_graph.nodes.items():
        if 'value' in spec:
            base_wf[key] = spec['value']
        elif (provider_spec := spec.get('provider')) is not None:
            base_wf.insert(provider_spec.func)

    return base_wf


def _simulate_fixed_wavelength_tof(
    wmin: TofSimulationMinWavelength,
    wmax: TofSimulationMaxWavelength,
    ltotal_range: LtotalRange,
    neutrons: NumberOfSimulatedNeutrons,
    seed: SimulationSeed,
) -> SimulationResults:
    """
    Simulate a pulse of neutrons propagating through a chopper cascade using the
    ``tof`` package (https://tof.readthedocs.io).

    Parameters
    ----------
    """
    source = tof.Source(
        facility="ess", neutrons=neutrons, pulses=2, seed=seed, wmax=wmax, wmin=wmin
    )
    nmx_det = tof.Detector(distance=max(ltotal_range), name="detector")
    model = tof.Model(source=source, choppers=[], detectors=[nmx_det])
    results = model.run()
    events = results["detector"].data.squeeze().flatten(to="event")
    return SimulationResults(
        time_of_arrival=events.coords["toa"],
        speed=events.coords["speed"],
        wavelength=events.coords["wavelength"],
        weight=events.data,
        distance=results["detector"].distance,
    )


def _ltotal_range(detector_ltotal: DetectorLtotal[SampleRun]) -> LtotalRange:
    margin = sc.scalar(0.1, unit='m').to(
        unit=detector_ltotal.unit
    )  # Hardcoded margin of 10 cm. No reason...
    ltotal_min = sc.min(detector_ltotal) - margin
    ltotal_max = sc.max(detector_ltotal) + margin
    return LtotalRange((ltotal_min, ltotal_max))


def _patch_workflow_lookup_table_steps(*, wf: sciline.Pipeline) -> sciline.Pipeline:
    patched_wf = wf.copy()

    # Use TofLookupTableWorkflow
    patched_wf = _merge_workflows(patched_wf, TofLookupTableWorkflow())
    patched_wf.insert(_simulate_fixed_wavelength_tof)
    patched_wf.insert(_ltotal_range)
    return patched_wf


def _merge_panels(*da: sc.DataArray) -> sc.DataArray:
    """Merge multiple DataArrays representing different panels into one."""
    merged = sc.concat(da, dim='panel')
    return merged


def assert_equal_mapped(*da):
    """Merge multiple DataArrays representing different runs/panels into one."""
    from scipp.testing import assert_identical

    def _assert_equals(*da):
        if len(da) == 1:
            return da[0]
        elif assert_identical(da[0], da[1]):
            raise AssertionError("DataArrays are not equal.")

        return _assert_equals(*da[1:])

    return _assert_equals(*da)


def _concatenate_events_bins(*da: sc.DataArray) -> sc.DataArray:
    """Merge multiple DataArrays representing different runs into one."""

    def _concatenate(*da: sc.DataArray) -> sc.DataArray:
        if len(da) == 1:
            return da[0]

        return _concatenate(da[0].bins.concatenate(da[1]), *da[2:])

    return _concatenate(*da)


def _get_transformation_chain(
    detector: NeXusComponent[Component, RunType],
) -> NeXusTransformationChain[Component, RunType]:
    """
    Extract the transformation chain from a NeXus detector group.

    Notes
    -----
    This provider patches the NXlog transformation by inserting
    value of `0` if they are empty.

    Parameters
    ----------
    detector:
        NeXus detector group.
    """
    import warnings

    chain = detector['depends_on']

    empty_transformations = [
        transformation
        for transformation in chain.transformations.values()
        if 'time' in transformation.value.dims
        and transformation.sizes['time'] == 0  # empty log
    ]
    if any(empty_transformations):
        warnings.warn(
            "Inserting scalar zero into transformation due to empty log entries. "
            "This may lead to incorrect results. Please check the data carefully.",
            UserWarning,
            stacklevel=2,
        )
    for transformation in empty_transformations:
        orig_value = transformation.value
        orig_value = sc.scalar(0, unit=orig_value.unit, dtype=orig_value.dtype)
        transformation.value = orig_value

    return NeXusTransformationChain[Component, RunType](chain)


def select_detector_names(
    *, input_files: list[str] | None = None, detector_ids: Iterable[int] = (0, 1, 2)
):
    if input_files is not None:
        detector_names = []
        # Collect all detector names from input files
        for input_file in input_files:
            with snx.File(input_file) as nexus_file:
                detector_names.extend(
                    nexus_file['entry/instrument'][snx.NXdetector].keys()
                )
        detector_names = sorted(set(detector_names))
        return [detector_names[i_d] for i_d in detector_ids]
    else:
        return ['detector_panel_0', 'detector_panel_1', 'detector_panel_2']


def map_detector_names(
    *,
    wf: sciline.Pipeline,
    detector_names: list[str] | None = None,
) -> sciline.Pipeline:
    """Map detector indices(`panel`) to detector names in the workflow."""
    detector_name_map = pd.DataFrame({NeXusName[snx.NXdetector]: detector_names})
    detector_name_map.rename_axis(index='panel', inplace=True)
    wf[TofDetector[SampleRun]] = (
        wf[TofDetector[SampleRun]].map(detector_name_map).reduce(func=_merge_panels)
    )
    return wf


def map_file_names(
    *,
    wf: sciline.Pipeline,
    file_names: list[str] | None = None,
    mapped_type: type = TofDetector[SampleRun],
    reduce_func=_concatenate_events_bins,
) -> sciline.Pipeline:
    """Map detector indices(`panel`) to detector names in the workflow."""
    file_name_map = pd.DataFrame({Filename[SampleRun]: file_names})
    file_name_map.rename_axis(index='run', inplace=True)
    wf[mapped_type] = wf[mapped_type].map(file_name_map).reduce(func=reduce_func)
    return wf


def NMXWorkflow(tof_simulation: bool = True) -> sciline.Pipeline:
    generic_wf = GenericTofWorkflow(run_types=[SampleRun], monitor_types=[])
    if tof_simulation:
        generic_wf = _patch_workflow_lookup_table_steps(wf=generic_wf)

    generic_wf.insert(_get_transformation_chain)
    for key, value in default_parameters.items():
        generic_wf[key] = value

    return generic_wf


@register_workflow
def NMXPreDialsWorkflow() -> sciline.Pipeline:
    return NMXWorkflow()


__all__ = ['NMXWorkflow']
