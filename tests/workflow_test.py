# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pandas as pd
import pytest
import sciline as sl
import scipp as sc

from ess.nmx import default_parameters
from ess.nmx.data import small_mcstas_2_sample, small_mcstas_3_sample
from ess.nmx.mcstas.load import providers as load_providers
from ess.nmx.reduction import (
    NMXReducedDataGroup,
    format_nmx_reduced_data,
    merge_panels,
    reduce_raw_event_counts,
)
from ess.nmx.types import (
    DetectorIndex,
    FilePath,
    NMXRawEventCountsDataGroup,
    TimeBinSteps,
)


@pytest.fixture(params=[small_mcstas_2_sample, small_mcstas_3_sample])
def mcstas_file_path(
    request: pytest.FixtureRequest, mcstas_2_deprecation_warning_context
) -> str:
    if request.param == small_mcstas_2_sample:
        with mcstas_2_deprecation_warning_context():
            return request.param()

    return request.param()


@pytest.fixture
def mcstas_workflow(mcstas_file_path: str) -> sl.Pipeline:
    return sl.Pipeline(
        [*load_providers, reduce_raw_event_counts, format_nmx_reduced_data],
        params={
            FilePath: mcstas_file_path,
            TimeBinSteps: 50,
            **default_parameters,
        },
    )


@pytest.fixture
def multi_bank_mcstas_workflow(mcstas_workflow: sl.Pipeline) -> sl.Pipeline:
    pl = mcstas_workflow.copy()
    pl[NMXReducedDataGroup] = (
        pl[NMXReducedDataGroup]
        .map(pd.DataFrame({DetectorIndex: range(3)}).rename_axis('panel'))
        .reduce(index='panel', func=merge_panels)
    )
    return pl


def test_pipeline_builder(mcstas_workflow: sl.Pipeline, mcstas_file_path: str) -> None:
    assert mcstas_workflow.get(FilePath).compute() == mcstas_file_path


def test_pipeline_mcstas_loader(mcstas_workflow: sl.Pipeline) -> None:
    """Test if the loader graph is complete."""
    mcstas_workflow[DetectorIndex] = 0
    nmx_data = mcstas_workflow.compute(NMXRawEventCountsDataGroup)
    assert isinstance(nmx_data, sc.DataGroup)
    assert nmx_data.sizes['id'] == 1280 * 1280


def test_pipeline_mcstas_reduction(multi_bank_mcstas_workflow: sl.Pipeline) -> None:
    """Test if the loader graph is complete."""
    nmx_reduced_data = multi_bank_mcstas_workflow.compute(NMXReducedDataGroup)
    assert nmx_reduced_data.shape == (3, (1280, 1280)[0] * (1280, 1280)[1], 50)
    # Panel, Pixels, Time bins
    assert isinstance(nmx_reduced_data, sc.DataGroup)
    assert nmx_reduced_data.sizes['t'] == 50
