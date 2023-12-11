# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pytest
import sciline as sl
import scipp as sc


@pytest.fixture
def mcstas_workflow() -> sl.Pipeline:
    from ess.nmx.data import small_mcstas_sample
    from ess.nmx.loader import (
        DefaultMaximumProbability,
        DefaultMcStasEventDataSchema,
        InputFileName,
        MaximumProbability,
        McStasEventDataSchema,
    )
    from ess.nmx.reduction import TimeBinStep, get_intervals_mcstas
    from ess.nmx.workflow import collect_default_parameters, providers

    return sl.Pipeline(
        list(providers) + [get_intervals_mcstas],
        params={
            **collect_default_parameters(),
            InputFileName: small_mcstas_sample(),
            TimeBinStep: TimeBinStep(1),
            McStasEventDataSchema: DefaultMcStasEventDataSchema,
            MaximumProbability: DefaultMaximumProbability,
        },
    )


def test_pipeline_builder(mcstas_workflow: sl.Pipeline) -> None:
    from ess.nmx.data import small_mcstas_sample
    from ess.nmx.loader import InputFileName

    assert mcstas_workflow.get(InputFileName).compute() == small_mcstas_sample()


def test_pipeline_mcstas_loader(mcstas_workflow: sl.Pipeline) -> None:
    """Test if the loader graph is complete."""
    from ess.nmx.loader import Events

    assert isinstance(mcstas_workflow.get(Events).compute(), sc.DataArray)


def test_pipeline_mcstas_binning(mcstas_workflow: sl.Pipeline) -> None:
    """Test if the data reduction graph is complete."""
    from ess.nmx.reduction import GroupedByPixelID, TimeBinned

    results = mcstas_workflow.get((GroupedByPixelID, TimeBinned)).compute()

    assert isinstance(results[GroupedByPixelID], sc.DataArray)
    assert isinstance(results[TimeBinned], sc.DataArray)
