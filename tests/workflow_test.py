# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pytest
import sciline as sl
import scipp as sc


@pytest.fixture
def mcstas_workflow() -> sl.Pipeline:
    from ess.nmx import build_workflow
    from ess.nmx.data import small_mcstas_sample

    return build_workflow(small_mcstas_sample())


def test_pipeline_builder(mcstas_workflow: sl.Pipeline) -> None:
    from ess.nmx.data import small_mcstas_sample
    from ess.nmx.loader import InputFileName

    assert mcstas_workflow.get(InputFileName).compute() == small_mcstas_sample()


def test_pipeline_mcstas_loader(mcstas_workflow: sl.Pipeline) -> None:
    """Test if the loader graph is complete."""
    from ess.nmx.loader import Events, FileTypeMcStas

    assert isinstance(
        mcstas_workflow.get(Events[FileTypeMcStas]).compute(), sc.DataArray
    )


def test_pipeline_mcstas_binning(mcstas_workflow: sl.Pipeline) -> None:
    """Test if the data reduction graph is complete."""
    from ess.nmx.loader import FileTypeMcStas
    from ess.nmx.reduction import GroupedByPixelID, TimeBinned

    results = mcstas_workflow.get(
        (GroupedByPixelID[FileTypeMcStas], TimeBinned[FileTypeMcStas])
    ).compute()

    assert isinstance(results[GroupedByPixelID[FileTypeMcStas]], sc.DataArray)
    assert isinstance(results[TimeBinned[FileTypeMcStas]], sc.DataArray)
