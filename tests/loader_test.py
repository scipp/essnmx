# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc


@pytest.fixture
def small_sample_dg() -> sc.DataGroup:
    return sc.DataGroup(
        weights=sc.linspace(dim='event', unit='counts', start=0.01, stop=0.02, num=100),
        x_list=sc.arange(dim='event', unit='m', start=0, stop=1, step=0.01),
        y_list=sc.arange(dim='event', unit='m', start=0, stop=1, step=0.01),
        id_list=sc.arange(
            dim='event', unit='dimensionless', start=0, stop=100, step=1, dtype=int
        ),
        t_list=sc.linspace(dim='event', unit='s', start=0, stop=1, num=100),
    )


def test_file_reader_mcstas() -> None:
    import scippnexus as snx

    from ess.nmx.data import small_mcstas_sample
    from ess.nmx.loader import (
        DefaultMaximumProbability,
        DefaultMcStasEventDataSchema,
        InputFileName,
        _get_entry_path_mcstas,
        read_file,
    )

    file_path = InputFileName(small_mcstas_sample())
    entry_path = _get_entry_path_mcstas(DefaultMcStasEventDataSchema)
    da = read_file(file_path)

    with snx.File(file_path) as file:
        raw_data: sc.Variable = file[entry_path]["events"][()]

    weights = raw_data['dim_1', 0].copy()
    weights.unit = '1'
    expected_data = (DefaultMaximumProbability / weights.max()) * weights

    assert isinstance(da, sc.DataArray)
    assert list(da.values) == list(expected_data.values)
    assert list(da.coords['x'].values) == list(raw_data['dim_1', 1].values)
    assert list(da.coords['y'].values) == list(raw_data['dim_1', 2].values)
    assert list(da.coords['id'].values) == list(raw_data['dim_1', 4].values)
    assert list(da.coords['t'].values) == list(raw_data['dim_1', 5].values)
    assert da.unit == '1'
    assert da.coords['x'].unit == 'm'
    assert da.coords['y'].unit == 'm'
    assert da.coords['id'].unit == 'dimensionless'
    assert da.coords['t'].unit == 's'
