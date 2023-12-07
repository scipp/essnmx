# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc


def test_mcstas_default_data_path() -> None:
    from ess.nmx.loader import DataPath, FileTypeMcStas, get_data_path_mcstas

    assert get_data_path_mcstas() == DataPath(
        entry_path="entry1/data/bank01_events_dat_list_p_x_y_n_id_t",
        event_path="events",
        file_type=FileTypeMcStas,
    )


def test_mcstas_data_path() -> None:
    from ess.nmx.loader import (
        DataPath,
        FileTypeMcStas,
        McStasEventDataSchema,
        get_data_path_mcstas,
    )

    data_schema = McStasEventDataSchema("p_x_y_n_id_t_L_L")

    assert get_data_path_mcstas(data_schema) == DataPath(
        entry_path=f"entry1/data/bank01_events_dat_list_{data_schema}",
        event_path="events",
        file_type=FileTypeMcStas,
    )


def test_mcstas_data_loader() -> None:
    from ess.nmx.data import small_mcstas_sample
    from ess.nmx.loader import get_data_path_mcstas, read_h5file_mcstas

    file_path = small_mcstas_sample()
    data_path = get_data_path_mcstas()
    loaded = read_h5file_mcstas(file_path, data_path)
    import scippnexus as snx

    with snx.File(file_path) as file:
        raw_data = file[data_path.entry_path][data_path.event_path][()]

    assert isinstance(loaded, sc.DataGroup)
    assert list(loaded['weights'].values) == list(raw_data['dim_1', 0].values)
    assert loaded['weights'].unit == 'counts'
    assert list(loaded['x_list'].values) == list(raw_data['dim_1', 1].values)
    assert loaded['x_list'].unit == 'm'
    assert list(loaded['y_list'].values) == list(raw_data['dim_1', 2].values)
    assert loaded['y_list'].unit == 'm'
    assert list(loaded['id_list'].values) == list(raw_data['dim_1', 4].values)
    assert loaded['id_list'].unit == 'dimensionless'
    assert list(loaded['t_list'].values) == list(raw_data['dim_1', 5].values)
    assert loaded['t_list'].unit == 's'


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


def test_mcstas_compute_weights(small_sample_dg: sc.DataGroup) -> None:
    from ess.nmx.loader import MaximumPropability, get_weights_mcstas

    prop = MaximumPropability(1_000)
    expected = (prop / small_sample_dg['weights'].max()) * small_sample_dg['weights']
    assert sc.identical(get_weights_mcstas(small_sample_dg, prop), expected)
