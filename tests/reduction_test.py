# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ess.nmx.reduction import _zip_and_group


def test_zip_and_group_str() -> None:
    da = sc.DataArray(
        data=sc.ones(dims=["xy"], shape=[6]),
        coords={
            "x": sc.array(dims=["xy"], values=[1, 1, 2, 2, 3, 3]),
            "y": sc.array(dims=["xy"], values=[0, 1, 2, 2, 0, 3]),
        },
    )
    var_xy = sc.array(dims=["xy"], values=["1 0", "1 1", "2 2", "3 0", "3 3"])

    grouped = _zip_and_group(da, "x", "y")
    assert sc.identical(grouped.coords["xy"], var_xy)


def test_zip_and_group_variables_all_possibilities() -> None:
    da = sc.DataArray(
        data=sc.ones(dims=["xy"], shape=[6]),
        coords={
            "x": sc.array(dims=["xy"], values=[1, 1, 2, 2, 3, 3]),
            "y": sc.array(dims=["xy"], values=[0, 1, 2, 2, 0, 3]),
        },
    )

    var_x = sc.array(dims=["x"], values=[1, 1, 2, 3, 3], unit=None)
    var_y = sc.array(dims=["y"], values=[0, 1, 2, 0, 3], unit=None)
    var_xy = sc.array(dims=["xy"], values=["1 0", "1 1", "2 2", "3 0", "3 3"])

    grouped = _zip_and_group(da, var_x, var_y)
    assert sc.identical(grouped.coords["xy"], var_xy)


def test_zip_and_group_variables_less_groups() -> None:
    da = sc.DataArray(
        data=sc.ones(dims=["xy"], shape=[6]),
        coords={
            "x": sc.array(dims=["xy"], values=[1, 1, 2, 2, 3, 3]),
            "y": sc.array(dims=["xy"], values=[0, 1, 2, 2, 0, 3]),
        },
    )

    var_x = sc.array(dims=["x"], values=[1, 1, 3, 3], unit=None)
    var_y = sc.array(dims=["y"], values=[0, 2, 0, 3], unit=None)
    var_xy = sc.array(dims=["xy"], values=["1 0", "1 2", "3 0", "3 3"])

    grouped = _zip_and_group(da, var_x, var_y)
    assert sc.identical(grouped.coords["xy"], var_xy)


def test_zip_and_group_variables_empty_group() -> None:
    da = sc.DataArray(
        data=sc.ones(dims=["xy"], shape=[6]),
        coords={
            "x": sc.array(dims=["xy"], values=[1, 1, 2, 2, 3, 3]),
            "y": sc.array(dims=["xy"], values=[0, 1, 2, 2, 0, 3]),
        },
    )

    var_x = sc.array(dims=["x"], values=[1, 1, 3, 3, 4], unit=None)
    var_y = sc.array(dims=["y"], values=[0, 2, 0, 3, 4], unit=None)
    var_xy = sc.array(dims=["xy"], values=["1 0", "1 2", "3 0", "3 3", "4 4"])

    grouped = _zip_and_group(da, var_x, var_y)
    assert sc.identical(grouped.coords["xy"], var_xy)
    assert grouped[0].values.size == 1
    # last group should be empty
    assert grouped[-1].values.size == 0
