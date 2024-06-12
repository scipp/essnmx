# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pathlib

import gemmi
import pytest
import scipp as sc

from ess.nmx.data import get_small_mtz_samples
from ess.nmx.mtz_io import DEFAULT_SPACE_GROUP_DESC  # P 21 21 21
from ess.nmx.mtz_io import (
    MergedMtzDataFrame,
    MTZFileIndex,
    MTZFilePath,
    NMXMtzDataArray,
    NMXMtzDataFrame,
    RawMtz,
    get_reciprocal_asu,
    get_space_group,
    merge_mtz_dataframes,
    mtz_to_pandas,
    nmx_mtz_dataframe_to_scipp_dataarray,
    process_mtz_dataframe,
    process_single_mtz_to_dataframe,
    read_mtz_file,
)


@pytest.fixture(params=get_small_mtz_samples())
def file_path(request) -> pathlib.Path:
    return request.param


def test_gemmi_mtz(file_path: pathlib.Path) -> None:
    mtz = read_mtz_file(MTZFilePath(file_path))
    assert mtz.spacegroup == gemmi.SpaceGroup("C 1 2 1")  # Hard-coded value
    assert len(mtz.columns[0]) == 100  # Number of samples, hard-coded value


@pytest.fixture
def gemmi_mtz_object(file_path: pathlib.Path) -> gemmi.Mtz:
    return read_mtz_file(MTZFilePath(file_path))


def test_mtz_to_pandas_dataframe(gemmi_mtz_object: gemmi.Mtz) -> None:
    df = mtz_to_pandas(gemmi_mtz_object)
    assert set(df.columns) == set(gemmi_mtz_object.column_labels())
    # Check if the test data are not all-same
    first_column_name, second_column_name = df.columns[0:2]
    assert not all(df[first_column_name] == df[second_column_name])

    # Check if the data are the same
    for column in gemmi_mtz_object.columns:
        assert column.label in df.columns
        assert all(df[column.label] == column.array)


def test_mtz_to_process_pandas_dataframe(gemmi_mtz_object: gemmi.Mtz) -> None:
    df = process_single_mtz_to_dataframe(RawMtz(gemmi_mtz_object))
    for expected_colum in ["hkl", "d", "resolution", *"HKL", "wavelength", "I", "SIGI"]:
        assert expected_colum in df.columns

    for hkl_column in "HKL":
        assert hkl_column in df.columns
        assert df[hkl_column].dtype == int

    assert "hkl_asu" not in df.columns  # It should be done on merged dataframes


@pytest.fixture
def mtz_list() -> list[RawMtz]:
    return [
        read_mtz_file(MTZFilePath(file_path)) for file_path in get_small_mtz_samples()
    ]


def test_get_space_group(mtz_list: list[RawMtz]) -> None:
    assert (
        get_space_group(mtz_list).short_name() == "C2"
    )  # Expected value in test files


def test_get_space_group_with_spacegroup_desc(
    mtz_list: list[RawMtz],
) -> None:
    assert get_space_group(mtz_list, DEFAULT_SPACE_GROUP_DESC).short_name() == "P212121"


@pytest.fixture
def conflicting_mtz_series(
    mtz_list: list[RawMtz],
) -> list[RawMtz]:
    mtz_list[MTZFileIndex(0)].spacegroup = gemmi.SpaceGroup(DEFAULT_SPACE_GROUP_DESC)
    # Make sure the space groups are different
    assert (
        mtz_list[MTZFileIndex(0)].spacegroup.short_name()
        != mtz_list[MTZFileIndex(1)].spacegroup.short_name()
    )

    return mtz_list


def test_get_space_group_conflict_raises(
    conflicting_mtz_series: list[RawMtz],
) -> None:
    reg = r"Multiple space groups found:.+P 21 21 21.+C 1 2 1"
    with pytest.raises(ValueError, match=reg):
        get_space_group(conflicting_mtz_series)


def test_get_space_conflict_but_desc_provided(
    conflicting_mtz_series: list[RawMtz],
) -> None:
    assert (
        get_space_group(conflicting_mtz_series, DEFAULT_SPACE_GROUP_DESC).short_name()
        == "P212121"
    )


@pytest.fixture
def merged_mtz_dataframe(mtz_list: list[RawMtz]) -> MergedMtzDataFrame:
    """Tests if the merged data frame has the expected columns."""
    reduced_mtz = [process_single_mtz_to_dataframe(mtz) for mtz in mtz_list]
    return merge_mtz_dataframes(*reduced_mtz)


@pytest.fixture
def nmx_data_frame(
    mtz_list: list[RawMtz],
    merged_mtz_dataframe: MergedMtzDataFrame,
) -> NMXMtzDataFrame:
    space_gr = get_space_group(mtz_list)
    reciprocal_asu = get_reciprocal_asu(space_gr)

    return process_mtz_dataframe(
        mtz_df=merged_mtz_dataframe,
        reciprocal_asu=reciprocal_asu,
        sg=space_gr,
    )


def test_process_merged_mtz_dataframe(
    merged_mtz_dataframe: MergedMtzDataFrame,
    nmx_data_frame: NMXMtzDataFrame,
) -> None:
    assert "hkl_asu" not in merged_mtz_dataframe.columns
    assert "hkl_asu" in nmx_data_frame.columns


@pytest.fixture
def nmx_data_array(nmx_data_frame: NMXMtzDataFrame) -> NMXMtzDataArray:
    return nmx_mtz_dataframe_to_scipp_dataarray(nmx_data_frame)


def test_to_scipp_dataarray(
    nmx_data_array: NMXMtzDataArray,
) -> None:
    # Check the intended modification
    for indices_coord_name in ("hkl", "hkl_asu"):
        assert nmx_data_array.coords[indices_coord_name].dtype == str

    assert sc.all(nmx_data_array.data > 0)
