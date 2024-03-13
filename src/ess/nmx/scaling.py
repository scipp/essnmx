# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import NewType, Optional

import gemmi
import numpy as np
import pandas as pd
import sciline as sl

from .mtz_io import FileName, RawMtz

# User defined or configurable types
LambdaBinSize = NewType("LambdaBinSize", int)
SpaceGroupDesc = NewType("SpaceGroupDesc", str)
DEFAULT_SPACE_GROUP_DESC = SpaceGroupDesc("P 21 21 21")
"""The default space group description to use if not found in the mtz files."""
# Column names may vary from file to file
# depending on the software used to generate the mtz file.
IntensityColumnName = NewType("IntensityColumnName", str)
"""The name of the intensity column in the mtz file."""
DEFAULT_INTENSITY_COLUMN_NAME = IntensityColumnName("I")
SigmaIntensityColumnName = NewType("SigmaIntensityColumnName", str)
"""The name of the standard uncertainty (of the intensity) column in the mtz file."""
DEFAULT_SIGMA_INTENSITY_COLUMN_NAME = SigmaIntensityColumnName("SIGI")
WavelengthColumnName = NewType("WavelengthColumnName", str)
"""The name of the wavelength column in the mtz file."""
DEFUAULT_WAVELENGTH_COLUMN_NAME = WavelengthColumnName("LAMBDA")
HKLColumnName = NewType("HKLColumnName", str)
"""The name of the miller indices (HKL) column in the mtz file."""
DEFAULT_HKL_COLUMN_NAME = HKLColumnName("hkl")

# Retrieved or Calculated types
SpaceGroup = NewType("SpaceGroup", gemmi.SpaceGroup)
RapioAsu = NewType("RapioAsu", gemmi.ReciprocalAsu)
NMXMtzDataFrame = NewType("NMXMtzDataFrame", pd.DataFrame)
MergedMtzDataFrame = NewType("MergedMtzDataFrame", pd.DataFrame)
LAMBDABinned = NewType("LAMBDABinned", pd.DataFrame)


def reduce_mtz(mtz: RawMtz) -> NMXMtzDataFrame:
    """Select and derive columns from the original ``MtzDataFrame``.

    Parameters
    ----------
    mtz:
        The raw mtz dataset.

    Returns
    -------
    :
        The new mtz dataframe with derived columns.
        The derived columns are:

    Notes
    -----
    :class:`pandas.DataFrame` is the data structure
    that the rest of the steps are using,
    but :class:`gemmi.Mtz` has :func:`gemmi.Mtz:calculate_d`
    that can derive the ``d`` using ``HKL``.

    """
    from .mtz_io import mtz_to_pandas

    orig_df = mtz_to_pandas(mtz)
    mtz_df = pd.DataFrame()

    # HKL should always be integer.
    mtz_df[["H", "K", "L"]] = orig_df[["H", "K", "L"]].astype(int)
    mtz_df['hkl'] = mtz_df[["H", "K", "L"]].values.tolist()

    def _calculate_d(row: pd.Series) -> float:
        return mtz.get_cell().calculate_d(row['hkl'])

    mtz_df['d'] = mtz_df.apply(
        _calculate_d,
        axis=1,
    )
    mtz_df['resolution'] = (
        1 / mtz_df['d']
    ) ** 2 / 4  # $(2d)^{-2} = \sin^2(\theta)/\lambda^2

    mtz_df['I_div_SIGI'] = orig_df['I'] / orig_df['SIGI']

    return NMXMtzDataFrame(mtz_df)


def merge_mtz_dataframes(
    mtz_dfs: sl.Series[FileName, NMXMtzDataFrame], rapio_asu: RapioAsu, sg: SpaceGroup
) -> MergedMtzDataFrame:
    merged_df = pd.concat(mtz_dfs.values(), ignore_index=True)

    def _rapio_asu_to_asu(row: pd.Series) -> list[int]:
        return rapio_asu.to_asu(row["hkl"], sg.operations())[0]

    merged_df['hkl_eq'] = merged_df.apply(_rapio_asu_to_asu, axis=1)

    return MergedMtzDataFrame(merged_df)


def get_space_group(
    mtzs: sl.Series[FileName, RawMtz],
    spacegroup_desc: Optional[SpaceGroupDesc] = DEFAULT_SPACE_GROUP_DESC,
) -> SpaceGroup:
    '''Retrieves spacegroup from file or uses parameter.

    Manually provided space group description is prioritized over
    space group descriptions found in the mtz files.

    Parameters
    ----------
    mtzs:
        A series of raw mtz datasets.
    spacegroup_desc:
        The space group description to use if not found in the mtz files.

    Returns
    -------
    SpaceGroup
        The space group.

    Raises
    ------
    ValueError
        If multiple or no space groups are found
        but space group description is not provided.

    '''
    space_groups = set(
        sgrp for mtz in mtzs.values() if (sgrp := mtz.spacegroup) is not None
    )
    if spacegroup_desc is not None:
        return gemmi.SpaceGroup(spacegroup_desc)
    elif len(space_groups) > 1:
        raise ValueError(f"Multiple space groups found: {space_groups}")
    elif len(space_groups) == 1:
        return SpaceGroup(space_groups.pop())
    else:
        raise ValueError(
            "No space group found and no space group description provided."
        )


def get_reciprocal_asu(spacegroup: SpaceGroup) -> RapioAsu:
    '''gets spacgroup from file or uses manual addet value'''
    return RapioAsu(gemmi.ReciprocalAsu(spacegroup))


def get_lambda_bin(
    mtz_df: MergedMtzDataFrame,
    lambda_bin_size: LambdaBinSize,
    lambda_column_name: Optional[
        WavelengthColumnName
    ] = DEFUAULT_WAVELENGTH_COLUMN_NAME,
) -> LAMBDABinned:
    """Bin the whole dataset by lambda(wavelength).

    Notes
    -----
        Lambda binning should always be done on the merged dataset.

    """

    lambda_column = mtz_df[lambda_column_name]
    bins = np.linspace(
        lambda_column.min(), lambda_column.max(), lambda_bin_size
    )  # 500 is an example, you can change this, we can optimize this.
    # print("lambda min and mx", mtz_df.LAMBDA.min(), mtz_df.LAMBDA.max())
    labels = ['{:.3}-{:.3}'.format(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
    labels = range(len(bins) - 1)
    mtz_df['LAM_bin'] = pd.cut(mtz_df.LAMBDA, bins=bins, labels=labels)
    return LAMBDABinned(mtz_df)


ReferenceGroup = NewType("ReferenceGroup", pd.DataFrame)
ReferenceNumber = NewType("ReferenceNumber", int)
"""TODO: Write a function to find the middle number of the group."""


def get_reference_group(
    lambda_binned: LAMBDABinned, reference_number: ReferenceNumber
) -> ReferenceGroup:
    grouped = lambda_binned.groupby('LAM_bin')
    ref = grouped.get_group(reference_number)  # -> DataFrame
    # for group_name, group_data in grouped:
    # Perform comparisons within each group
    # print(f"Group: {group_name}")
    # print("reflexes per bin",len(ref))
    # print(len(grouped),len(bins))
    return ReferenceGroup(ref)


def get_lambda_scale(lambda_binned: LAMBDABinned, ref_gr: ReferenceGroup):
    """Scan through the reflexes and find the same reflexes
    in the reference and test data set.

    scale_tmp: intensity of the test dataset divided by the intensity
    of the reference dataset for a specific reflex

    scale_sig_tmp: the same as scale_tmp
    but each intensity divided by its sigma(standard uncertainty).
    """
    test_nr = 10  # index of the data array to compare with the reference
    test = lambda_binned.get_group(test_nr)

    scale_tmp = []
    scale_sig_tmp = []
    for i in range(len(ref_gr)):
        for j in range(len(test)):
            # print(ref['hkl_eq'].iloc[i],  test['hkl_eq'].iloc[j])
            if ref_gr['hkl_eq'].iloc[i] == test['hkl_eq'].iloc[j]:
                # print(
                # "ref:",ref['I'].iloc[i],"test:",test['I'].iloc[j],
                # "It/Ir:",(test['I'].iloc[j])/(ref['I'].iloc[i]),"(It/sigt)/(Ir/sigr):",
                # (test['I'].iloc[j]/test['SIGI'].iloc[j])/(ref['I'].iloc[i]/ref['SIGI'].iloc[i]))
                # print("equal reflex found")
                if (
                    test['I'].iloc[j] > 0
                    and test['SIGI'].iloc[j] > 0
                    and ref_gr['I'].iloc[i] > 0
                    and ref_gr['SIGI'].iloc[i] > 0
                ):
                    scale_tmp.append((test['I'].iloc[j]) / (ref_gr['I'].iloc[i]))
                    scale_sig_tmp.append(
                        (test['I'].iloc[j] / test['SIGI'].iloc[j])
                        / (ref_gr['I'].iloc[i] / ref_gr['SIGI'].iloc[i])
                    )

    def calculate_average(lst):
        if not lst:  # Check if the list is empty
            return None  # Return None for an empty list
        return sum(lst) / len(lst)

    print("scale_tmp", scale_tmp)
    print("scale_sig_tmp", scale_sig_tmp)
    scale = calculate_average(scale_tmp)
    scale_sig = calculate_average(scale_sig_tmp)
    print("scale and sacale_sig", scale, scale_sig)
    return scale, scale_sig
