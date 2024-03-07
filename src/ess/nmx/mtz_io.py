# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""
Should be able to set ``SpaceGroup``,
``Name of the Intensity(I) or Sigma of Intensity(SIGI) or LAMBDA Column``,
``hkl`` should be unviversal... but not sure...
These should "usually" have same names.
It's because the previous step might use different software.
(Users should be able to choose which software to use.)

"""
from typing import NewType, Optional

import gemmi
import numpy as np
import pandas as pd
import sciline as sl

SpaceGroupDesc = NewType("SpaceGroupDesc", str)
DEFAULT_SPACE_GROUP = SpaceGroupDesc("P 21 21 21")
SpaceGroup = NewType("SpaceGroup", gemmi.SpaceGroup)

IntensityColumnName = NewType("IntensityColumnName", str)
SigmaIntensityColumnName = NewType("SigmaIntensityColumnName", str)
LambdaColumnName = NewType("LambdaColumnName", str)
DEFUAULT_LAMBDA_COLUMN = LambdaColumnName("LAMBDA")
LambdaBinSize = NewType("LambdaBinSize", int)

HKLColumnName = NewType("HKLColumnName", str)
RapioAsu = NewType("RapioAsu", gemmi.ReciprocalAsu)

MTZFilepath = NewType("MTZFilepath", str)
RawMtz = NewType("RawMtz", gemmi.Mtz)
MtzDataFrame = NewType("MtzDataFrame", pd.DataFrame)
MergedMtzDataFrame = NewType("MergedMtzDataFrame", pd.DataFrame)


def read_mtz_file(file_path: MTZFilepath) -> RawMtz:
    '''read mtz file'''

    return RawMtz(gemmi.read_mtz_file(file_path))


def mtz_to_pandas(file_path: MTZFilepath) -> MtzDataFrame:
    '''load mtz file into pandas dataframe'''
    mtz = gemmi.read_mtz_file(file_path)
    df = pd.DataFrame(  # Recommended in the gemmi documentation.
        data=np.array(mtz, copy=False), columns=mtz.column_labels()
    )
    df['I_div_SIGI'] = df['I'] / df['SIGI']
    df['d'] = df.apply(
        lambda row: mtz.get_cell().calculate_d(
            [int(row['H']), int(row['K']), int(row['L'])]
        ),
        axis=1,
    )
    df['resolution'] = (1 / df['d']) ** 2 / 4  # $(2d)^{-2} = \sin^2(\theta)/\lambda^2
    df.insert(
        0,
        'hkl',
        df.apply(lambda row: [int(row["H"]), int(row["K"]), int(row["L"])], axis=1),
    )

    return MtzDataFrame(df)


def merge_mtz_dataframes(
    mtz_dfs: sl.Series[MTZFilepath, MtzDataFrame], rapio_asu: RapioAsu, sg: SpaceGroup
) -> MergedMtzDataFrame:
    merged_df = pd.concat(mtz_dfs.values(), ignore_index=True)
    for ignored_column in ['MINHARM', 'MAXHARM', 'NOVPIX', 'XF', 'YF']:
        merged_df = merged_df.drop(ignored_column, axis=1)

    merged_df.insert(
        0,
        'hkl_eq',
        merged_df.apply(
            lambda row: rapio_asu.to_asu(row["hkl"], sg.operations())[0], axis=1
        ),
    )

    return MergedMtzDataFrame(merged_df)


def get_space_group(
    mtzs: sl.Series[MTZFilepath, RawMtz],
    spacegroup_in: Optional[SpaceGroupDesc] = DEFAULT_SPACE_GROUP,
) -> SpaceGroup:
    '''Retrieves spacegroup from file or uses parameter.

    Parameters
    ----------
    mtzs:
        A series of raw mtz datasets.
    spacegroup_in:
        The space group to use if not found in the mtz files.

    Returns
    -------
    SpaceGroup
        The space group.

    Raises
    ------
    ValueError
        If multiple space groups are found.

    '''
    try:
        space_groups = set(mtz.spacegroup for mtz in mtzs.values())
        if len(space_groups) > 1:
            raise ValueError(f"Multiple space groups found: {space_groups}")
        return SpaceGroup(space_groups.pop())

    except AttributeError:
        return gemmi.SpaceGroup(spacegroup_in)


def get_reciprocal_asu(spacegroup: SpaceGroup) -> RapioAsu:
    '''gets spacgroup from file or uses manual addet value'''
    return RapioAsu(gemmi.ReciprocalAsu(spacegroup))


LAMBDABinned = NewType("LAMBDABinned", pd.DataFrame)


def get_lambda_bin(
    mtz_df: MergedMtzDataFrame,
    lambda_bin_size: LambdaBinSize,
    lambda_column_name: Optional[LambdaColumnName] = DEFUAULT_LAMBDA_COLUMN,
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
