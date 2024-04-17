# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import NewType

import scipp as sc

from .mtz_io import DEFAULT_WAVELENGTH_COLUMN_NAME, NMXMtzDataArray
from .reduction import _join_variables, _split_variable

# User defined or configurable types
WavelengthBinSize = NewType("WavelengthBinSize", int)
"""The size of the wavelength(LAMBDA) bins."""
WavelengthEdgeCut = NewType("WavelengthEdgeCut", float)
"""The proportional cut of the wavelength binned data. 0 < proportion < 0.5."""


# Computed types
WavelengthBinned = NewType("WavelengthBinned", sc.DataArray)
"""Binned mtz dataframe by wavelength(LAMBDA) with derived columns."""
ReferenceWavelengthBin = NewType("ReferenceWavelengthBin", sc.DataArray)
"""The reference bin in the binned dataset."""
ReferenceScaleFactor = NewType("ReferenceScaleFactor", sc.DataArray)
"""The reference scale factor, grouped by HKL_EQ."""
ScaleFactorIntensity = NewType("ScaleFactorIntensity", float)
"""The scale factor for intensity."""
ScaleFactorSigmaIntensity = NewType("ScaleFactorSigmaIntensity", float)
"""The scale factor for the standard uncertainty of intensity."""
WavelengthScaled = NewType("WavelengthScaled", sc.DataArray)
"""Scaled wavelength by the reference bin."""
WavelengthScaledTrimmed = NewType("WavelengthScaledTrimmed", sc.DataArray)
"""Scaled wavelength by the reference bin with the edges cut."""


def _is_bin_empty(binned: sc.DataArray, idx: int) -> bool:
    return binned[idx].values.size == 0


def get_lambda_binned(
    mtz_da: NMXMtzDataArray,
    wavelength_bin_size: WavelengthBinSize,
) -> WavelengthBinned:
    """Bin the whole dataset by wavelength(LAMBDA).

    Notes
    -----
        Wavelength(LAMBDA) binning should always be done on the merged dataset.

    """

    return WavelengthBinned(
        mtz_da.bin({DEFAULT_WAVELENGTH_COLUMN_NAME: wavelength_bin_size})
    )


def get_reference_bin(binned: WavelengthBinned) -> ReferenceWavelengthBin:
    """Find the reference group in the binned dataset.

    The reference group is the group in the middle of the binned dataset.
    If the middle group is empty, the function will search for the nearest.

    Parameters
    ----------
    binned:
        The wavelength binned data.

    Raises
    ------
    ValueError:
        If no reference group is found.

    """
    middle_number, offset = len(binned) // 2, 0

    while 0 < (cur_idx := middle_number + offset) < len(binned) and _is_bin_empty(
        binned, cur_idx
    ):
        offset = -offset + 1 if offset <= 0 else -offset

    if _is_bin_empty(binned, cur_idx):
        raise ValueError("No reference group found.")

    return binned[cur_idx].values.copy(deep=False)


def calculate_scale_factor_per_hkl_eq(
    ref_bin: ReferenceWavelengthBin,
) -> ReferenceScaleFactor:
    # Workaround for https://github.com/scipp/scipp/issues/3046
    grouped = ref_bin.group("H_EQ", "K_EQ", "L_EQ").flatten(
        dims=["H_EQ", "K_EQ", "L_EQ"], to="HKL_EQ"
    )
    non_empty = grouped[grouped.bins.size().data > sc.scalar(0, unit=None)]

    return ReferenceScaleFactor((1 / non_empty).bins.mean())


def scale_by_reference_bin(
    binned: WavelengthBinned,
    scale_factor: ReferenceScaleFactor,
    _mtz_da: NMXMtzDataArray,
) -> WavelengthScaled:
    """Scale the intensity by the scale factor.

    Parameters
    ----------
    binned:
        Binned data by wavelength(LAMBDA) to be grouped and scaled.

    scale_factor:
        The scale factor to be used for scaling per HKL group.

    _mtz_da:
        The original mtz data array to get the HKL_EQ coordinates.
        This argument will be removed once the issue in scipp is fixed.

    Returns
    -------
    sc.DataArray
        Scaled intensty by the scale factor.

    """
    # This part is a temporary solution before we fix the issue in scipp.
    # The complicated  be replaced with the scipp group function once it's possible
    # to group by string-type coordinates or ``tuple[int]`` type of coordinates.
    # See https://github.com/scipp/scipp/issues/3046 for more details.
    da = _mtz_da.copy(deep=False)
    hkl_eq_coords = [_mtz_da.coords[f"{coord}_EQ"] for coord in "HKL"]
    da.coords["HKL_EQ"] = _join_variables(*hkl_eq_coords)
    binned = da.bin(
        {DEFAULT_WAVELENGTH_COLUMN_NAME: binned.coords[DEFAULT_WAVELENGTH_COLUMN_NAME]}
    )
    group_coords = tuple(scale_factor.coords[f"{coord}_EQ"] for coord in "HKL")
    group_vars = _join_variables(*group_coords)

    # Grouping by HKL_EQ
    grouped = binned.group(group_vars)
    # Putting back the real coordinates
    # This part should be removed once the issue in scipp is fixed
    # mentioned in the comment above.
    real_coords = _split_variable(grouped.coords["HKL_EQ"])
    for i_coord, name in enumerate([f"{coord}_EQ" for coord in "HKL"]):
        grouped.coords[name] = real_coords[i_coord]

    # Drop variances of the scale factor
    copied_scale_factor = scale_factor.copy(deep=False)
    copied_scale_factor.variances = None
    # Scale each group each bin by the scale factor
    return WavelengthScaled(grouped.bins.mean() * copied_scale_factor)


def cut_edges(
    scaled: WavelengthScaled, edges: WavelengthEdgeCut
) -> WavelengthScaledTrimmed:
    """Cut the edges of the scaled data.

    Parameters
    ----------
    scaled:
        Scaled data to be cut.

    edges:
        The number of edges to be cut.

    Returns
    -------
    sc.DataArray
        The scaled data with the edges cut.

    """
    cutting_index = int(edges * len(scaled))
    return WavelengthScaledTrimmed(
        scaled[DEFAULT_WAVELENGTH_COLUMN_NAME, cutting_index:-cutting_index]
    )


# Providers and default parameters
scaling_providers = (
    get_lambda_binned,
    get_reference_bin,
    calculate_scale_factor_per_hkl_eq,
    scale_by_reference_bin,
    cut_edges,
)
"""Providers for scaling data."""
