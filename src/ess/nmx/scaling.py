# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import NewType

import scipp as sc

from .mtz_io import DEFAULT_WAVELENGTH_COLUMN_NAME, NMXMtzDataArray
from .reduction import _join_variables

# User defined or configurable types
WavelengthBinSize = NewType("WavelengthBinSize", int)
"""The size of the wavelength(LAMBDA) bins."""
WavelengthEdgeCut = NewType("WavelengthEdgeCut", float)
"""The proportional cut of the wavelength binned data. 0 < proportion < 0.5."""
QuadRootStadardDeviationCut = NewType("QuadRootStadardDeviationCut", int)
"""The number of standard deviations to be cut from the 4-th root data."""

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
ScaledIntensity = NewType("WavelengthScaled", sc.DataArray)
"""Scaled intensity by the reference bin."""
ScaledTrimmedIntensity = NewType("ScaledTrimmedIntensity", sc.DataArray)
"""Scaled intensity by the reference bin with the edges cut."""
IntensitySampleVariation = NewType("IntensitySampleVariation", sc.DataArray)
"""The sample variation of the intensity."""
FilteredIntensity = NewType("FilteredIntensity", sc.DataArray)
"""The filtered intensity by the quad root of the sample standard deviation."""
ScaledIntensity = NewType("ScaledIntensity", sc.DataArray)
"""Scaled intensity within the desired certainty."""


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
) -> ScaledIntensity:
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

    # Drop variances of the scale factor
    copied_scale_factor = scale_factor.copy(deep=False)
    copied_scale_factor.variances = None
    # Scale each group each bin by the scale factor
    return ScaledIntensity(grouped.bins.mean() * copied_scale_factor)


def cut_edges(
    scaled: ScaledIntensity, edges: WavelengthEdgeCut
) -> ScaledTrimmedIntensity:
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
    return ScaledTrimmedIntensity(
        scaled[DEFAULT_WAVELENGTH_COLUMN_NAME, cutting_index:-cutting_index]
    )


def _calculate_sample_standard_deviation(var: sc.Variable) -> sc.Variable:
    """Calculate the sample variation of the data.

    This helper function is a temporary solution before
    we release new scipp version with the statistics helper.
    """
    import numpy as np

    return sc.scalar(np.nanstd(var.values))


def _bin_edge_to_midpoint_coord(bin_edges: sc.Variable) -> sc.Variable:
    """Convert the bin edges to the midpoint coordinates."""

    return (bin_edges[1:] + bin_edges[:-1]) / 2


def cut_by_quad_root_sample_std(
    da: ScaledTrimmedIntensity, n_cut: QuadRootStadardDeviationCut
) -> FilteredIntensity:
    """Cut the data by the quad root of the sample standard deviation.

    The data is flattened since the standard deviation cut
    and `nan` filtering may not be aligned with the
    original coordinates, and we do not need `nan` values
    or wavelength/HKL dimension from here.

    Parameters
    ----------
    da:
        The scaled and trimmed data.

    n_cut:
        The number of standard deviations to be kept from mean.

    """
    copied = da.copy(deep=False)

    # Make bin edge coordinate to normal coordinate.
    copied.coords[DEFAULT_WAVELENGTH_COLUMN_NAME] = _bin_edge_to_midpoint_coord(
        copied.coords[DEFAULT_WAVELENGTH_COLUMN_NAME]
    )  # Bin edges are lost when they are flattened.
    flattened = copied.flatten(dims=da.dims, to="row")
    quad_root = flattened.data ** (0.25)

    # Calculate the mean and standard deviation of the quad root
    quad_root.variances = None
    quad_root_mean = quad_root.nanmean()
    quad_root_std = _calculate_sample_standard_deviation(quad_root)
    half_window = n_cut * quad_root_std
    keep_range = (quad_root_mean - half_window, quad_root_mean + half_window)

    # Keep the data within the range
    flattened.coords["keep"] = sc.logical_and(
        quad_root >= keep_range[0], quad_root < keep_range[1]
    )
    keep_or_discard = flattened.group("keep")
    if not sc.any(keep_or_discard.coords['keep']):
        raise ValueError("No data fell into the keeping window. Try wider cut.")

    return FilteredIntensity(
        keep_or_discard["keep", sc.scalar(True)].values.copy(deep=False)
    )


# Providers and default parameters
scaling_providers = (
    get_lambda_binned,
    get_reference_bin,
    calculate_scale_factor_per_hkl_eq,
    scale_by_reference_bin,
    cut_edges,
    cut_by_quad_root_sample_std,
)
"""Providers for scaling data."""
