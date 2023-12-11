# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from .detector import default_params as detector_default_params
from .loader import read_file
from .reduction import get_grouped_by_pixel_id, get_ids, get_time_binned

providers = (read_file, get_ids, get_grouped_by_pixel_id, get_time_binned)


def collect_default_parameters() -> dict:
    return dict(detector_default_params)
