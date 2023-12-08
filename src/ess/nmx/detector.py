# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from typing import NewType

MaxNumberOfPixelsPerAxis = NewType("MaxNumberOfPixelsPerAxis", int)
PixelStep = NewType("PixelStep", int)
NumberOfDetectors = NewType("NumberOfDetectors", int)
NumberOfAxes = NewType("NumberOfAxes", int)


@dataclass
class InstrumentInfo:
    number_of_pixels: MaxNumberOfPixelsPerAxis
    pixel_step: PixelStep
    number_of_detectors: NumberOfDetectors
    number_of_axis: NumberOfAxes

    @property
    def output_data_points(self):
        return self.number_of_pixels**self.number_of_axis * self.number_of_detectors


default_params = {
    MaxNumberOfPixelsPerAxis: MaxNumberOfPixelsPerAxis(1280),
    NumberOfAxes: NumberOfAxes(2),
    PixelStep: PixelStep(1),
    NumberOfDetectors: NumberOfDetectors(3),
}
