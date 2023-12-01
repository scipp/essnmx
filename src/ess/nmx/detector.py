from dataclasses import dataclass
from typing import NewType

from scipp.logging import get_logger as get_scipp_logger

# Is the number of pixel always same for all dimensions?
MaxNumberOfPixelsPerAxis = NewType("MaxNumberOfPixelsPerAxis", int)
PixelStep = NewType("PixelStep", int)
NumberOfDetectors = NewType("NumberOfDetectors", int)
NumberOfAxis = NewType("NumberOfAxis", int)
_DefaultNumberOfAxis = NumberOfAxis(2)


@dataclass
class InstrumentInfo:
    number_of_pixels: MaxNumberOfPixelsPerAxis
    pixel_step: PixelStep
    number_of_detectors: NumberOfDetectors
    number_of_axis: NumberOfAxis = _DefaultNumberOfAxis

    @property
    def output_data_points(self):
        # TODO: Check if this is correct
        return self.number_of_pixels**self.number_of_axis * self.number_of_detectors

    def log_output_data_points(self):
        get_scipp_logger().info("Data points in output: %s", self.output_data_points)
