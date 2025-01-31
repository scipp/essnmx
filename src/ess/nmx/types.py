from typing import Any, NewType

import scipp as sc

FilePath = NewType("FilePath", str)
"""File name of a file containing the results of a McStas run"""

DetectorIndex = NewType("DetectorIndex", int | sc.Variable | sc.DataArray)
"""Index of the detector to load. Index ordered by the id:s of the pixels"""

DetectorName = NewType("DetectorName", str)
"""Name of the detector to load"""

DetectorBankPrefix = NewType("DetectorBankPrefix", str)
"""Prefix identifying the event data array containing
the events from the selected detector"""

MaximumCounts = NewType("MaximumCounts", int)
"""Maximum number of counts after scaling the event counts"""

MaximumProbability = NewType("MaximumProbability", float)
"""Maximum probability to scale the McStas event counts"""

RawEventData = NewType("RawEventData", sc.DataArray)
"""DataArray containing the event counts read from the McStas file,
has coordinates 'id' and 't' """

RawEventCounts = NewType("RawEventCounts", sc.Variable)
"""Variable containing the event counts read from the McStas file"""

RawHistogram = NewType("RawHistogram", sc.DataArray)
"""Histogrammed the event counts in the pixel id and time dimensions"""

NMXHistogram = NewType("NMXHistogram", sc.DataArray)
"""Histogrammed event counts in the pixel id and time dimensions"""

NMXRawData = NewType("NMXRawData", sc.DataGroup)
"""DataGroup containing the raw event data with geometry information"""

EventData = NewType("EventData", sc.DataArray)
"""The scaled RawEventData"""

ProtonCharge = NewType("ProtonCharge", sc.Variable)
"""The proton charge signal"""

CrystalRotation = NewType("CrystalRotation", sc.Variable)
"""Rotation of the crystal"""

DetectorGeometry = NewType("DetectorGeometry", Any)
"""Description of the geometry of the detector banks"""

TimeBinSteps = NewType("TimeBinSteps", int)
"""Number of bins in the binning of the time coordinate"""
