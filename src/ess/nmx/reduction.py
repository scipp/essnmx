from typing import NewType

TimeBinStep = NewType("TimeBinStep", int)
DefaultTimeBinStep = TimeBinStep(1)
MaximumPropability = NewType("MaximumPropability", int)
DefaultMaximumPropability = MaximumPropability(100000)
