# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ..types import (
    EventData,
    MaximumCounts,
    MaximumProbability,
    NMXHistogram,
    RawEventData,
    RawHistogram,
)


def maximum_probability(da: RawEventData) -> MaximumProbability:
    """Find the maximum probability in the data."""
    return MaximumProbability(da.max())


def event_counts_from_probability(
    da: RawHistogram, max_counts: MaximumCounts, max_probability: MaximumProbability
) -> NMXHistogram:
    """Create event weights by scaling probability data.

    event_counts = max_probability * (probabilities / max(probabilities))

    Parameters
    ----------
    da:
        The probabilities of the events

    max_probability:
        The maximum probability to scale the weights.

    """
    return EventData(sc.scalar(max_counts, unit='counts') * da / max_probability)
