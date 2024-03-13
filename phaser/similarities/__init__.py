"""

The :mod:`phaser.similarities` module includes various ...
"""


from ._helpers import (
    IntraDistance,
    InterDistance,
    find_inter_samplesize,
    validate_metrics,
)


from ._distances import DISTANCE_METRICS, test_synthetic

__all__ = [
    "IntraDistance",
    "InterDistance",
    "find_inter_samplesize",
    "validate_metrics",
    "DISTANCE_METRICS",
    "test_synthetic",
]
