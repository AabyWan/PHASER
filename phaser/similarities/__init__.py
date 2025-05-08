"""

The :mod:`phaser.similarities` module includes various ...
"""


from ._helpers import (
    IntraDistance,
    InterDistance,
    find_inter_samplesize,
    validate_metrics,
)


from ._distances import DISTANCE_METRICS, test_synthetic, convolution_distance, hatched_matrix, hatched_matrix2, ngram_cosine_distance, hatched_matrix_fast

__all__ = [
    "IntraDistance",
    "InterDistance",
    "find_inter_samplesize",
    "validate_metrics",
    "DISTANCE_METRICS",
    "test_synthetic",
    "convolution_distance",
    "hatched_matrix",
    "hatched_matrix2",
    "ngram_cosine_distance",
    "hatched_matrix_fast"
]
