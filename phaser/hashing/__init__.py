"""
The :mod:`phaser.hashing` module includes various utilities.

H1 inside module contents
=========================
Add some important text here to describe or link to examples.

"""

from ._algorithms import PerceptualHash, PHash, ColourHash, WaveHash, PdqHash, AverageHash, DifferenceHash, NeuralHash

from ._helpers import ComputeHashes

# Include names of private functions to autodoc
__all__ = [
    "AverageHash",
    "DifferenceHash",
    "PerceptualHash",
    "PHash",
    "ComputeHashes",
    "ColourHash",
    "WaveHash",
    "PdqHash",
    "NeuralHash",
]
