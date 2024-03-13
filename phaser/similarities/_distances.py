"""
Module to define custom distances not already present in scipt.spatial.distance.
Distance metrics should return a normalised value between 0 and 1, and take two vectors (u,v by convention in scipy), and possibly other kwargs, as input.
"""

# Imports for test synthtic distance generator
import sys
import numpy as np

# for testing synthetic metric
import matplotlib.pyplot as plt 


def test_synthetic(u=None, v=None, w=None):
    """
    A dummy distance metric to simulate "good" performance.
    Performs caller introspection to determine which distribution to draw from.
    """

    # /Slightly/ hacky caller introspection
    caller_name = sys._getframe().f_back.f_code.co_name

    # In this library:
    # intra-distance are measured using scipy.spatial.distance.cdist
    # inter-distance are measured using scipy.spatial.distance.pdist
    # Check if we are doing intra-distances and draw from a pareto distribution
    # Otherwise, draw from a normal distribution

    if "cdist" in caller_name.lower():
        # Assume Intra-distance, pareto distribution largely falling near 0
        # Simulate good matching with some variance
        a = 10
        m = 0.1
        return np.random.pareto(a) * m

    else:
        # Simulate a tight normal distribution around 0.5
        mu = 0.5
        sigma = 0.05
        rng = np.random.default_rng()
        return rng.normal(mu, sigma)


# Keep track of distance metrics, add here and import in __init__.py when creating new distance metrics.
DISTANCE_METRICS = [test_synthetic.__name__]



# Demo code to test the synthetic test metric. 
# It ignores data and draw from different distributions depending on the caller function.
def pdist_test():
    nums = []
    for n in range(10000):
        nums.append(test_synthetic())

    plt.xlim(0,1)
    plt.hist(nums, bins=50)
    plt.show()

def cdist_test():
    nums = []
    for n in range(10000):
        nums.append(test_synthetic())

    plt.xlim(0,1)
    plt.hist(nums, bins=50)
    plt.show()

def synthetic_demo():
    pdist_test() # simulate inter-distances
    cdist_test() # simulate intra-distances

if __name__ == "__main__":
    # Demo the synthetic metric
    synthetic_demo()
