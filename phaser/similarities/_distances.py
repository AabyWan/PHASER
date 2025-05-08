"""
Module to define custom distances not already present in scipy.spatial.distance
Distance metrics should return a normalised value between 0 and 1, and take two vectors (u,v by convention in scipy), and possibly other kwargs, as input.
"""

# Imports for test synthtic distance generator
import sys
import numpy as np
from math import sqrt
from scipy.ndimage import convolve
import scipy.spatial.distance


# numbas implementations
from fastdist import fastdist
from numba import jit

# for testing synthetic metric
import matplotlib.pyplot as plt 
from scipy.stats import binom

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


# Experimental Distances

def ngram_cosine_distance(u, v, w=None, ngram_size=2, rounding=4, discard=0):
    """ NGram Cosine Distance. Create a square array from input hashes u and v. Extract square ngrams (size ngram_size) from each.
    The array of ngrams is then flattened and compared using Cosine distance to generate a measure of hash similarity.

    Args:
        u (ndarray): The first hash to compare.
        v (ndarray): The second hash to compare.
        w (ndarray, optional): NOT USED in this implementation. Defaults to None.
        ngram_size (int, optional): The size of the ngrams to generate. Defaults to 2.
        rounding (int, optional): Number of decimal places to round to. Defaults to 4.
        discard (int, optional): percentage of highest distance to discard. Defaults to 0.
    """    
        
    def generate_2d_ngrams(array, ngram_size):
        """ Generate a list of 2D ngrams from a given matrix.
        Args:
            array (ndarray): The matrix to generate ngrams from.
            ngram_size (int): The size of the ngrams - assumes square ngrams.

        Returns:
            ndarray: The array of ngrams.
        """
        sqrlength = int(sqrt(len(array)))
        matrix = array.reshape(sqrlength, sqrlength)
        rows, cols = matrix.shape
        ngrams = []

        # Iterate over each possible starting point for an ngram
        for row in range(rows - ngram_size + 1):
            for col in range(cols - ngram_size + 1):
                # Extract the ngram from the matrix
                ngram = matrix[row:row+ngram_size, col:col+ngram_size]
                ngrams.append(ngram)

        return np.array(ngrams)
    
    # Generate ngrams
    ngrams1 = generate_2d_ngrams(u, ngram_size=ngram_size)
    ngrams2 = generate_2d_ngrams(v, ngram_size=ngram_size)
    
    return round(float(scipy.spatial.distance.cosine(ngrams1.flatten(), ngrams2.flatten())), rounding)



def convolution_distance(u, v, w=None, max_value=None, mode='sum_diffs', cmode='constant', cval=0, XOR=True,
                         filter = np.array([[1, 1, 1],
                                            [1, 1, 1],
                                            [1, 1, 1]])
                         ):
    """Convolution-based hash distance metric, useful for hashes which encode positional information in their hash. 
    XOR the inputs u,v, and then create a square difference matrix from the result. This is then convolved and normalised by the maximum convolution matrix for the given array size and filter.

    Args:
        u (ndarray): The first hash to compare.
        v (ndarray): The second hash to compare.
        w (ndarray, optional): Array of bit-weights. This is applied to the difference/similarity array of the hashes prior to convolution. Defaults to None.
        max_value (_type_): Used for normalisation. Should ideally be the maximum value produced by a matrix of the hash size with the passed kernel filter.
        mode (str, optional): {"sum_diffs", "sum_similar"} Select wheether to convolve a matrix of the differences between u and v, or the same bits. (i.e. select between XOR and NOT-XOR for passing to the convolution) Defaults to "sum_diffs".
        conv_filter (ndarray), optional): Kernel filter weights. Defaults to numpy.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).

    Returns:
        float: Convolution sum distance of the difference between the two arrays
    """
    
    # Check mode
    if mode not in ["sum_diffs", "sum_similar"]:
        raise(Exception(f"Mode must be either 'sum_diffs' or 'sum_similar'"))
    
    # Assume squarable hash length
    sqr_size = int(sqrt(len(u)))
    if XOR:
        diff = np.logical_xor(u,v).astype(int)
    else:
        diff = np.array([0 if x == y else 1 for x, y in zip(u, v)])

    if mode == 'sum_similar':
        # Mode set to sum similar bits - perform a logical NOT of the difference vector to achieve this.
        diff = np.logical_not(diff)
        
    if w is not None:
        # weights are provided, apply them to the difference vector.
        w = w / sum(w)
        diff = diff * w
        
    # Create 2d matrix from array
    diff2d = np.asarray(diff).reshape(sqr_size,sqr_size)
    # Calculate concolution matrix (with input filter or default) and sum
    conv_sum  = np.sum(convolve(diff2d, filter, mode=cmode, cval=cval))
    
    if max_value is None:
        # Calculate possible maximum value. Ideally this wouldn't be done every time, but it changes with filter and hash length.
        ones = np.ones((sqr_size, sqr_size))
        max_value = np.sum(convolve(ones, filter,  mode=cmode, cval=cval))
        
    # Normalise the sum to a value between 0 and 1. max_value required for this reason.
    return round(conv_sum / max_value, 4)



def hatched_matrix(u, v, w=None, distance_fun=scipy.spatial.distance.cosine):
    """Hatched Matrix Distance for comparing DCT-based hashes.
    Performs pairwise distance_fun analysis of each odd/even row/column pair between input hashes u,v (after converting to square matrices).
    Rows and columns are then compared separately. The mean value of odd/even rows is compared, taking the minimum, then the same for columns.
    The mean of the column/row distance then becomes the final distance value, which is returned.
    The idea is to capture a balance between row and column distances across both hashes, while biasing towards the rows/columns with the lowest distances.

    Args:
        u (ndarray): The first hash to compare.
        v (ndarray): The second hash to compare.
        w (ndarray, optional): Array of bit-weights. Applied to u,v prior to extracting DCT components.. Defaults to None.
        distance_fun (Callable, optional): The distance metric to use when comparing extracted hatched matrix components. Defaults to scipy.spatial.distance.cosine.

    Returns:
        Float: The distance value, balancing the row/column hatched patterns generated by DCT-based hashes. 

    """
    def split_DCT(matrix):
        cols = [matrix[:,x] for x in range(matrix.shape[0])]
        rows = [matrix[x] for x in range(matrix.shape[1])]
        return {"cols": cols, "rows": rows}
    
    # Apply weights
    if w is not None:
        w = w / sum(w)
        u = u * w
        v = u * w
    
    # Assume squarable hash length
    sqr_size = int(sqrt(len(u)))
    usplit = split_DCT(u.reshape(sqr_size ,sqr_size))
    vsplit = split_DCT(v.reshape(sqr_size ,sqr_size))
    
    # compare rows
    rows_even, rows_odd, cols_even, cols_odd = [], [], [], []
    
    for index, value in enumerate(usplit["rows"]):
        if index % 2 == 0:
            rows_even.append(distance_fun(usplit["rows"][index], vsplit["rows"][index]))
        else:
            rows_odd.append(distance_fun(usplit["rows"][index], vsplit["rows"][index]))
    # compare cols
    for index, value in enumerate(usplit["cols"]):
        if index % 2 == 0:
            cols_even.append(distance_fun(usplit["cols"][index], vsplit["cols"][index]))
        else:
            cols_odd.append(distance_fun(usplit["cols"][index], vsplit["cols"][index]))

    # print("rows_even:",rows_even,"rows_odd:", rows_odd)
    # print("cols_even:",cols_even,"cols_odd:", cols_odd)
    
    rowval = min(np.mean(rows_even), np.mean(rows_odd))
    colval = min(np.mean(cols_even), np.mean(cols_odd))
    
    return round(np.mean([rowval, colval]), 4)

def hatched_matrix2(u, v, w=None,  distance_fun=scipy.spatial.distance.cosine):
    """Hatched Matrix Distance 2 for comparing DCT-based hashes. This is the version described in the 2025 paper: https://doi.org/10.1016/j.fsidi.2025.301878
    Similar to the first implementation, except the odd/even rows/columns are accumulated into their own lists first, before being compared with the specified distance function.
    The mean of the column/row distance then becomes the final distance value, which is returned.
    The idea is to capture a balance between row and column distances across both hashes, while biasing towards the rows/columns with the lowest distances.

    Args:
        u (ndarray): The first hash to compare.
        v (ndarray): The second hash to compare.
        w (ndarray, optional): Array of bit-weights. Applied to u,v prior to extracting DCT components.. Defaults to None.
        distance_fun (Callable, optional): The distance metric to use when comparing extracted hatched matrix components. Defaults to scipy.spatial.distance.cosine.

    Returns:
        Float: The distance value, balancing the row/column hatched patterns generated by DCT-based hashes. 
    """
    def split_DCT(matrix):
        cols = [matrix[:,x] for x in range(matrix.shape[0])]
        rows = [matrix[x] for x in range(matrix.shape[1])]
        return {"cols": cols, "rows": rows}
    
    # Apply weights
    if w is not None:
        w = w / sum(w)
        u = u * w
        v = u * w
    
    # Assume squarable hash length
    sqr_size = int(sqrt(len(u)))
    usplit = split_DCT(u.reshape(sqr_size ,sqr_size))
    vsplit = split_DCT(v.reshape(sqr_size ,sqr_size))
    
    # compare rows
    rows_even, rows_odd, cols_even, cols_odd = [[],[]], [[],[]], [[],[]], [[],[]]
    
    # concat all even rows/cols for each matrix
    for index, value in enumerate(usplit["rows"]):
        if index % 2 == 0:
            rows_even[0].extend(usplit["rows"][index])
            cols_even[0].extend(usplit["cols"][index])
            rows_even[1].extend(vsplit["rows"][index])
            cols_even[1].extend(vsplit["cols"][index])
        else:
            rows_odd[0].extend(usplit["rows"][index])
            cols_odd[0].extend(usplit["cols"][index])
            rows_odd[1].extend(vsplit["rows"][index])
            cols_odd[1].extend(vsplit["cols"][index])
    
    # print("rows_even:",np.array(rows_odd).flatten(),"rows_odd:", np.array(rows_even).flatten())
    # print("cols_even:",np.array(cols_odd).flatten(),"cols_odd:", np.array(cols_even).flatten())

    rowval = min(distance_fun(rows_even[0], rows_even[1]), distance_fun(rows_odd[0], rows_odd[1]))
    colval = min(distance_fun(cols_even[0], cols_even[1]), distance_fun(cols_odd[0], cols_odd[1]))
    
    
    return round(np.mean([rowval, colval]), 4)

@jit(fastmath=True, cache=True)
def hatched_matrix_fast(u, v, dist_fun=fastdist.hamming):
    """A fast implementation of Hatched Matric Distance, see that for details.
    Defaults to fastdist.hamming (cosine sometimes generates zero division errors), and uses fastmath/jit to pre-compile the function for speed.
    Otherwise, the details are the same, though it's more unstable overall and compilation requires a bit of time upfront.
    """
    
    if np.array_equal(u, v): # they're the same, return 0
        return 0.0

    def split_DCT(matrix):
        rows = matrix
        cols = matrix.T
        return {"cols": cols, "rows": rows}
    
    # Apply weights
    # w = w / sum(w)
    # u = u * w
    # v = u * w

    # Assume squarable hash length
    sqr_size = int(sqrt(len(u)))
    usplit = split_DCT(u.reshape(sqr_size ,sqr_size))
    vsplit = split_DCT(v.reshape(sqr_size ,sqr_size))
    
    # compare rows
    rows_even:list[float] = []    
    rows_odd:list[float] = []    
    cols_even:list[float] = []    
    cols_odd:list[float] = []    

    
    for index, value in enumerate(usplit["rows"]):
        if index % 2 == 0:
            rows_even.append(dist_fun(usplit["rows"][index], vsplit["rows"][index]))
            cols_even.append(dist_fun(usplit["cols"][index], vsplit["cols"][index]))
        else:
            rows_odd.append(dist_fun(usplit["rows"][index], vsplit["rows"][index]))
            cols_odd.append(dist_fun(usplit["cols"][index], vsplit["cols"][index]))
    
    # print("rows_even:",np.array(rows_odd).flatten(),"rows_odd:", np.array(rows_even).flatten())
    # print("cols_even:",np.array(cols_odd).flatten(),"cols_odd:", np.array(cols_even).flatten())
    # return None
    rowval = min(sum(rows_even)/len(rows_even), sum(rows_odd)/len(rows_odd))
    colval = min(sum(cols_even)/len(cols_odd), sum(cols_odd)/len(cols_odd))
    
    return (rowval+colval)/2
    

# Keep track of distance metrics, add here and import in __init__.py when creating new distance metrics.
DISTANCE_METRICS = [test_synthetic.__name__, convolution_distance.__name__, 
                    hatched_matrix.__name__, hatched_matrix2.__name__, hatched_matrix_fast.__name__,
                    ngram_cosine_distance.__name__]



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
