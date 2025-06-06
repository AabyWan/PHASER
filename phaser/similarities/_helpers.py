import logging, pathlib
import numpy as np
import pandas as pd
from ._distances import DISTANCE_METRICS
from scipy.ndimage import convolve
from scipy.spatial import distance as dist
from scipy.spatial.distance import cdist, pdist
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
from tqdm.auto import tqdm
from typing import Callable





pathlib.Path("./logs").mkdir(exist_ok=True)
logging.basicConfig(
    filename="./logs/process.log",
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def find_inter_samplesize(num_images: int) -> int:
    for n in range(0, num_images):
        if (n * (n - 1)) / 2 > num_images:
            return n
    return 0

def validate_metrics(metrics: dict) -> bool:
    """Function to perform light validation of metrics.
    If the distance provided is a string, it should be in scipy.spatial.distance.
    If a custom metric is provided, we at least check to see that it is callable.
    If either of the above conditions are not met, or the metrics list is empty, throw an Exception.
    """
    if not metrics:
        raise (Exception("No distance metrics specified."))

    invalid = []

    for mname, value in metrics.items():
        keyword_args = {}
        if not isinstance(value, str) and not isinstance(value, Callable):
            if len(value) not in [1,2]:
                raise Exception("""Invalid format for distance metric. Should be a callable, or scipy.spatial.distance function name string. 
                                Pass keywords arguments by using a tuple where the second element is a dict of keyword args""")
            if len(value) == 2:
                keyword_args = value[1]
                if type(keyword_args) != dict:
                    raise(Exception("Keyword argument list should be a dict."))
                value = value[0]
            
        if isinstance(value, str):
            if value not in dist._METRICS_NAMES: #type:ignore
                invalid.append(
                    f"{mname} does not match the name of a distance metric in scipy.spatial.distance."
                )
        elif not isinstance(value, Callable):
            invalid.append(f"{mname} is not a valid Callable object.")
        elif isinstance(value, Callable):
            if value.__name__ not in DISTANCE_METRICS:
                invalid.append(
                    f"{mname} does not appear to be a valid distance function in phaser.similarities"
                )
        # Check to see if we can actually call it with these arguments
        if isinstance(value, str):
            func = getattr(dist, value)
            test_result = func(u=np.ones(64,), v=np.ones(64,), **keyword_args)
            if test_result == 0:
                test_result = float(0)
            assert isinstance(test_result, float), f"The function {func.__name__} does not return a float. Got {test_result}."

        else:
            test_result = value(u=np.ones(64,), v=np.ones(64,), **keyword_args)
            if test_result == 0:
                test_result = float(0)
            assert isinstance(test_result, float), f"The function {value.__name__} does not return a float. Got {test_result}."

    # If no invalid items, all entries in the list pass basic check.
    if not invalid:
        return True
    else:
        # There are invalid objects, avoid running the tests.
        message = f"Invalid metrics found:{invalid}"
        raise (Exception(message))

# DISTANCE COMPUTATION
class IntraDistance:
    def __init__(
        self,
        m_dict:dict,
        le:dict, 
        set_class=1, 
        bit_weights=None, # would expect a dicitonary
        progress_bar=False):
        #
        self.le = le 
        self.bit_weights = bit_weights
        self.m_dict = m_dict
        self.set_class = set_class
        self.progress_bar = progress_bar

        validate_metrics(self.m_dict)
        
    def _intradistance(self, x, algorithm, metric, weights):
        # store the first hash and reshape into 2d array as required by cdist func.
        xa = x[algorithm].iloc[0].reshape(1, -1)

        # row stack the other hashes
        xb = x.iloc[1:][algorithm].values
        xb = np.row_stack(xb)

        # Get the vlaue corresponding to the metric key.
        # This is either a string representing a name from scipy.spatial.distances
        # or a callable function implementing another metric.
        metric_value = self.m_dict[metric]
        keyword_arguments = {}
        
        if not isinstance(metric_value, Callable):
            if len(metric_value) == 2:
                keyword_arguments = metric_value[1] # The keyword dict
                metric_value = metric_value[0] # The function
        
        return cdist(xa, xb, metric=metric_value, w=weights, **keyword_arguments)

    def fit(self, data):
        logging.info("===Begin processing Intra-Distance.===")
        self.files_ = data["filename"].unique()
        self.n_files_ = len(self.files_)

        distances = []

        for a in tqdm(self.le['a'].classes_, disable=not self.progress_bar, desc="Hash"):
            for m in self.le['m'].classes_:
                # Check if metric is specified in the metric dictionary.
                if m in self.m_dict:
                
                    if self.bit_weights:
                        w = self.bit_weights[f"{a}_{m}"]
                    else: w=None
                    
                    # Compute the distances for each filename
                    grp_dists = data.groupby(["filename"]).apply(
                        self._intradistance, 
                        algorithm=a, 
                        metric=m,
                        weights=w
                    )

                    # Stack each distance into rows
                    # Note: If this throws:
                    # "ValueError: all the input array dimensions except for the concatenation axis must match exactly"
                    # then it's likely that the input list of hashes contains duplicate filenames/rows.
                    grp_dists = np.row_stack(grp_dists)

                    # Get the integer labels for algo and metric
                    a_label = self.le['a'].transform(a.ravel())[0]
                    m_label = self.le['m'].transform(m.ravel())

                    grp_dists = np.column_stack(
                        [
                            self.files_,  # fileA
                            self.files_,  # fileB (same in intra!)
                            np.repeat(a_label, self.n_files_),
                            np.repeat(m_label, self.n_files_),
                            np.repeat(self.set_class, self.n_files_),
                            grp_dists,
                        ]
                    )
                    distances.append(grp_dists)

        distances = np.concatenate(distances)

        # Create the dataframe output
        
        # Get the order of transforms from the hash dataframe
        num_transforms = len(self.le['t'].classes_)
        transform_order =  list(data['transformation'][:num_transforms])
        transform_classes = list(self.le['t'].inverse_transform(transform_order))
        transform_classes.remove("orig")
        
        cols = ["fileA", "fileB", "algo", "metric", "class", *transform_classes]
        distances = pd.DataFrame(distances, columns=cols)
        
        # Insert originals in the correct place (according to hash dataframe)
        # Get index of class column and add 5 (to account for filea, fileb, algo, metric, class) 
        distances.insert(loc=self.le['t'].transform(["orig"])[0]+5, column='orig', value=0)        
        
   

        # set int columns accordingly
        int_cols = cols[:5]
        distances[int_cols] = distances[int_cols].astype(int)

        # Convert distances to similarities
        sim_cols = distances.columns[5:]
        distances[sim_cols] = 1 - distances[sim_cols]

        logging.info(f"Generated {len(distances)} Intra-distance observations.")

        return distances

class InterDistance:
    def __init__(
        self,
        m_dict:dict,
        le:dict,
        set_class=0,
        bit_weights=None,
        n_samples=100,
        random_state=42,
        progress_bar=False,
    ):
        self.le = le
        self.bit_weights = bit_weights
        self.m_dict = m_dict
        self.set_class = set_class
        self.n_samples = n_samples
        self.random_state = random_state
        self.progress_bar = progress_bar

        validate_metrics(self.m_dict)

    def _interdistance(self, x, algorithm, metric, weights):
        # get hashes into a 2d array
        hashes = np.row_stack(x[algorithm])

        # Get the vlaue corresponding to the metric key.
        # This is either a string representing a name from scipy.spatial.distances
        # or a callable function implementing another metric.
        metric_value = self.m_dict[metric]
        keyword_arguments = {}
        if not isinstance(metric_value, Callable):
            if len(metric_value) == 2:
                keyword_arguments = metric_value[1] # The keyword dict
                metric_value = metric_value[0] # The function

        # return pairwise distances of all combinations
        return pdist(hashes, metric=metric_value, w=weights, **keyword_arguments)

    def fit(self, data):
        logging.info(
            f"===Begin processing Inter-Distance with {self.n_samples} pairwise samples per file.==="
        )
                
        # Get the label used to encode 'orig'
        orig_label = self.le['t'].transform(np.array(["orig"]).ravel())[0]

        # Assert sufficient data to sample from.
        assert len(data[data["transformation"] == orig_label]) >= self.n_samples

        # Pick the samples
        self.samples_ = (
            data[data["transformation"] == orig_label]
            .sample(self.n_samples, random_state=self.random_state)["filename"]
            .values
        )

        # Subset the data
        subset = data[data["filename"].isin(self.samples_)]

        # Create unique pairs matching the output of scipy.spatial.distances.pdist
        self.pairs_ = np.array(
            [c for c in combinations(subset["filename"].unique(), 2)]
        )

        # Count the number of unique pairs
        self.n_pairs_ = len(self.pairs_)

        # List to hold distances while looping over algorithms and metrics
        distances = []

        # Do the math using Pandas groupby
        for a in tqdm(self.le['a'].classes_, disable=not self.progress_bar, desc="Hash"):
            for m in self.le['m'].classes_:
                # Check if metric is specified in the metric dictionary.
                if m in self.m_dict:

                    if self.bit_weights:
                        w = self.bit_weights[f"{a}_{m}"]
                    else: w=None
                    
                    # Compute distances for each group of transformations
                    grp_dists = subset.groupby(["transformation"]).apply(
                        self._interdistance,  # type:ignore
                        algorithm=a,
                        metric=m,
                        weights=w
                    )

                    # Transpose to create rows of observations
                    X_dists = np.transpose(np.row_stack(grp_dists.values))

                    # Get the integer labels for algo and metric
                    a_label = self.le['a'].transform(a.ravel())[0]
                    m_label = self.le['m'].transform(m.ravel())

                    # Add columns with pairs of the compared observations
                    X_dists = np.column_stack(
                        [
                            self.pairs_,
                            np.repeat(a_label, self.n_pairs_),
                            np.repeat(m_label, self.n_pairs_),
                            np.repeat(self.set_class, self.n_pairs_),
                            X_dists,
                        ]
                    )

                    # Add the results to the distances array
                    distances.append(X_dists)

        # Flatten the distances array
        distances = np.concatenate(distances)

        # Create the dataframe output
        
        # No need to re-order these.
        cols = ["fileA", "fileB", "algo", "metric", "class", *self.le['t'].classes_]
        distances = pd.DataFrame(distances, columns=cols)

        # Set datatype to int on all non-distance columns
        int_cols = cols[:5]
        distances[int_cols] = distances[int_cols].astype(int)

        # Convert distances to similarities
        sim_cols = distances.columns[5:]
        distances[sim_cols] = 1 - distances[sim_cols]

        logging.info(f"Generated {len(distances)} Inter-distance observations.")

        return distances


def max_convolution_value(matrix_dimensions, filter):
    """ Calculate the maximum value for a convolution using a matrix of 1s of size matrix_dimensions, using a specified filter.
    """
    m = np.ones(matrix_dimensions, dtype=int)
    sum = np.sum(convolve(m, filter, mode='constant', cval=0))
    return sum