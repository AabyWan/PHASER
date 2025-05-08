import traceback
import logging, pathlib, os
import pandas as pd
import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed
from tqdm.auto import tqdm

## Local imports from ..utils
from ..utils import ImageLoader


pathlib.Path("./logs").mkdir(exist_ok=True)

logging.basicConfig(
    filename="./logs/process.log",
    encoding="utf-8",
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class ComputeHashes:
    """Compute Perceptual Hashes using a defined dictionary of algorithms, \\
        and a corresponding list for transformations to be applies
    """

    def __init__(
        self,
        algorithms: dict,
        transformations: list,
        n_jobs=1,
        backend="loky",
        progress_bar=False,
    ) -> None:
        """_summary_

        Args:
            algorithms (dict): Dictionary containing {'phash': phaser.hashing.PHASH(<settings>)}
            transformations (list): A list of transformations to be applies [phaser.transformers.Flip(<setting>)]
            n_jobs (int, optional): How many CPU cores to use. -1 uses all resources. Defaults to 1.
            backend (str, optional): Pass backend parameter to joblib. Defaults to 'loky'.
        """
        self.algos = algorithms
        self.trans = transformations
        self.n_jobs = n_jobs
        self.backend = backend
        self.progress_bar = progress_bar

    def fit(self, paths: list) -> pd.DataFrame:
        """Run the computation

        Args:
            paths (list): A list of absolute paths to original images

        Returns:
            pd.DataFrame: Dataset containing all computations
        """
        dirpath = os.path.dirname(paths[0])
        logging.info(f"===Beginning to process directory {dirpath}===")

        hashes = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
            delayed(sim_hashing)(
                img_path=p, algorithms=self.algos, transformations=self.trans
            )
            for p in tqdm(paths, desc="Files", disable=not self.progress_bar)
        )

        # joblib returns a list of numpy arrays from sim_hashing
        # the length depends on how many transformations are applied
        # concatenate the list and pass to a dataframe below
        hashes = np.concatenate(hashes, dtype=object)
        
        # derive the column names based on the list of algorithms
        cols = ["filename", "transformation", *list(self.algos.keys())]
        df = pd.DataFrame(hashes, columns=cols)

        # remove rows with any nan values
        pre_clean_count = len(df)
        df.dropna(how="any", inplace=True, axis=0)
        post_clean_count = len(df)

        # if we have removed some rows, log ths in the logger
        num_removed = pre_clean_count - post_clean_count
        # count how many versions of each file there are (i.e. how many rows per file)
        num_versions = 1 + len(self.trans)  # original + transofmrations
        if num_removed:
            logging.info(
                f"Dropped null records for {int(num_removed/num_versions)} files. "
            )

        return df


def sim_hashing(img_path:str, transformations:list=[], algorithms:dict={}) -> np.ndarray:
    """_summary_

    Args:
        img_path (str): Path to the original image
        transformations (list, optional): List of transformations to apply. Defaults to [].
        algorithms (dict, optional): List of perceptual hashing algorithms to calculate hashes for. Defaults to {}.

    Returns:
        np.ndarray: An array of hashes, where each row is [filemame:str, type:str (original or transform name), hashes:list]
        Note: If the original image fails to load, or there is an error when creating any of the transforms (or looking them up from disk)
        then all hashes in the response are set to None to avoid incomplete observation sets. (Making them easier to remove later)
    """

    error = False

    try:
        image_obj = ImageLoader(img_path)
        img = deepcopy(image_obj)
    except Exception as err:
        logging.error(f"Error processing path {img_path}: {err}")
        error = True

    outputs = []
    if not error:
        # loop over a set of algorithms
        hashes = [a.fit(img.image) for a in algorithms.values()]
        outputs.append([img.filename, "orig", *hashes])

        if len(transformations) > 0:
                for transform in transformations:
                    try:
                        _img = transform.fit(img)
                    except Exception as err:
                        logging.error(f"Error applying transform {transform.name} to {img.filename}: {err}.")
                        error=True
                        break
                    try:
                        hashes = [a.fit(_img) for a in algorithms.values()]
                        outputs.append([img.filename, transform.name, *hashes])
                    except Exception as err:
                        tb = traceback.extract_tb(err.__traceback__)
                        logging.error(f"Error hashing {transform.name} for {img.filename}: {err}.")
                        error=True
                        break

    # Return None for all observations if there was an error at the previous stage
    if error:
        # An error occured, so we want to abandon this set of observations.
        # Generate stacked np.nan arrays as placeholders to remove later.
        hashes = [None] * len(algorithms)  # each hash is replaced by NAN
        outputs = []
        outputs.append([img_path, "orig", *hashes])
        logging.info(f"Errors found, dropping all hashes for {img_path}.")
        for transform in transformations:
            outputs.append(
                [img_path, transform.name, *hashes]
            )  # one set of NAN hashes for each transformation
            
    # wrap outputs as np.array and dtype=object to handle [1,[4,5]] shapes
    return np.row_stack(np.array(outputs, dtype=object))
