import os
from PIL import Image
import numpy as np
import pandas as pd
from math import ceil
from joblib import dump, load

class ImageLoader():
    def __init__(self, path:str):
        # do not use path.name since it depends on the path being WindowsPath etc.
        # instead, convert the path to a string value
        self.path = str(path)
        self.filename = os.path.basename(self.path)
        
        # load the image from the provided path
        self.image = Image.open(path)
        
        # get image dimensions
        self.width = self.image.size[0]
        self.height = self.image.size[1]

def bool2binstring(hash):
    # not sure if the check is necessary
    # all inputs should be bool arrays?
    if hash.dtype == 'bool':
        hash = np.array(hash, dtype=int)
    return "".join(str(b) for b in hash)

def bin2bool(hash):
    # Create an iterable of booleans by converting each character in the string to a boolean value
    bool_iterable = (bit == '1' for bit in hash)

    # Convert the iterable to a NumPy boolean array
    bool_array = np.fromiter(bool_iterable, dtype=np.bool_)

    return bool_array

def dump_labelencoders(encoders:dict, path:str, filename:str="LabelEncoders.bz2") -> None:
    #for name, enc in encoders.items():
    dump(encoders, os.path.join(path, filename), compress=9)

def load_labelencoders(path:str, filename:str="LabelEncoders.bz2"):
    return load(os.path.join(path, filename))


def create_file_batches(filelist:list[str], batch_size:int=100_000) -> list[list]:
    """Split a list of files in to batches which can be processed separately. Useful for large datasets which would cause memory issues.

    Args:
        filelist (list[str]): The list of files to split up.
        batch_size (int, optional): The size of the batches to create. Defaults to 100_000.

    Returns:
        list[list]: A list, where each element is a list[str] of files to process.
    """
    if batch_size >= len(filelist):
        return [filelist]
    n = ceil(len(filelist) / batch_size)
    return np.array_split(filelist, n)


def format_temp_name(batch_num:int) -> str:
    return f"temp_hashes_{batch_num}.csv.bz2"


def combine_temp_hash_dataframes(output_directory:str, num_batches:int) -> pd.DataFrame: 
    return_df = pd.DataFrame()

    # Loop through all temp CSV files and append them to the return DF.
    for bat in range(0, num_batches):
        csv_path = os.path.join(output_directory, format_temp_name(bat))
        df_h = pd.read_csv(csv_path)
        return_df = pd.concat([return_df, df_h])
        # delete temp CSV
        os.remove(csv_path)

    return return_df