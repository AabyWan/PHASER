import os
from PIL import Image
import numpy as np
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

def dump_labelencoders(encoders:dict, path:str) -> None:
    #for name, enc in encoders.items():
    dump(encoders,f"{path}LabelEncoders.bz2", compress=9)

def load_labelencoders(filename:str, path:str):
    return load(f"./{path}{filename}.bz2")