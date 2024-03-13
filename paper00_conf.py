# Things that are useful across all steps in the framework.
import os, pathlib
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder

# Define a random number seed
RNG = 42 
np.random.seed(RNG)

# Get the working directory according to where the .py file is executed
#SCRIPT_DIR = f"{os.sep}".join(os.path.abspath(__file__).split(os.sep)[:-1])
SCRIPT_DIR = f"C:/Users/aabywan/Downloads/Flickr_8k"
#SCRIPT_DIR = f"C:/Users/aabywan/Downloads/MIRflickr1m"

# Make sure SCRIPT_DIR is the current working dir.
os.chdir(SCRIPT_DIR)

# Make folder for outputs if not already there.
pathlib.Path("./demo_outputs/figs").mkdir(parents=True, exist_ok=True)

IMGPATH = os.path.join(SCRIPT_DIR, "Images")

# Set a uniform imagesize for all figures
FIGSIZE    = (5, 3)

# ---------------------------------------------------------------------
# Experimental configuration 
# ---------------------------------------------------------------------
# Define which hashing algorithms to test
# Make sure overwrite any default settings
from phaser.hashing import PHash, WaveHash, PdqHash
ALGOS_dict = {
    "phash": PHash(hash_size=8), 
    "wave": WaveHash(), 
    "pdq": PdqHash()
    }

# Define which transformations to apply
# Make sure overwrite any default settings
from phaser.transformers import Border, Crop, Flip, Rescale, Rotate, Watermark
TRANS_list = [
    Border(border_colour=(255,255,255), border_width=30),
    Crop(cropbox_factors=[.05,.05,.05,.05]),
    Flip(direction='Horizontal'),
    Rescale(fixed_dimensions=(96,96), thumbnail_aspect=True),
    Rotate(degrees_counter_clockwise=5),
    Watermark()
    ]

# Define the distance metrics used for similarity scoring
# "key": value <- value can be a SciPy string or func(u,v,w)
METR_dict = {
    "Hamming": "hamming", 
    "Cosine": "cosine"
    #"Dummy test": test
    }