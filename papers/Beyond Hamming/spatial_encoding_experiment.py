# %% Config Hashes, Transforms =================================
""" Small-scale experiment to determine if various Perceptual Hashes encode spatial information about an image.
We can surmise this from a description of the hash, but this models the extent of any encoding.
The basic idea is to produce image manipulations which have obvious positional properties (e.g. embedding the same
)
"""
import os, sys

# Set path for importing PHASER
module_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(module_dir, "../..")))


# Specify path of the original (non-transformed) dataset, edit as appropriate
# e.g. "F:\Datasets\images"
original_path = os.path.abspath(r"E:/Work etc/Datasets/Flickr_1_Million/flattened")

# Specify output directory, edit as appropriate
output_directory = os.path.abspath(r"D:/experiments/spatial_test2/")

from phaser.hashing import *
# Specify Perceptual Hashing Algorithms
ALGORITHMS = {
        "ahash": AverageHash(),
        "dhash": DifferenceHash(),
        "dhash_vertical": DifferenceHash(vertical=True),
        "neuralhash": NeuralHash(pad=2), # Pad Neuralhash to make it square for later analysis.
        "phash": PHash(), 
        "wave": WaveHash(), 
        "pdq": PdqHash()
        }


# Get path to resources
import phaser
phaser_path = os.path.dirname(os.path.abspath(phaser.__file__))
high_freq_jpeg = os.path.join(phaser_path,"resources", "high_freq_grass.jpg")
low_freq_png = os.path.join(phaser_path,"resources", "low_freq_gradiant.png")

# Note: the Composite transform is quite slow - which is the main reason for using a smaller dataset here to understand spatial representation.
from phaser.transformers import *
TRANSFORMERS = [
    Border(border_colour=(255,255,255), border_width_fraction=1/16, name="border-frac-white"),
    Crop(cropbox_factors=[.10,.10,.10,.10], name="Crop_factors_10_10_10_10"), # ~65% of the area remaining.
    Crop(cropbox_factors=[.25,.25,.0,.0], name="Crop_factors_25_25_0_0"), # 25% from left and top edges.
    
    Composite(position="top-left", scale=True, static_image=low_freq_png, name="composite_top-left_lf"),
    Composite(position="top-right", scale=True, static_image=low_freq_png, name="composite_top-right_lf"),
    Composite(position="bottom-right", scale=True, static_image=low_freq_png, name="composite_bottom-right_lf"),
    Composite(position="bottom-left", scale=True, static_image=low_freq_png, name="composite_bottom-left_lf"),
    Composite(position="left", scale=True, static_image=low_freq_png, name="composite_left_lf"),
    Composite(position="right", scale=True, static_image=low_freq_png, name="composite_right_lf"),
    Composite(position="top", scale=True, static_image=low_freq_png, name="composite_top_lf"),
    Composite(position="bottom", scale=True, static_image=low_freq_png, name="composite_bottom_lf"),
    
    Flip(direction='Horizontal', name="Flip-h"),
    Flip(direction='Vertical', name="Flip-v"),
    Rotate(degrees_counter_clockwise=15),
    Watermark(),
    ]


# %% Do hashing (Needs above cell to run first) =================================

import pathlib, os
from sklearn.preprocessing import LabelEncoder
from joblib import dump
import numpy as np


num_threads = 16 # edit this as appropriate

# Get list of images
list_of_images = [str(i) for i in pathlib.Path(original_path).glob("**/*")]
print(f"Found {len(list_of_images)} images in {os.path.abspath(original_path)}.")

# Sample 20k
RNG = 42
np.random.seed(RNG)
list_of_images = np.random.choice(list_of_images, 20_000, replace=False)
print(f"Sampled: {len(list_of_images)} images")

# Create output directory
print(f"Creating output directory at {os.path.abspath(output_directory)}...")
pathlib.Path(output_directory).mkdir(exist_ok=True)

print("Doing hashing...")
ch = ComputeHashes(ALGORITHMS, TRANSFORMERS, n_jobs=num_threads, progress_bar=True, backend="threading")
df_h = ch.fit(list_of_images)


compression_opts = dict(method="bz2", compresslevel=9)

# Create and fit LabelEncoders according to experiment
le = {
    "f": LabelEncoder().fit(df_h["filename"]),
    "t": LabelEncoder().fit(df_h["transformation"]),
    "a": LabelEncoder().fit(list(ALGORITHMS.keys())),
    "c": LabelEncoder(),
}

# Hard-code class labels for use when plotting
le["c"].classes_ = np.array(["Inter (0)", "Intra (1)"])

# Apply LabelEncoder on df_h
df_h["filename"] = le["f"].transform(df_h["filename"])
df_h["transformation"] = le["t"].transform(df_h["transformation"])

# Dump LabelEncoders and df_h to disk
print("Saving Label Encodings...")
dump(
    value=le,
    filename=os.path.join(output_directory, "LabelEncoders.bz2"),
    compress=9,
)

print("Saving compressed hash DataFrame to Hashes.df.bz2...")
# Also save compressed dataframe, this makes it faster to load in larger datasets.
outfile = os.path.join(output_directory, "Hashes.df.bz2")
dump(value=df_h, filename=outfile, compress=9)



# %% Get Distances =================================
from joblib import load, dump
from sklearn.preprocessing import LabelEncoder

# Distance Metrics. Only Hamming distance is necessary here as the spatial encoding is determined by determining which bits in the hash correspond to better matches for each transform.
from phaser.similarities import *
import pandas as pd
import numpy as np
DISTANCE_METRICS = {
    "Hamming": "hamming",
}

# Test that metrics have been entered correctly
from phaser.similarities import validate_metrics
if validate_metrics(DISTANCE_METRICS):
    print("Metrics look valid!")
    
if "df_h" not in locals():
    df_path = os.path.join(output_directory, "Hashes.df.bz2")
    df_h = load(df_path)
    # Load the Label Encoders used when generating hashes
    le = load(os.path.join(output_directory, "LabelEncoders.bz2"))
    print(f"DataFrame loaded from {os.path.abspath(df_path)}")
    

# Add distance metrics to label encodings
le["m"] = LabelEncoder().fit(list(DISTANCE_METRICS.keys()))

# Dump LabelEncoders and df_h to disk
print(
    "Updating label encoder with distance metrics."
)
# dump_labelencoders(le, path=output_directory)
dump(
    value=le,
    filename=os.path.join(output_directory, "LabelEncoders.bz2"),
    compress=9,
)


# Compute the intra distances
intra = IntraDistance(DISTANCE_METRICS, le, progress_bar=True)
intra_df = intra.fit(df_h)
print(f"Number of total intra-image comparisons = {len(intra_df)}")

# Compute the inter distances using subsampling
n_samples = find_inter_samplesize(len(df_h["filename"].unique() * 1))
inter = InterDistance(
    DISTANCE_METRICS, le, n_samples=n_samples, progress_bar=True
)
inter_df = inter.fit(df_h)
print(f"Number of pairwise comparisons = {inter.n_pairs_}")
print(f"Number of total inter distances = {len(inter_df)}")

# Combine distances
df_d = pd.concat([intra_df, inter_df])
compression_opts = dict(method="bz2", compresslevel=9)


distance_path = os.path.join(output_directory, "Distances.csv.bz2")
print(f"Saving distance scores to {os.path.abspath(distance_path)}.")
# Save as compressed CSV
df_d.to_csv(
    distance_path, index=False, encoding="utf-8", compression=compression_opts
)
# Save as compressed DF
distance_path = os.path.join(output_directory, "Distances.df.bz2")
dump(value=df_d, filename=(distance_path), compress=9)



    
# %% Calculate Bit-Weights for each transform  ========================
# Requires config cell to be run first.

from phaser.evaluation import ComputeMetrics
from phaser.evaluation import make_bit_weights
from phaser.similarities import IntraDistance, InterDistance, find_inter_samplesize
from joblib import load
import pandas as pd
import numpy as np

# If distance dataframe and label encodings are not already in memory from the previous step, load them from disk
if "df_d" not in locals():
    print(f"Loading dataframes from {os.path.abspath(output_directory)}...")
    df_h = load(os.path.join(output_directory, "Hashes.df.bz2"))
    df_d = load(os.path.join(output_directory , "Distances.df.bz2"))
    le = load(os.path.join(output_directory, "LabelEncoders.bz2"))
    print(f"Done loading dataframes.")
    DISTANCE_METRICS = {"Hamming": "hamming"}

    
# Inter (0), Intra (1)
dist_intra  = df_d[df_d["class"] == 1]
dist_inter  = df_d[df_d["class"] == 0]

    
TRANSFORMS = le['t'].classes_
METRICS    = le['m']
ALGORITHMS = le['a'].classes_

# Define the triplet combinations
triplets = np.array(np.meshgrid(
    ALGORITHMS, 
    [t for t in TRANSFORMS if t != 'orig'], # ignore 'orig'
    METRICS.classes_)).T.reshape(-1,3)

# Calculate metrics for each triplet
print("Calculating metrics")
cm = ComputeMetrics(le, df_d, df_h, analyse_bits=True, n_jobs=5)
m, b = cm.fit(triplets, weighted=False)


# Compute bit-weights
print("Calculating bit-weights")
bit_weights = make_bit_weights(b, le)
n_samples = find_inter_samplesize(len(df_h["filename"].unique() * 1))



# %% Produce graphs for exploratory analysis (could also load in to bit-weight ipynb)

# Requires previous cell to be run first.

import math
import seaborn as sns
import matplotlib.pyplot as plt
import os
from phaser.plotting import bit_weights_ax


def plot_bit_weight(algorithm, transform, distance_metric, save_path):
    """ 1-d array representaiton of weights
    """
    lookupstring = f"{algorithm}_{transform}_{distance_metric}"
    fig, ax = plt.subplots(1,1, figsize=(10,3), constrained_layout=True)
    _ = bit_weights_ax(b[lookupstring], title=lookupstring)
    ax.figure.savefig(save_path)
    plt.close()
    
def plot_tp_weights(algorithm, transform, distance_metric, save_path, quad="TP",):
    """2-d array representaiton of weights - requires sqrt of the length of the hash to be a whole number to work.
    This mnakes it much easier to see how the hash bits map to the original image.
    quad, iun this case, is the quadrant of the confusion matrix to render.
    """
    lookupstring = f"{algorithm}_{transform}_{distance_metric}"
    data = b[lookupstring][quad]
    sqrlength = int(math.sqrt(len(data)))
    reshaped = np.asarray(data).reshape(sqrlength,sqrlength)
    #plt.figure(figsize = (7,7))
    fig = sns.heatmap(reshaped, cmap='Greys', vmin=0, vmax=1,square=True)
    plt.savefig(save_path)
    plt.close()

for t in TRANSFORMERS:
    tname = t.name
    print(tname)
    save_folder = os.path.join(output_directory, "graphs", tname)
    # if "crop" in t.name.lower():
    #     tname = t.name.replace("Crop_", "Crop_fixed")
    try:
        os.makedirs(save_folder)
    except Exception:
        pass
    for a in ALGORITHMS:
        filename_quad = f"{a}-quad-Hamming.png"
        filename_square = f"{a}-squareTP-Hamming.png"
        save_path = os.path.join(save_folder, filename_quad)
        plot_bit_weight(algorithm=a, transform=tname, 
                        distance_metric="Hamming", save_path=os.path.join(save_folder, filename_quad))
        plot_tp_weights(algorithm=a, transform=tname,
                        distance_metric="Hamming", save_path=os.path.join(save_folder, filename_square))
    
print("Finished.")


# %% Produce Graphs specifically for the paper, rather than analysis =====================
import seaborn as sns
import matplotlib.pyplot as plt
import math
    
def plot_tp_weights(algorithm, transform, distance_metric, save_path, quad="TP",):
    """2-d array representaiton of weights - requires sqrt of the length of the hash to be a whole number to work.
    This mnakes it much easier to see how the hash bits map to the original image.
    quad, iun this case, is the quadrant of the confusion matrix to render.
    """
    lookupstring = f"{algorithm}_{transform}_{distance_metric}"
    data = b[lookupstring][quad]
    sqrlength = int(math.sqrt(len(data)))
    reshaped = np.asarray(data).reshape(sqrlength,sqrlength)
    #plt.figure(figsize = (7,7))
    fig = sns.heatmap(reshaped, cmap='Greys', vmin=0, vmax=1,)
    plt.savefig(save_path)
    plt.close()
    

# Selected items which appear in the paper
selected_items=["ahash;composite_top-left_lf;ahash composite top-left", "phash;composite_top-left_lf;phash composite top-left",
                "ahash;border-frac-white;ahash border", "phash;border-frac-white;phash border",
                "ahash;Crop_factors_25_25_0_0;ahash crop 25% top/left", "phash;Crop_factors_25_25_0_0;phash crop 25% top/left",
                "ahash;Flip-h;ahash mirror-x", "phash;Flip-h;phash mirror-x",
                "wave;Flip-v;whash mirror-y", "pdq;Flip-v;pdq mirror-y",
                "ahash;Rotate_15;ahash rotate 15", "phash;Rotate_15;phash rotate 15",
                "dhash_vertical;Watermark;dhash_vertical watermark", "phash;Watermark;phash watermark"]

# consider making this multiple sub plots of 2,1 for paper reference.
dimensions = (2,7)
ind = (0,0)
fig, ax = plt.subplots(nrows=dimensions[0], ncols=dimensions[1], figsize=(15,4), 
                       constrained_layout=True, sharex=True, sharey=True)

not_last = False
for i in selected_items:
    alg, trans, title = i.split(";")
    lookupstring = f"{alg}_{trans}_{"Hamming"}"
    
    data = b[lookupstring]["TP"]
    sqrlength = int(math.sqrt(len(data)))
    reshaped = np.asarray(data).reshape(sqrlength,sqrlength)
    x = sns.heatmap(reshaped, cmap='Greys', ax=ax[ind], cbar=False, vmin=0, vmax=1, 
                    square=True, cbar_kws={'label': 'Freq.'}, xticklabels=False, yticklabels=False)
    x.set_title(title)
    if ind[0] < dimensions[0]-1:
        ind = (ind[0] + 1, ind[1])
    elif ind[0] == dimensions[0]-1:
        ind = (0, ind[1] + 1)
    if ind == dimensions:
        not_last = True

    
plt.show()


selected_items=["phash;composite_top_lf;top", "phash;composite_left_lf;ph left",
                "phash;composite_bottom_lf;phash ", "phash;composite_right_lf;pht",]
selected_items=["neuralhash;composite_top_lf;top", "neuralhash;composite_left_lf;left",
                "neuralhash;composite_bottom_lf;bottom ", "neuralhash;composite_right_lf;right",]

# consider making this multiple sub plots of 2,1 for paper reference.
dimensions = (1,4)
ind = (0,0)
fig, ax = plt.subplots(nrows=dimensions[0], ncols=dimensions[1], figsize=(6,2), 
                       constrained_layout=True, sharex=True, sharey=True)
ind =0
not_last = False
for i in selected_items:
    alg, trans, title = i.split(";")
    lookupstring = f"{alg}_{trans}_{"Hamming"}"
    
    data = b[lookupstring]["TP"]
    sqrlength = int(math.sqrt(len(data)))
    reshaped = np.asarray(data).reshape(sqrlength,sqrlength)
    x = sns.heatmap(reshaped, cmap='Greys', ax=ax[ind], cbar=False, vmin=0, vmax=1, 
                    square=True, cbar_kws={'label': 'Freq.'}, xticklabels=False, yticklabels=False)
    x.set_title(title)
    fig.suptitle("Neuralhash compsite embedding")
    # if ind[0] < dimensions[0]-1:
    #     ind = (ind[0] + 1, ind[1])
    # elif ind[0] == dimensions[0]-1:
    #     ind = (0, ind[1] + 1)
    # if ind == dimensions:
    #     not_last = True
    ind +=1
    
plt.show()