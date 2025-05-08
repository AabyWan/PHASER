# %% Config Hashes, Transforms =================================
""" Large-scale experiment to measure the performance of the context-aware distance metircs.
Uses an existing 250k random sample of the Flickr 1 Million dataset. 
Some items were loaded from disk, while others were generated fresh.
"""
import os, sys

# Set path for importing PHASER
module_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(module_dir, "../..")))



# Specify path of the original (non-transformed) dataset
# e.g. "F:\Datasets\images"
# In this case, point to a flattened version of the Flickr 1 Million Dataset, where all files are in a single directory
# (By default they are in several subfolders)
original_path = os.path.abspath(r"E:/Work etc/Datasets/Flickr_1_Million/flattened")

# Specify output directory
output_directory = os.path.abspath(r"D:/experiments/metric_eval2/")

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
low_freq_png = os.path.join(phaser_path,"resources", "low_freq_gradiant.png")

# Note: the Composite transform is quite slow - which is the main reason for using a smaller dataset or to limit the number of them.
from phaser.transformers import *
TRANSFORMERS = [
    Border(border_colour=(0,0,0), border_width_fraction=1/16, name="border-frac-black"),
    Crop(cropbox_factors=[.10,.10,.10,.10], name="Crop_factors_10_10_10_10"), # ~65% of the area remaining.
    Crop(cropbox_factors=[.25,.25,.0,.0], name="Crop_factors_25_25_0_0"), # 25% from left and top edges.
    Composite(position="top-left", scale=True, static_image=low_freq_png, name="composite_top-left_lf"),
    Composite(position="left", scale=True, static_image=low_freq_png, name="composite_left_lf"),
    Composite(position="top", scale=True, static_image=low_freq_png, name="composite_top_lf"),
    Flip(direction='Horizontal', name="MirroX"),
    Flip(direction='Vertical', name="MirrorY"),
    Rotate(degrees_counter_clockwise=15, name="Rotate_15"),
    Watermark()
    ]


# %% Do hashing (Needs above cell to run first) =================================

import pathlib, os
from sklearn.preprocessing import LabelEncoder
from joblib import dump
from phaser.utils import create_file_batches, format_temp_name, combine_temp_hash_dataframes, remove_temp_hash_dataframes
import numpy as np

# Hashing performance options and sampling

num_threads = 10 # edit this as appropriate
batch_size = 50_000 # edit as appropriate
sample_size = 0 # set to a number larger than zero to sample, defaults to 0 which is no sample.
# sample_size = 2_000 # testing only

# Get list of images
print("Checking directory for original images.")
list_of_images = os.listdir(original_path)
print(f"Found {len(list_of_images)} images in {os.path.abspath(original_path)}.")

sample = []
# Get sample list
with open("file_list_250k.txt", "r") as f:
    for line in f:
        sample.append(line.strip())

# Reduce to the list from the file provided.
list_of_images = [os.path.join(original_path, f) for f in (set(list_of_images).intersection(set(set(sample))))]
print(f"Subset sampled to {len(list_of_images)} overlapping files.")

if sample_size != 0:
    list_of_images = np.random.choice(list_of_images, sample_size, replace=False)
    print(f"Sampled {len(list_of_images)}.")

if len(list_of_images) > batch_size:
    print(
        f"Dataset is large, splitting in to smaller chunks (batch size={batch_size})."
    )

batches = create_file_batches(list_of_images, batch_size)
num_batches = len(batches)
print(f"""Created {num_batches} batches to process. Batches are saved as separate dataframes on disk, 
      before being combined if there is more than one..""")


# Create output directory
print(f"Creating output directory at {os.path.abspath(output_directory)}...")
pathlib.Path(output_directory).mkdir(exist_ok=True)
# compression_opts = dict(method="bz2", compresslevel=9)

print("Doing hashing...")
for bat in range(0, num_batches):
    df_h = None
    if num_batches > 1:
        print(f"Batch {bat}...")
    ch = phaser.hashing._helpers.ComputeHashes(
        ALGORITHMS, TRANSFORMERS, n_jobs=num_threads, progress_bar=True, backend="threading"
    )
    df_h = ch.fit(batches[bat])

    # Dump temporary hashes, if any
    if num_batches > 1:
        outfile = os.path.join(output_directory, format_temp_name(bat))
        dump(value=df_h, filename=outfile, compress=9)

outfile = os.path.join(output_directory, "Hashes.df.bz2")
if num_batches > 1:
    # Create a single dataframe from the intermediatte dataframes
    print(f"Combining temporary dataframe files and saving to {outfile}...")
    df_h = combine_temp_hash_dataframes(
        output_directory=output_directory, num_batches=num_batches, save_to_disk=outfile
    )
else:
    print(f"Saving compressed hash DataFrame to {outfile}...")
    dump(value=df_h, filename=outfile, compress=9)

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

# Cleanup temp files if any
if num_batches > 1:
    print("Removing temp dataframes..")
    remove_temp_hash_dataframes( output_directory=output_directory, num_batches=num_batches)

print("Done.")

# %% Configure distance metrics

import numpy as np
from scipy.spatial.distance import hamming # needed for passing through to hatched matrix

# Distance Metrics.
from phaser.similarities import *

# List of distance metrics to test
# Commented out the various versions of each algorithm, leaving just the ones used in the paper.

DISTANCE_METRICS = {
    "Hamming": "hamming",
    # "Convolution_sumdiff_3_3": (convolution_distance, {"mode":"sum_diffs"}), # 3x3
    "Convolution_sumdiff_4_4": (convolution_distance, {"mode":"sum_diffs", "filter": np.ones((4,4))}),
    # "Convolution_sumdiff_5_5": (convolution_distance, {"mode":"sum_diffs", "filter": np.ones((5,5))}),
    # "Convolution_sumdiff_6_6": (convolution_distance, {"mode":"sum_diffs", "filter": np.ones((6,6))}),
    # "Convolution_sumdiff_7_7": (convolution_distance, {"mode":"sum_diffs", "filter": np.ones((7,7))}),
    # "Convolution_sumdiff_5_5_discountfar": (convolution_distance, {"mode":"sum_diffs", "filter": np.array([
    #                                                                                         [-0.5,-0.5,-0.5,-0.5,-0.5],
    #                                                                                         [-0.5,1,1,1,-0.5],
    #                                                                                         [-0.5,1,1,1,-0.5],
    #                                                                                         [-0.5,1,1,1,-0.5],
    #                                                                                         [-0.5,-0.5,-0.5,-0.5,-0.5],
    #                                                                                         ])}),
    # "Convolution_sumdiff_half_diag": (convolution_distance, {"mode":"sum_diffs", "filter": np.array([[1, 2, 1],
    #                                                                                                  [2, 2, 2],
    #                                                                                                  [1, 2, 1]])}),
    # "Convolution_sumdsim_laplace": (convolution_distance, {"mode":"sum_similar", "filter": np.array([[ 0, -1,  0],
    #                                                                                                  [-1,  4, -1],
    #                                                                                                  [ 0, -1,  0]])}),
    # "Convolution_sumdiff_adj": (convolution_distance, {"mode":"sum_diffs", "filter": np.array([[0, 1, 0],
    #                                                                                            [1, 1, 1],
    #                                                                                            [0, 1, 0]])}),
    # "Convolution_sumsim": (convolution_distance, {"mode":"sum_similar"}),
    # "Convolution_vertical_skip": (convolution_distance, {"mode":"sum_diffs", "filter": np.array([[1, 0, 1],
    #                                                                                              [1, 0, 1],
    #                                                                                              [1, 0, 1]])}),
    # "Convolution_horizontal_skip": (convolution_distance, {"mode":"sum_diffs", "filter": np.array([[1, 1, 1],
    #                                                                                                [0, 0, 0],
    #                                                                                                [1, 1, 1]])}),
    # "Convolution_de-emphasise-near": (convolution_distance, {"mode":"sum_diffs", "filter": np.array([[1, 1, 1],
    #                                                                                             [1, 10, 1],
    #                                                                                             [1, 1, 1]])}),
#     "Convolution_negative-near": (convolution_distance, {"mode":"sum_diffs", "filter": np.array([[-0.5, -0.5, -0.5],
#                                                                                                 [-0.5, 2, -0.5],
#                                                                                                 [-0.5,-0.5 , -0.5]])}),
#     "Convolution_6_discount_Far": (convolution_distance, {"mode":"sum_diffs", "filter": np.array([[-0.5, -0.5, -0.5],
#                                                                                             [-0.5, 2, -0.5],
#                                                                                             [-0.5,-0.5 , -0.5]])}),
    # "Hatched_Matrix": (hatched_matrix, {"distance_fun": hamming}),
    "Hatched_Matrix2": (hatched_matrix2, {"distance_fun": hamming}),
    "ngram_cosine_2gram": (ngram_cosine_distance, {"ngram_size": 2}),
    # "ngram_cosine_3gram": (ngram_cosine_distance, {"ngram_size": 3}),
    # "ngram_cosine_4gram": (ngram_cosine_distance, {"ngram_size": 4}),
    # "ngram_cosine_5gram": (ngram_cosine_distance, {"ngram_size": 5}),
    # "ngram_cosine_6gram": (ngram_cosine_distance, {"ngram_size": 6}), 
}

# Test that metrics have been entered correctly
if validate_metrics(DISTANCE_METRICS):
    print("Metrics look valid!")

    
# %% Get Distances ================================= 
# Requires first config cell and the above cell to be run first.

import pandas as pd
import numpy as np

from joblib import load, dump
from sklearn.preprocessing import LabelEncoder
    
# Load the hashes if not already in a local variable
if "df_h" not in locals():
    df_path = os.path.join(output_directory, "Hashes.df.bz2")
    df_h = load(df_path)
    # Load the Label Encoders used when generating hashes
    le = load(os.path.join(output_directory, "LabelEncoders.bz2"))
    print(f"DataFrame loaded from {os.path.abspath(df_path)}")
    

# Find hashes that sum to 0 since they can cause issues with distance metrics
for a in df_h.columns[2:]:
    mask = df_h[a].apply(lambda x: sum(x)) == 0
    bad_filenames = df_h[mask]["filename"].unique()
    if len(bad_filenames) > 0:
        df_h = df_h[~df_h["filename"].isin(bad_filenames)]



# Add distance metrics to label encodings
le["m"] = LabelEncoder().fit(list(DISTANCE_METRICS.keys()))

# Dump LabelEncoders and df_h to disk
print(
    "Updating label encoder with distance metrics."
)
dump(
    value=le,
    filename=os.path.join(output_directory, "LabelEncoders.bz2"),
    compress=9,
)

print("Calculating Distances.\n")

# Compute the intra distances
print("Intra Distances:")
intra = IntraDistance(DISTANCE_METRICS, le, progress_bar=True)
intra_df = intra.fit(df_h)
print(f"Number of total intra-image comparisons = {len(intra_df)}")


# Compute the inter distances using subsampling
print("Inter Distances:")
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



    
# %% Analysis Setup ========================
# Requires config cell to be run first.

from phaser.evaluation import ComputeMetrics
from joblib import load

import pandas as pd

# If distance dataframe and label encodings are not already in memory from the previous step, load them from disk
if "df_d" not in locals():
    print(f"Loading Distance dataframe from {output_directory}")
    df_d = load(os.path.join(output_directory , "Distances.df.bz2"))
    print(f"Loading Hash dataframe from {output_directory}")
    df_h = load(os.path.join(output_directory, "Hashes.df.bz2"))
    print(f"Loading labels from {output_directory}")
    le = load(os.path.join(output_directory, "LabelEncoders.bz2"))
    print(f"Data imported successfully.")

# Inter (0), Intra (1)
# intra_df  = df_d[df_d["class"] == 1]
# inter_df  = df_d[df_d["class"] == 0]

    
TRANSFORMS = le['t'].classes_
METRICS    = le['m'].classes_
ALGORITHMS = le['a'].classes_


print(TRANSFORMERS)
print(METRICS)
print(ALGORITHMS)

# %% Summary Metrics
from tqdm.auto import tqdm

# Create an empty dataframe, it's not needed by metric maker unless processing bits.
empty_df = pd.DataFrame()

# Create stats dir to receive text files for stats.
stats_dir = os.path.join(output_directory, "stats")
try:
    os.makedirs(stats_dir)
except Exception:
    pass

print("Create evaluation statistics")
cm = ComputeMetrics(le, df_d, empty_df, analyse_bits=False, n_jobs=3)

# If too much memory is used, try this:
# # Process metrics for each transform in turn - it can get heavy doing all at once.
# for t in tqdm(TRANSFORMS):
#     if t != 'orig':  # ignore 'orig'
#         print(t)
#         data = [df_d[t]]
#         triplets = np.array(np.meshgrid(
#             ALGORITHMS, 
#             t,
#             METRICS)).T.reshape(-1,3)
        
#         # Process the transform triplet sets.
#         m, b = cm.fit(triplets, weighted=False)
        
#         # Write the metric maker stats to a file for later.
#         outpath = os.path.join(stats_dir, f"{t}.csv")
#         x = m.to_csv(outpath)
        
# Define the triplet combinations
triplets = np.array(np.meshgrid(
    ALGORITHMS, 
    [t for t in TRANSFORMS if t != 'orig'], # ignore 'orig'
    METRICS)).T.reshape(-1,3)
m, b = cm.fit(triplets, weighted=False)

# AUC info
print(m.groupby(['Algorithm', "Metric"])[['AUC']].agg(['mean','std']))
# Dump entire stats dataframe to file
m.to_csv(os.path.join(stats_dir, "all_stats.csv"))
# Separately dump a handy AUC aggregation for all transforms in a metric/algorithm pair
m.groupby(['Algorithm', "Metric"])[['AUC', "FP", "FN", "TP", "TN"]].agg(['mean','std']).to_csv(os.path.join(stats_dir, "aggregate_AUC.csv"))


# %% Tables and Graphs

from phaser.evaluation import MetricMaker
from phaser.plotting import  hist_fig, kde_ax, eer_ax, roc_ax
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore Seaborn warnings due to underlying package using future deprecated calls
from joblib import Parallel, delayed

# requires intra/inter dfs and le to be assigned (run the Analysis Setup cell)
print("Generating and saving graphs:")

def kde_plot(transform, algorithm, metric, width=8, height=6, save_path=""):
    if transform.lower() != "orig":
        # Transform strings to labels
        m_label = le["m"].transform(np.array(metric).ravel())
        a_label = le["a"].transform(np.array(algorithm).ravel())

        # Subset data and get the distances for the chosen transformation
        _X = df_d.query(f"algo=={a_label} and metric == {m_label}")
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(width, height), constrained_layout=True)
        ax = kde_ax(_X, transform, label_encoding=le, fill=True, title=f"{algorithm}-{metric}", ax=ax)
        fig.savefig(save_path)
        plt.close()
  
def roc_plot(transform, algorithm, metric, width=8, height=6, save_path=""):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(width, height), constrained_layout=True)

    # Transform strings to labels
    m_label = le["m"].transform(np.array(metric).ravel())
    a_label = le["a"].transform(np.array(algorithm).ravel())

    # Subset data and get the distances for the chosen transformation
    _X = df_d.query(f"algo=={a_label} and metric == {m_label}")

    # get similarities and true class labels
    y_true = _X["class"]
    y_similarity = _X[transform]

    # Prepare metrics for plotting EER and AUC
    mm = MetricMaker(y_true=y_true, y_similarity=y_similarity, weighted=False)
    
    # Make predictions and compute cm using EER
    roc_ax(mm.fpr, mm.tpr, mm.auc, title=f"{algorithm}-{metric}", ax=ax)
    fig.savefig(save_path)
    plt.close()
    
    
def figs(transform, algorithm, metrics, t_folder):
    alg_folder = os.path.join(t_folder, algorithm)
    try:
        os.makedirs(alg_folder)
    except Exception:
        pass
    print(alg_folder)
    for d in metrics:
        kde_plot(transform=transform, algorithm=algorithm, metric=d, save_path=os.path.join(alg_folder, f"KDE_{d}"))
        roc_plot(transform=transform, algorithm=algorithm, metric=d, save_path=os.path.join(alg_folder, f"ROC_{d}"))



for t in TRANSFORMERS:
    tname = t.name
    print(tname)
    trans_folder = os.path.join(output_directory, "graphs", tname)
    
    try:
        os.makedirs(trans_folder)
    except Exception:
        pass
        
    Parallel(n_jobs=7, backend="threading")(
        delayed(figs)(transform=tname, algorithm=a, metrics=METRICS, t_folder=trans_folder
        ) for a in ALGORITHMS
    )

print("Done!")