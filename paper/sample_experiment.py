"""Code for the experiment to determine appropriate dataset sizes when using this framework
"""
#%% Initialise
import os
import pathlib
import sys

import numpy as np
import pandas as pd
from joblib import load
from tqdm.auto import tqdm

# local imports
module_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.abspath(os.path.join(module_dir, "..")))
from notebooksupport import calculate_distances
from phaser.utils import bin2bool, load_labelencoders
from phaser.evaluation import ComputeMetrics, MetricMaker
from phaser.similarities import find_inter_samplesize, InterDistance, IntraDistance, validate_metrics
from sklearn.preprocessing import LabelEncoder


DISTANCE_METRICS = {
    "Hamming": "hamming"
}

#hash_directory = os.path.abspath(r"../flickr_1_mil_out")
hash_directory = os.path.abspath(r"../demo_outputs")
output_directory_base = os.path.abspath(os.path.join(module_dir, "sample_data"))

sample_sizes = [1_000, 10_000, 100_000, 250_000] # sample test size
iterations = 150 # number of times to generate distances for each sample size

#%% Load HashData

hash_path = os.path.join(module_dir, hash_directory, "Hashes.df.bz2")
df_h = load(hash_path)
le = load(os.path.join(hash_directory, "LabelEncoders.bz2"))
print(f"Dataframe (len:{len(df_h)}) loaded from {os.path.abspath(hash_path)}")


# Only look at necessary metrics
le["m"] =  LabelEncoder().fit(list(DISTANCE_METRICS.keys()))

# Get values to construct triplets
TRANSFORMS = le["t"].classes_
METRICS = le["m"].classes_
ALGORITHMS = le["a"].classes_

print(", ".join(TRANSFORMS))
print(", ".join(METRICS))
print(", ".join(ALGORITHMS))

# Clean up zero sum hashes
for a in df_h.columns[2:]:
    mask = df_h[a].apply(lambda x: sum(x)) == 0
    bad_filenames = df_h[mask]["filename"].unique()

    print(f"{len(bad_filenames)} bad hashes found for {a}")

    if len(bad_filenames) > 0:
        df_h = df_h[~df_h["filename"].isin(bad_filenames)]


#%% Process Samples

# Generate triplet combinations without 'orig'
triplets = np.array(
    np.meshgrid(
        le["a"].classes_, [t for t in le["t"].classes_ if t != "orig"], le["m"].classes_
    )
).T.reshape(-1, 3)

print(triplets)

        
aggregate = pd.DataFrame()

for s in sample_sizes:
    print(f"Working on sample size of {s}")
    for i in tqdm(range(0, iterations)):
        df_d = None        
        # Load hashes and labels from the output generated by the previous step and calculate inter- and intra-distances.
        # sample size is passed through to calculate_distances
        df_d = calculate_distances(hash_dataframe=df_h, label_encodings=le, sample_files=s, distance_metrics=DISTANCE_METRICS, progress_report=False, remove_bad_hash=False, out_dir="",save_to_disk=False)
    
        cm = ComputeMetrics(le, df_d, df_h, analyse_bits=False)
        metrics, bitfreq = cm.fit(triplets=triplets)
        metrics["sample_size"] = s
        metrics["iteration"] = i
        aggregate = pd.concat([aggregate, metrics])
    # build an aggregate csv file in case of error for each sample size
    aggregate.to_csv(os.path.join(hash_directory, f"aggregate_{s}.csv.bz2"), index=False)
# final aggregate file combining all samples
aggregate.to_csv(os.path.join(hash_directory, "aggregate.csv.bz2"), index=False)


# %% Visualise
from seaborn import *
import matplotlib.pyplot as plt
import itertools
from scipy import stats


# Future warnings in these library versions, supress them.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load in intermediatte iterations and combine
if not'aggregate' in locals():
    
# Code to piece together the bits if a problem occurred
#     aggregate = pd.DataFrame()
#     for it in iterations:
#         folderpath = os.path.join(hash_directory, f"iter_{it}")
#         for s in sample_sizes:
#             filepath = os.path.join(folderpath, f"aggregate_{s}.csv.bz2")
#             print(filepath)
#             temp_df = pd.read_csv(filepath)
#             aggregate = pd.concat([aggregate, temp_df])  
# aggregate.drop_duplicates(inplace=True)

    filepath = os.path.join(hash_directory, f"aggregate.csv.bz2")
    aggregate = pd.read_csv(filepath)

for s in sample_sizes:
    print(f"Stats for sample sise of {s}:")
    _X = aggregate[aggregate["sample_size"] == s].copy()
    print(_X.describe())


#%%

# Lots of commented out code for experimenting with plots - but uncommented current code produces EER plot used in the paper.

if not'aggregate' in locals():
  aggregate = pd.read_csv(os.path.join(hash_directory, "aggregate.csv.bz2"))

# for a in ALGORITHMS:
#     print(a)
#     #for t in TRANSFORMS:
#     #print(a,t)
#     #data = aggregate.loc[aggregate["Transform"] == t]
#     data = aggregate[aggregate["Algorithm"] == a]
#     origvals = data[data["Transform"] == "orig"].index
#     data.drop(origvals, inplace=True)
#     #print(data.head())
#     fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8,5), constrained_layout=True)
#     ax_ = scatterplot(data=data, x="sample_size", y="Threshold" , hue="Transform", ax=ax)

#     plt.show()
        

# Plot grid with transforms as legend.

METRICS = {"Threshold": "EER Decision Threshold"}

n_cols = len(METRICS)
n_rows = len(ALGORITHMS)
# Subset data
#fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(15,8), constrained_layout=True, sharex=True, sharey=False)


#TRANSFORMS = ["Rescale_fixed(96, 96)", "Watermark"]

mks = itertools.cycle(['o', 'x', '+', '^', '*', '8', 's', 'p', 'D', 'V'])
markers = [next(mks) for i in TRANSFORMS]

for col_i, metric in enumerate(METRICS.keys()):
    for row_i, algo in enumerate(ALGORITHMS):
            # Transform strings to labels
            data = aggregate[aggregate["Algorithm"] == algo].copy()
            dropvals = data[data["Transform"] == "orig"].index
            data.drop(dropvals, inplace=True)
            # dropvals = data[data["Transform"] == "Flip_Horizontal"].index
            # data.drop(dropvals, inplace=True)
            # Change similarity to distance
            # data["Threshold"] = 1 - data["Threshold"]
        
            
            # print(algo, metric)
            # d = data[data["Transform"] == "Rescale_fixed(96, 96)"] 
            # d100k = d[d["sample_size"] == 1_000] 
            # d100kvals = d100k[metric]
            # d250k = d[d["sample_size"] == 250_000] 
            # d250kvals = d250k[metric]
            # print(f"t-test for Rescale-{metric} between 100k and 250k")
            # print(stats.mannwhitneyu(d100kvals, d250kvals))

            # d = data[data["Transform"] == "Watermark"] 
            # d100k = d[d["sample_size"] == 1_000] 
            # d100kvals = d100k[metric]
            # d250k = d[d["sample_size"] == 250_000] 
            # d250kvals = d250k[metric]
            # print(f"t-test for Watermark-{metric} between 100k and 250k")
            # print(stats.mannwhitneyu(d100kvals, d250kvals))

            
            # d = data[data["Transform"] == "Flip_Horizontal"] 
            # d100k = d[d["sample_size"] == 1_000] 
            # d100kvals = d100k[metric]
            # d250k = d[d["sample_size"] == 250_000] 
            # d250kvals = d250k[metric]
            # print(f"t-test for Flip_Horizontal-{metric} between 100k and 250k")
            # print(stats.mannwhitneyu(d100kvals, d250kvals))

            d = data[data["Transform"] == "Rescale_fixed(96, 96)"] 
            d[d["sample_size"] == 1_000] 
            print("rescale1k")
            print(d.describe())
            # d = data[data["Transform"] == "Watermark"] 
            # d[d["sample_size"] == 1_000] 
            # print("water1k")
            # print(d.describe())
            # d = data[data["Transform"] == "Flip_Horizontal"] 
            # d[d["sample_size"] == 1_000] 
            # print("flip1k")
            # print(d.describe())

            d = data[data["Transform"] == "Rescale_fixed(96, 96)"] 
            d[d["sample_size"] == 10_000] 
            print("rescale10k")
            print(d.describe())
            # d = data[data["Transform"] == "Watermark"] 
            # d[d["sample_size"] == 10_000] 
            # print("water10k")
            # print(d.describe())
            # d = data[data["Transform"] == "Flip_Horizontal"] 
            # print("flip10k")
            # d[d["sample_size"] == 10_000] 
            # print(d.describe())



            # Rename labels for plot.
            cols = []
            for i, c in enumerate(data.columns):
                if c in METRICS:
                    cols.append(METRICS[c])
                elif c == "sample_size":
                    cols.append("Sample Size")
                else:
                    cols.append(c)
            data.columns = cols

            # dropvals = data[data["Transform"] == "Flip_Horizontal"].index
            # data.drop(dropvals, inplace=True)
            #water = data[data["Transform"] == "Flip_Horizontal"].index
            # data["total"] = data["FP"] + data["FN"] + data["TP"] + data["TN"]
            # data["fprate"]  = data["FP"] /  data["total"]
            # data["fnrate"]  = data["FN"] /  data["total"]

            fig, ax_ = plt.subplots(ncols=1, nrows=1, figsize=(5,5), constrained_layout=True, 
                                sharex=True, sharey=False)


            #data.drop(water, inplace=True)
            hue_order = ["Rescale_fixed(96, 96)", "Watermark", "Flip_Horizontal"]
            ax_ = boxenplot(data=data, x="Sample Size", y=METRICS[metric], hue="Transform", hue_order=hue_order, palette="Set2")#, alpha=.30)

            # ax_.set(yticks=np.arange(0.0, 1.01, 0.2))
            #ax_ = barplot(data=data, x="sample_size", y="Threshold" , ax=axes[row_i, col_i], legend=False)
            #ax_.get_legend().remove()
            ax_.set_title(f"{METRICS[metric]} - {algo}")
            ax_.legend()
            plt.show()
            fig.savefig(fname=os.path.join(hash_directory, f"scale_{metric}-{algo}.pdf"))
#ax_.legend()
plt.show()


# Do T-testing
# for m in metric
# hundrek_vals = ""
# twofiftyk_vals = ""

# stats.ttest_ind(hundrek_vals, twofiftyk_vals)

#=====
# plot grid with algorithm as legend
# METRICS = ["FP", "FN", "Threshold"]

# n_cols = len(METRICS)
# n_rows = len(TRANSFORMS)
# # Subset data
# fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(15,8), constrained_layout=True, 
#                             sharex=True, sharey=False)

# mks = itertools.cycle(['o', 'x', '+', '^', '*', '8', 's', 'p', 'D', 'V'])
# markers = [next(mks) for i in TRANSFORMS]

# for col_i, metric in enumerate(METRICS):
#     for row_i, trans in enumerate(TRANSFORMS):
#             # Transform strings to labels
#             data = aggregate[aggregate["Transform"] == trans].copy()
#             origvals = data[data["Transform"] == "orig"].index
#             #water = data[data["Transform"] == "Flip_Horizontal"].index
#             # data["total"] = data["FP"] + data["FN"] + data["TP"] + data["TN"]
#             # data["fprate"]  = data["FP"] /  data["total"]
#             # data["fnrate"]  = data["FN"] /  data["total"]
#             data.drop(origvals, inplace=True)
#             #data.drop(water, inplace=True)
#             ax_ = scatterplot(data=data, x="sample_size", y=metric, hue="Algorithm", ax=axes[row_i, col_i])#, alpha=.30)

#             #ax_ = barplot(data=data, x="sample_size", y="Threshold" , ax=axes[row_i, col_i], legend=False)
#             #ax_.get_legend().remove()
#             ax_.set_title(f"{trans}-{metric}")
# ax_.legend()
# plt.show()
#=====


# data = aggregate.loc[aggregate["Transform"] == "Watermark"]
# data = data.loc[data["sample_size"] <20_000]
# data = data.loc[data["Algorithm"] == "Wavehash"]
# print(data.head())
# #data = aggregate.query("Algorithm == PDQ, Transform == Watermark").copy()

# ax = jointplot(data=data, x="sample_size", y="AUC")# , hue="time", style="time")
# #ax.set_ylim(0.0,1.0)
# plt.show()

# ax = jointplot(data=data, x="sample_size", y="Threshold")# , hue="time", style="time")
# #ax.set_ylim(0.0,1.0)
# plt.show()

# data = aggregate.loc[aggregate["Transform"] == "Flip_Horizontal"]
# data = data.loc[data["sample_size"] <20_000]
# data = data.loc[data["Algorithm"] == "Wavehash"]
# print(data.head())
# ax = jointplot(data=data, x="sample_size", y="AUC")# , hue="time", style="time")
# #ax.set_ylim(0.0,1.0)
# plt.show()

# ax = jointplot(data=data, x="sample_size", y="Threshold")# , hue="time", style="time")
# #ax.set_ylim(0.0,1.0)
# plt.show()

# %%