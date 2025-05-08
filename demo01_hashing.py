from demo00_conf import *
from phaser.hashing import ComputeHashes

# Find all the images and compute hashes
list_of_image_paths = [str(i) for i in pathlib.Path(IMGPATH).glob("**/*")]


# Ensure files are sorted consistently when subset
list_of_image_paths = sorted(list_of_image_paths)
print(f"Found {len(list_of_image_paths)} files/directories in '{SCRIPT_DIR}/Images'")

# Prepare for parallel processing
ch = ComputeHashes(ALGOS_dict, TRANS_list, n_jobs=-1, progress_bar=True)
df_h = ch.fit(list_of_image_paths)

# Create and fit LabelEncoders according to experiment
le = {
    "f": LabelEncoder().fit(df_h["filename"]),
    "t": LabelEncoder().fit(df_h["transformation"]),
    "a": LabelEncoder().fit(list(ALGOS_dict.keys())),
    "m": LabelEncoder().fit(list(METR_dict.keys())),
    "c": LabelEncoder(),
}

# Hard-code class labels for use when plotting
le["c"].classes_ = np.array(["Inter (0)", "Intra (1)"])

# Apply LabelEncoder on df_h
df_h["filename"] = le["f"].transform(df_h["filename"])
df_h["transformation"] = le["t"].transform(df_h["transformation"])

# Pickle and compress objects to disk
dump(value=le, filename="./demo_outputs/LabelEncoders.bz2", compress=9)
dump(value=df_h, filename="./demo_outputs/Hashes.df.bz2", compress=9)
print(f"Script completed")
