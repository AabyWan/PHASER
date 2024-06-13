from demo00_conf import *
from phaser.similarities import IntraDistance, InterDistance, find_inter_samplesize

# Read the precomputed hashes
print(f"Loading LabelEncoders and hashing dataset...", end="")
le = load("./demo_outputs/LabelEncoders.bz2")
df_h = load("./demo_outputs/Hashes.df.bz2")
print(f"complete", end="\n")

# Find hashes that sum to 0 since they can cause issues with distance metrics
for a in df_h.columns[2:]:
    mask = df_h[a].apply(lambda x: sum(x)) == 0
    bad_filenames = df_h[mask]["filename"].unique()

    print(f"{len(bad_filenames)} bad filenames found for {a}")

    if len(bad_filenames) > 0:
        df_h = df_h[~df_h["filename"].isin(bad_filenames)]

# Downsample original hashes
DOWNSMPL = False
if DOWNSMPL:
    samplesize = 2000
    unique_files = sorted(df_h["filename"].unique())
    sample_files = np.random.choice(unique_files, samplesize, replace=False)
    df_h = df_h[df_h["filename"].isin(sample_files)]
    print(f"Saving subset...")
    dump(value=df_h, filename="./demo_outputs/Hashes_subset.df.bz2", compress=9)
    print(
        f"Number of unique files in downsampled data = {len(df_h['filename'].unique())}"
    )

# Compute the intra distances
intra_df = IntraDistance(METR_dict, le, 1, progress_bar=True).fit(df_h)
print(f"Number of total intra-image comparisons = {len(intra_df)}")

# Compute the inter distances using subsampling
n_samples = find_inter_samplesize(len(df_h["filename"].unique() * 1))
print(f"Number of pairwise comparisons = {n_samples}")

inter_df = InterDistance(
    METR_dict, le, set_class=0, n_samples=n_samples, progress_bar=True
).fit(df_h)
df_d = pd.concat([intra_df, inter_df])

print(f"Number of total inter distances = {len(inter_df)}")

# Combine distances and save to disk
dump(value=df_d, filename="./demo_outputs/Distances.df.bz2", compress=9)
print(f"Script completed")
