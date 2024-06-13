import pathlib, os, phaser.hashing, phaser.transformers
from phaser.utils import ImageLoader as IL
from phaser.utils import dump_labelencoders, load_labelencoders, bin2bool

# for do hashing and comparisons
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder
from phaser.utils import (
    create_file_batches,
    format_temp_name,
    combine_temp_hash_dataframes,
)

# for list modules
import scipy.spatial.distance

# for comparisons
import pandas as pd
import numpy as np
from phaser.similarities import (
    find_inter_samplesize,
    IntraDistance,
    InterDistance,
    validate_metrics,
    DISTANCE_METRICS,
)


def list_modular_components():

    # Get hashes - checks each item in phaser.hashing._algorithms and checks to see if the class is a subclass
    # of the abstract class PerceptualHash. If it is, include it in the list of hashes.
    hashes = []
    for name in dir(phaser.hashing):
        try:
            if issubclass(getattr(phaser.hashing, name), phaser.hashing.PerceptualHash):
                hashes.append(name)
        except TypeError as err:
            pass

    # Get the list of transformers in the same way, except look in phaser.transformers._transforms
    # and check for the phaser.transformers._transforms.Transformer class.
    transformers = []
    for name in dir(phaser.transformers):
        try:
            if issubclass(
                getattr(phaser.transformers, name), phaser.transformers.Transformer
            ):
                if name != "Transformer":
                    transformers.append(name)
        except TypeError:
            pass

    builtin_distance_metrics = (
        scipy.spatial.distance._METRICS_NAMES
    )  # Not sure there's a better way to get these.
    comparison_metrics = DISTANCE_METRICS
    return {
        "Hashes": hashes,
        "Transformers": transformers,
        "Scipy Built-in Distance Metrics": builtin_distance_metrics,
        "Custom Distance Metrics": comparison_metrics,
    }


def do_hashing(
    originals_path: str,
    algorithms: dict,
    transformers: list,
    distance_metrics: dict,
    output_directory: str,
    n_jobs=-1,
    progress_report: bool = True,
    batch_size=100_000,
) -> None:

    # Get list of images
    imgpath = originals_path
    list_of_images = [str(i) for i in pathlib.Path(imgpath).glob("**/*")]

    print(f"Found {len(list_of_images)} images in {os.path.abspath(imgpath)}.")
    print(f"Creating output directory at {os.path.abspath(output_directory)}...")
    pathlib.Path(output_directory).mkdir(exist_ok=True)

    if len(list_of_images) > batch_size:
        print(
            f"Dataset is large, splitting in to smaller chunks (batch size={batch_size})."
        )

    batches = create_file_batches(list_of_images, batch_size)
    num_batches = len(batches)

    compression_opts = dict(method="bz2", compresslevel=9)
    print("Doing hashing...")
    for bat in range(0, num_batches):
        df_h = None
        if num_batches > 1:
            print(f"Batch {bat}...")
        ch = phaser.hashing._helpers.ComputeHashes(
            algorithms, transformers, n_jobs=n_jobs, progress_bar=progress_report
        )
        df_h = ch.fit(batches[bat])

        # Dump temporary hashes, if any
        if num_batches > 1:
            outfile = os.path.join(output_directory, format_temp_name(bat))
            df_h.to_csv(
                outfile, index=False, encoding="utf-8", compression=compression_opts
            )

    if num_batches > 1:
        # Create a single dataframe from the intermediatte dataframes
        print("Combining temporary csv files...")
        df_h = combine_temp_hash_dataframes(
            output_directory=output_directory, num_batches=num_batches
        )

    # Create and fit LabelEncoders according to experiment
    le = {
        "f": LabelEncoder().fit(df_h["filename"]),
        "t": LabelEncoder().fit(df_h["transformation"]),
        "a": LabelEncoder().fit(list(algorithms.keys())),
        "m": LabelEncoder().fit(list(distance_metrics.keys())),
        "c": LabelEncoder(),
    }

    # Hard-code class labels for use when plotting
    le["c"].classes_ = np.array(["Inter (0)", "Intra (1)"])

    # Apply LabelEncoder on df_h
    df_h["filename"] = le["f"].transform(df_h["filename"])
    df_h["transformation"] = le["t"].transform(df_h["transformation"])

    # Dump LabelEncoders and df_h to disk
    print(
        "Saving hashes.csv.bz2 and labels for filenames (f), algorithms (a), transforms (t), and metrics (m) to bzip files.."
    )
    # dump_labelencoders(le, path=output_directory)
    dump(
        value=le,
        filename=os.path.join(output_directory, "LabelEncoders.bz2"),
        compress=9,
    )
    outfile = os.path.join(output_directory, "Hashes.csv.bz2")
    df_h.to_csv(outfile, index=False, encoding="utf-8", compression=compression_opts)

    print("Saving compressed hash DataFrame to Hashes.df.bz2...")
    # Also save compressed dataframe, this makes it faster to load in larger datasets.
    outfile = os.path.join(output_directory, "Hashes.df.bz2")
    dump(value=df_h, filename=outfile, compress=9)


def calculate_distances(
    distance_metrics: list,
    hash_directory: str = "",
    load_df=True,
    hash_dataframe: pd.DataFrame = None,
    label_encodings: dict = {},
    progress_report: bool = True,
    sample_files: int = 0,
    remove_bad_hash: bool = True,
    out_dir: str = "",
    save_to_disk=True,
) -> pd.DataFrame:
    if hash_directory:
        # A hash directory is specified, load the hash dataframe and the enodings from here, rather than from memmory.
        # Read the precomputed hashes from hashes.csv.bz2
        if load_df:
            df_path = os.path.join(hash_directory, "Hashes.df.bz2")
            df_h = load(df_path)
            print(f"DataFrame loaded from {os.path.abspath(df_path)}")
        else:
            csv_path = os.path.join(hash_directory, "Hashes.csv.bz2")
            df_h = pd.read_csv(csv_path)
            print(f"DataFrame loaded from {os.path.abspath(csv_path)}")

        # Load the Label Encoders used when generating hashes
        le = load(os.path.join(hash_directory, "LabelEncoders.bz2"))
    else:
        # Work with a DataFrame and encodings which are already in memory

        if (type(hash_dataframe) == type(None)) or (label_encodings == dict({})):
            raise Exception(
                "Must provide both hash dataframe and label encoding, or a directory to read them from."
            )
        df_h = hash_dataframe
        le = label_encodings

    # Validate distance metrics, or throw an Exception.
    validate_metrics(distance_metrics)

    if remove_bad_hash:
        # Find hashes that sum to 0 since they can cause issues with distance metrics
        for a in df_h.columns[2:]:
            mask = df_h[a].apply(lambda x: sum(x)) == 0
            bad_filenames = df_h[mask]["filename"].unique()
            if progress_report:
                print(f"{len(bad_filenames)} bad hashes found for {a}")

            if len(bad_filenames) > 0:
                df_h = df_h[~df_h["filename"].isin(bad_filenames)]

    # If a sub-sample of the files have been chosen, subset the data.
    # e.g 100 files have been chosen as the subset, present all hashes relating to those 100 files
    # The total number of rows would then be 100 * len(TRANSFORMS)
    if sample_files:
        # Pick the samples
        unique_filenames = df_h["filename"].unique()
        unique_filenames = np.random.choice(
            unique_filenames, size=sample_files, replace=False
        )

        # Subset the data
        df_subset = df_h[df_h["filename"].isin(unique_filenames)]
        if progress_report:
            print(f"Sampled {sample_files} files.")
    else:
        df_subset = df_h

    # Compute the intra distances
    intra = IntraDistance(distance_metrics, le, progress_bar=progress_report)
    intra_df = intra.fit(df_subset)
    if progress_report:
        print(f"Number of total intra-image comparisons = {len(intra_df)}")

    # Compute the inter distances using subsampling
    n_samples = find_inter_samplesize(len(df_h["filename"].unique() * 1))
    inter = InterDistance(
        distance_metrics, le, n_samples=n_samples, progress_bar=progress_report
    )
    inter_df = inter.fit(df_subset)

    if progress_report:
        print(f"Number of pairwise comparisons = {inter.n_pairs_}")
        print(f"Number of total inter distances = {len(inter_df)}")

    # Combine distances
    dist_df = pd.concat([intra_df, inter_df])
    compression_opts = dict(method="bz2", compresslevel=9)

    if save_to_disk:
        # Default to the same directory as the hashes if no output directory is specified
        if not out_dir:
            out_dir = hash_directory
        distance_path = os.path.join(out_dir, "Distances.csv.bz2")
        print(f"Saving distance scores to {os.path.abspath(distance_path)}.")
        # Save as compressed CSV
        dist_df.to_csv(
            distance_path, index=False, encoding="utf-8", compression=compression_opts
        )
        # Save as compressed DF
        distance_path = os.path.join(out_dir, "Distances.df.bz2")
        dump(value=dist_df, filename=(distance_path), compress=9)
    return dist_df


def main():
    print("Test listing of modular components...")
    nl = "\n"
    for module_name, functions in list_modular_components().items():
        print(f"{module_name}:{nl}{nl.join(functions)}")
        print(nl)


if __name__ == "__main__":
    main()
