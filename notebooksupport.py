
import pathlib, os, phaser.hashing, phaser.transformers
from phaser.utils import ImageLoader as IL
from phaser.utils import dump_labelencoders, load_labelencoders, bin2bool

# for do hashing and comparisons
from sklearn.preprocessing import LabelEncoder

# for list modules
import scipy.spatial.distance 

# for comparisons
import pandas as pd
import numpy as np
from phaser.similarities import find_inter_samplesize, IntraDistance, InterDistance, validate_metrics, DISTANCE_METRICS

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
            if issubclass(getattr(phaser.transformers, name), phaser.transformers.Transformer):
                if name != "Transformer":
                    transformers.append(name)
        except TypeError:
            pass

    builtin_distance_metrics = scipy.spatial.distance._METRICS_NAMES# Not sure there's a better way to get these.
    comparison_metrics = DISTANCE_METRICS
    return {"Hashes": hashes, "Transformers": transformers, "Scipy Built-in Distance Metrics": builtin_distance_metrics, "Custom Distance Metrics": comparison_metrics}


def do_hashing(originals_path:str, algorithms:dict, transformers:list, distance_metrics:dict, output_directory:str, n_jobs=-1, progress_report:bool=True) -> None:

    # Get list of images
    imgpath = originals_path
    list_of_images = [str(i) for i in pathlib.Path(imgpath).glob('**/*')]

    print(f"Found {len(list_of_images)} images in {os.path.abspath(imgpath)}.")
    
    print(f"Creating output directory at {os.path.abspath(output_directory)}...")
    pathlib.Path(output_directory).mkdir(exist_ok=True)

    print("Doing hashing...")
    ch = phaser.hashing._helpers.ComputeHashes(algorithms, transformers, n_jobs=n_jobs, progress_bar=progress_report)
    df_h = ch.fit(list_of_images)

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
    dump_labelencoders(le, path=output_directory)

    # Dump the dataset
    print("Saving hashes.csv and labels for filenames (f), algorithms (a) and transforms (t) to bzip files..")
    compression_opts = dict(method='bz2', compresslevel=9)
    outfile = os.path.join(output_directory, "hashes.csv.bz2")
    df_h.to_csv(outfile, index=False, encoding='utf-8', compression=compression_opts)

def calculate_distances(hash_directory:str, distance_metrics:list, progress_report:bool=True, sample_files:int=0, out_dir:str="") -> None:

    # Read the precomputed hashes from hashes.csv.bz2
    csv_path = os.path.join(hash_directory, "hashes.csv.bz2")
    df_h = pd.read_csv(csv_path)
    print(f"Dataframe loaded from {os.path.abspath(csv_path)}")

    # Load the Label Encoders used when generating hashes
    le = load_labelencoders(filename="LabelEncoders", path=hash_directory)

    # Convert binary hashes to boolean for distance computation
    for a in le["a"].classes_:
        df_h[a] = df_h[a].apply(bin2bool)



    # print(f"Algorithms: \t {', '.join(ALGORITHMS)}")
    # print(f"Transforms: \t {', '.join(TRANSFORMS)}")
    # print(f"Metrics: \t {', '.join(METRICS)}")

    # Validate distance metrics, or throw an Exception.    
    validate_metrics(distance_metrics)

    # If a sub-sample of the files have been chosen, subset the data.
    # e.g 100 files have been chosen as the subset, present all hashes relating to those 100 files
    # The total number of rows would then be 100 * len(TRANSFORMS)
    if sample_files:
        # Pick the samples
        unique_filenames = df_h["filename"].unique()
        unique_filenames = np.random.choice(unique_filenames, size=sample_files, replace=False)
        
        # Subset the data
        df_h = df_h[df_h["filename"].isin(unique_filenames)]
        print(f"Sampled for {sample_files} files.")

    # Compute the intra distances
    intra = IntraDistance(distance_metrics, le, 1, progress_bar=True)
    intra_df = intra.fit(df_h)
    print(f"Number of total intra-image comparisons = {len(intra_df)}")

    # Compute the inter distances using subsampling
    n_samples = find_inter_samplesize(len(df_h["filename"].unique() * 1))
    inter = InterDistance(distance_metrics, le, set_class=0, n_samples=n_samples, progress_bar=True)
    inter_df = inter.fit(df_h)

    print(f"Number of pairwise comparisons = {inter.n_pairs_}")
    print(f"Number of total inter distances = {len(inter_df)}")

    # Combine distances and save to disk
    dist_df = pd.concat([intra_df,inter_df])
    compression_opts = dict(method='bz2', compresslevel=9)

    # Default to the same directory as the hashes if no output directory is specified
    if not out_dir:
        out_dir = hash_directory
    distance_path = os.path.join(out_dir, "distances.csv.bz2")
    print(f"Saving distance scores to {os.path.abspath(distance_path)}.")
    dist_df.to_csv(distance_path, index=False, encoding='utf-8', compression=compression_opts)


def main():
    print("Test listing of modular components...")
    nl = '\n'
    for module_name, functions in list_modular_components().items():
        print( f"{module_name}:{nl}{nl.join(functions)}")
        print(nl)


if __name__ == "__main__":
    main()
