# Paper Notes

This section houses the code for the associated paper at DFRWS EU 2025:
McKeown, S. (2025). Beyond Hamming Distance: Exploring spatial encoding in perceptual hashes. Forensic Science International: Digital Investigation, 52, 301878. https://doi.org/10.1016/j.fsidi.2025.301878


This work had two main stages:
- Testing spatial data encoding in perceptual hashes. Essentially, analysing to what extent positional information from an image is kept by the hash, if any. Hamming distance, the defacto distance metric, does not exploit any positional/localised hash features.
- Evaluate new/proposed distance metrics which are aware of this additional information in perceptual hashes. From the prior section, two main behaviours were noted: spatial hashes and DCT (frequency transform domain). For the former, **2D ngram distance** and **normalised convolution distance** were tested, while the latter was targeted by a DCT-coefficient aware approach called **hatched matrix distance**.

## Dataset

The paper uses the Flickr 1 Million Dataset.
- For spatial encoding experiments: 20k images are randomly sampled from the full list of Flickr 1 Million files. This is achieved using a fixed RNG value of 42, which is passed to np.random.seed and subsequently used in np.random.choice.
- A fixed random sample of 250k images for metric evaluation. This is a fixed set (see file_list_250k.txt) which was generated in prior work, and allowed for the re-use of existing transformations on disk.
- Note that to use the code here, the Flickr 1 Million Dataset needs to be unpacked and flattened so that all files are in a single directory (as opposed to the folder 0..1).

## Files

- file_list_250k.txt: The 250k sample of Flickr 1 Million used for the metric evaluation.
- metric_evalaution_experiment.py: Generate the dataset and evaluate the metrics at-scale using the fixed set of 250k Flickr 1 Million images. Allows for replication of the evaluation.
- spatial_encoding_experiment.py: The full experiment for generating various transformations which have positional changes (e.g. borders, watermarks), hashing them, calculating distances, weights, and graphs. Enables re-running this part of the experiment.
- timed_benchmarks.py: Run timed benchmarks for the various distance metrics.

## Neuralhash

- Neuralhash was included in the experiment. Neuralhash resources taken from iPhone14,5_15.3.1_19D52_Restore.ipsw using the instructions from https://github.com/AsuharietYgvar/AppleNeuralHash2ONNX.
- These can't be distributed due to copyright. After extracting the neuralhash_128x96_seed1.dat file and generating model.onnx, place them in the phaser/resources folder (with the watermark and other static images)

