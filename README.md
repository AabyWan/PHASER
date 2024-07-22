# PHASER
A Perceptual Hashing Algorithms Evaluation and Results framework.

The framework has a corresponding paper published at DFRWS EU 2023:
McKeown, S., Aaby, P., & Steyven, A. (2024). PHASER: Perceptual hashing algorithms evaluation and results-An open source forensic framework. Forensic Science International: Digital Investigation, 48, 301680.

The library simplifies much of the process of performing Perceptual Hashing (Semantic Approximate Matching in the image domain) experiments and evaluations by wrapping standard Python Data Science libraries. It allows for the easy manipulation of base images to produce variants via transformations (such as crop, rotate, flip), and mechanisms for specifying arbitrary Perceptual Hashes and Distance metrics.
Wrappers are also provided to make the evaluation and visualisation straightforward, while being extensible.

### Notes and Caveats
- Note that PHASER uses the Python Image Library (PIL) to load and manipulate images. PIL ignores EXIF rotation metadata, which may cause issues if transforms are imported from disk and their EXIF rotation does not match the original image. PHASER is internally consistent, though, so it is only a problem for imported transforms.

## Contents

### The Library

The PHASER library itself is located in the **phaser** folder, which contains modules for creating hashes, calculating similarities, evaluating results, and plotting relevant graphs. 
These modules simplify the process of handling the data and expose a few high-level functions which coordinate most of the work, usually in the _helpers.py of each module.

The demonstration code provided at the top-level of this Github illustrates how to call these functions. We leverage common Data Science libraries and conventions throughout. For example, label encoders are used to reduce the size of the data when stored on disk.

The execution stages are essentially:
1. **Configure/define Perceptual Hash Algorithms**. See demo00_conf.py
2. **Generate hashes** for base images and the specified transforms, for a specified list of hashes. See demo01_hashing.py
	- The **phaser.hashing._helpers.ComputeHashes(perceptual_algs, transforms)** function streamlines this process. 
	- Create the object, specifying perceptual hashes to test, and image transforms, and then call .fit(my_files) to process.
	- Save the hashes and labels.
3. **Calculate distances** between hashes. See demo02_similarities.py
	- **phaser.similarities.IntraDistance** and **phaser.similarities.InterDistance** create objects with parameters for the distance metric (e.g. Hamming distance), and label encodings.
	- Call .fit() and pass in the DataFrame of hashes from step 1.
	- Intra-distances are between the original image and the transform, while Inter-distances are between different base images in the dataset (as well comparing their transform-transform counterparts).
	- Save scores to a DataFrame, update label encoding for distance metrics if necessary.
4. **Calculate evaluation metrics** See demo03_eval.py
	- Much of the work here is handled by **phaser.evaluation.ComputeMetrics**, which just requires the DataFrames for hashes and distances, as well as labels. It then creates triplets of hash/transform/distance_metric and evaluates each in turn.
	-  ComputeMetrics relies on **phaser.evaluation.MetricMaker** behind the scenes, which generates the confusion matrix and other statistics. Values are weighted appropriately when the number of intra/inter samples are different.
	- Note: The data science libraries and conventions lend themselves better to calculating statistics based on **similarity* rather than distance. So distances are flipped at this stage, but the negation (distance) is simply calculated by 1-similarity.
	- Optional: Perform bit-weight optimisation calculations. (Best exemplified by 3_bit-analysis.pynb).
5. **Plot graphs**. See demo04_plotting.py
	- The **phaser.plotting** module leverages the seaborn library, providing support for histograms, KDE plots, ROC  curves, confusion matrices, EER plots, and bit-weighting/AUC comparison for bit-weighting.

#### Extensibility
There are three things that we may wish to manipulate in Perceptual Hashing experiments which are easily specified here:

- Perceptual Hash (e.g., pHash, PDQ), see phaser.hashing._algorithms.py
- Distance Metrics (e.g., Hamming Distance, Cosine Distance), see phaser.similarities._distances.py
- Image Transform (e.g., Crop, Rotate), see phaser.transformers._transforms.py

The SciPy library is leveraged to provide many distance metrics already, and some Transforms are provided out of the box. However, any one of these items can be specified for inclusion in the framework as long as they have a class or function defined in the appropriate place, noted above.
The limitation with Perceptual Hash functions is that they should take a Python PIL.Image object as an argument, rather than a file path. This is due to the way the helper (phaser.hashing._helpers.sim_hashing) implements the wrapper for calling hashes, which could be modified.
Distance metrics should follow the parameter conventions for the functions in Scipy.Spatial.Distance, i.e., accepting a u and v array-like object of values corresponding to the hash bits, with an optional w parameter for specifying a weighting (for the bit-weighting functionality).
Image Transforms are similar to hashes, in that they should take a Python PIL.Image object as the primary argument (and perform a deepcopy of the object prior to manipulating it).

Additional evaluation metrics can be added easily by extending **phaser.evaluation.MetricMaker**.

### Demonstration Code and Notebooks

The top-level directory contains several files which demonstrate PHASER's functionality. The main steps are outlined in the demo0x files, with all of the necessary steps and function calls laid out.

Three Jupyter Notebook files are also provided which simplify the process and bypass most of the need to write any code (aside from specifying transforms and function imports in the configuration). Most of the code for the first notebook is hidden in notebooksupport.py to keep it clean.

**1_process.ipynb**: can be used standalone to generate hashes and similarities, saving them to compressed DataFrame and csv files. All that is required is a dataset location and imports for the necessary perceptual hashing/distance metric/transform functions, some of which are specified by default.

**2_analyse.ipynb**  can similarly facilitate a hands off approach to visualising the results. It provides interactive capabilities to view inter/intra distance, scores and false positive/negative rates through histogram, KDE, EER and ROC plots.

**3_bit-analysis.ipynb** is a bit more hands on, but allows for optimising hash-bit weights and re-evaluating these compared to the original run in notebook 2.
An alternative version (prior to some backend updates) is provided in the Paper folder, which has the outputs used in the experiment found in the paper.

#### Interpreting Results
The idea is that the inter- and intra-comparison classes should be neatly separated if the hashing  algorithm and distance metric pair are to function as discriminators between matches and non-matches. This discrimination is achieved by selecting a similarity/distance threshold for what constitutes a "match". i.e., if we set a similarity threshold of 0.9, we would consider any image with a similarity >= 0.9 to be a match for our source image.

The KDE plot is a good indicator of general behaviour. Ideally the inter-score distribution should be narrow and centred around 0.5 similarity, while the intra-scores (should match), should be as close to 1 similary (or 0 distance) as possible, while also not overlapping with the intra-scores.
Any overlap indicates that the distributions cannot be completely separated, which will result in false positives (the match happened by chance due to the inter-score distribution infiltrating into the threshold area) and false negatives (the image was outside of our threshold area because the threshold is too tight, perhaps to avoid excessive false positives).

In some cases it is clear that a given algorithm cannot function for a given transform, for example, when the two distributions almost overlap completely. In this case, no threshold would be viable.

### Paper Folder

The paper folder has the code for the two short experiments located in the paper as a proof of concept, namely, **dataset size** evaluation and **bit-weight analysis**.

- **sample_experiment.py** assumes that the hashes have already been calculated (using, for example, the provided notebook). The original experiment used the Flickr 1 Million dataset.
	- The program then sub-samples the hashes a specified number of times, generating an aggregate file to keep track of all evaluation results from MetricMaker on each run.
	- The values are then plotted and analysed to gauge variance between runs for different sample sizes.
- **_03-demo-bit-analysis.ipynb** this notebook was used in the original bit-weighting experiment, but doesn't run as-is with the current version of the PHASER library (see the top-level example notebook instead). It does however contain the original results from the experiment.
	- Essentially, the evaluation is run as normal, but then optimal weights are calculated for the hash-distance metric pairs across all transforms. This is then used to re-evaluate the hash-distance pair, which is compared to the original results via an AUC (area under curve) plot.

In both cases these files should work with any set of hashes, though the sample size will need to be adjusted accordingly

