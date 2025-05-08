# Paper Notes

This section houses the code for the associated paper at DFRWS EU 2024:
McKeown, S., Aaby, P., & Steyven, A. (2024). PHASER: Perceptual hashing algorithms evaluation and results-An open source forensic framework. Forensic Science International: Digital Investigation, 48, 301680. https://doi.org/10.1016/j.fsidi.2023.301680


This work had two main stages:
- Calculate classification evaluation metrics for various dataset sample sizes and plot graphs to determine how large a dataset should be when performing perceptual hash experiments.
- Run a bit-weight analysis of hashes to determine if re-weighting hashes can be an effective way of boosting classification performance. This component has largely been superseded by follow-up work:
    McKeown, S. (2025). Beyond Hamming Distance: Exploring spatial encoding in perceptual hashes. Forensic Science International: Digital Investigation, 52, 301878. https://doi.org/10.1016/j.fsidi.2025.301878
## Dataset

The paper uses the Flickr 1 Million Dataset.
- Note that to use the code here, the Flickr 1 Million Dataset needs to be unpacked and flattened so that all files are in a single directory (as opposed to the folder 0..1).

## Files

- sample_experiment.py: contains the code for the sampling experiment to test metric stability for different dataset sizes
- _03-demo-bit-analysis.ipynb: The original notebook used for the bit-weighting experiment. Redundant now, see 3_bit-analysis.ipynb in the root of the PHASER Github as this now contains extended functionality. The notebook provided here is for archival purposes and refers to deprecated/modified library elements.
