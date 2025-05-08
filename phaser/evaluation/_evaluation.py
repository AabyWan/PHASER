import numpy as np
import pandas as pd

from sklearn.utils import compute_sample_weight
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# For EER calculations
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from joblib import Parallel, delayed
from tqdm import tqdm


def calc_eer(fpr: np.ndarray, tpr: np.ndarray, threshold: np.ndarray):
    """
    Discovers the threshold where FPR and FRR intersects.

    Parameters
    ----------
    fpr : np.ndarray
        Array with False Positive Rate from sklearn.metrics.roc_curve
    tpr : np.ndarray
        Array with True Positive Rate from sklearn.metrics.roc_curve
    threshold : np.ndarray
        Array with thresholds from sklearn.metrics.roc_curve

    Returns
    -------
    floats, float
        eer_score, eer_threshold


    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([1, 1, 2, 2])
    >>> scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
    >>> eer_score, eer_threshold = calc_eer(fpr, tpr, thresholds)


    """
    # Implementation from -> https://yangcha.github.io/EER-ROC/
    # first position is always set to max_threshold+1 (close to 2) by sklearn,
    # overwrite with 1.0 to avoid EER threshold exceeding 1.0.
    # threshold[0] = 1.0
    eer_score = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    eer_thresh = interp1d(fpr, threshold)(eer_score)

    return eer_score, float(eer_thresh)


def pred_at_threshold(y_scores, threshold, pos_label=1):
    # Make predictions based on a specific decision threshold
    # y_scores : array with predicted probabilities, similarities, or distances.
    # threshold : the specified threshold to seperate the two classes.
    # pos_label : integer defining the positive class.
    if pos_label == 0:
        return np.array((y_scores <= threshold)).astype(int)
    else:
        assert pos_label == 1
        return np.array((y_scores >= threshold)).astype(int)


class MetricMaker:
    def __init__(self, y_true: list, y_similarity: list, weighted=True) -> None:
        """
        Compute performance metrics using the ground thruth and similarity scores

        Parameters
        ----------
        y_true : list
            The ground truth values.
        y_similarity : list
            The similarity scores for each ground truth
        weighted : bool, optional
            Whether to weight class distribution when computing metrics, by default True
        """
        self.y_true = y_true
        self.y_sims = y_similarity
        self.weighted = weighted

        # call the fit function when instantiated
        self._fit()

    def _fit(self):
        # Create balanced sample weights for imbalanced evaluation
        if self.weighted:
            self.smpl_w = compute_sample_weight(class_weight="balanced", y=self.y_true)
        else:
            self.smpl_w = None

        # Compute the FPR, TPR, and Thresholds
        self.fpr, self.tpr, self.thresholds = roc_curve(
            y_true=self.y_true, y_score=self.y_sims, sample_weight=self.smpl_w
        )

        # Compute the AUC score
        self.auc = auc(self.fpr, self.tpr)
        self.eer_score, self.eer_thresh = calc_eer(self.fpr, self.tpr, self.thresholds)

    def get_fpr_threshold(self, max_fpr: float) -> float:
        """
        Find the decision threshold at a certain max False Positive Rate (FPR)

        Parameters
        ----------
        max_fpr : float
            Desired maximum FPR value

        Returns
        -------
        float
            Decision threshold at the desired FPR
        """
        return np.interp(max_fpr, self.fpr, self.thresholds)

    def get_cm(self, threshold: float, normalize="true", breakdown=False) -> np.ndarray:
        """
        Compute and returns the Confusion Matrix at a certain decision threshold

        Parameters
        ----------
        threshold : float
            The decision threshold used to make predictions
        normalize : str, optional
            normalize the confusion matrix, by default "true"
            Options: 'true', 'none'
        breakdown : bool, optional
            Instead returns [tn, fp, fn, tp], by default False

        Returns
        -------
        np.ndarray
            Confusion Matrix with tn, fp, fn, tp as a single matrix.
        """
        # an ugly patch to allow passing 'none' to sklean arg
        if normalize == "none":
            normalize = None

        # Setting y_pred for use in later evaluation
        self.y_pred = pred_at_threshold(self.y_sims, threshold, pos_label=1)

        cm = confusion_matrix(
            y_true=self.y_true,
            y_pred=self.y_pred,
            sample_weight=self.smpl_w,
            normalize=normalize,
        )

        if breakdown:
            return cm.ravel()

        else:
            return cm


def makepretty(styler, **kwargs):
    # https://pandas.pydata.org/docs/user_guide/style.html#Styler-Object-and-Customising-the-Display
    title = kwargs["title"]
    styler.set_caption(f"Stats for '{title}'")

    styler.format(precision=4, thousands=".", decimal=",")
    styler.background_gradient(
        axis=None, subset=["25%", "75%"], vmin=0, vmax=1, cmap="Greys"
    )
    styler.hide(subset=["count"], axis=1)
    styler.format_index(str.upper, axis=1)

    return styler


def dist_stats(data, le, transform, style=True):
    stats = data.groupby(["algo", "metric"])[transform].describe().reset_index()
    stats["algo"] = le["a"].inverse_transform(stats["algo"])
    stats["metric"] = le["m"].inverse_transform(stats["metric"])

    if style:
        stats = stats.style.pipe(makepretty, title=transform)

    return stats


def _bit_flip(
    q: pd.DataFrame,
    df_h: pd.DataFrame,
    orig_label: int,
    t_l: int,
    a_s: str,
    bit_stay: bool,
):
    """
    Vectorized comparison of bit changes between hashes

    Parameters
    ----------
    q : pd.DataFrame
        Subset dataframe containing distances for the CM quadrant (q)
    df_h : pd.DataFrame
        The overall dataframe with the hashes
    orig_label : int
        Original label corresponding to the 'orig' string
    t_l : int
        Transformation label corresponding to the transformation string
    a_s : str
        Algorigthm string value
    bit_stay : bool
        True if the positive case causes bits to stay, else set to False.

    Returns
    -------
    np.ndarray
        Array with a total count of 'good' bit outcomes
    """
    # Shortnames for column on left and right - A,B pairs
    fA = q["fileA"]  # Order of file A
    fB = q["fileB"]  # Order of file B

    # if the bit should stay, comparing orig->orig_transform
    if bit_stay:
        # Subset fileA according to original transform
        u = df_h.query(f"filename in @fA and transformation == {orig_label}")

        # Keep order of filenames accordingly and get the hashes
        u = u.set_index("filename").loc[fA].reset_index()[a_s].values

    else:  # if the bit should flip, comparing inter-transform
        # Subset fileA according to target transform
        u = df_h.query(f"filename in @fA and transformation == {t_l}")
        u = u.set_index("filename").loc[fA].reset_index()[a_s].values

    # Subset fileB according to the other transform
    v = df_h.query(f"filename in @fB and transformation == {t_l}")
    v = v.set_index("filename").loc[fB].reset_index()[a_s].values

    # Stack rows (rows, bits)
    u = np.row_stack(u)
    v = np.row_stack(v)

    # Vectorized comparison
    if bit_stay:
        good_bits = u == v
    else:
        good_bits = u != v

    return good_bits


class BitAnalyzer:
    def __init__(self, df_h: pd.DataFrame, le: dict) -> None:
        """
        Interface for analysing bit changes on triplet subsets.

        Parameters
        ----------
        df_h : pd.DataFrame
            Dataframe containing the all or a subset of raw hashes
        le : dict
            Dictionary with LabelEncoders

        Example use:
        ------------
        Assuming a subset dataframe with a defined distances
        >>> from phaser.evaluation import MetricMaker, BitAnalyzer
        >>> y_true = subset['class']
        >>> y_sims = subset[t_s]
        >>> mm = MetricMaker(y_true, y_sims, weighted=False)
        >>> cm = mm.get_cm(mm.eer_thresh, normalize='none')
        >>> BA = BitAnalyzer(df_hashes, le)
        >>> BA.fit(subset, mm.y_pred, t_l, a_s)
        """
        # Dataframe with all the orignal hashes to compare
        self.df_h = df_h
        self.le = le

    # Analyse a subset of data
    def fit(self, subset: pd.DataFrame, y_pred: np.ndarray, t_l: int, a_s: str):
        """
        Analyse bits on a given subset of data defined by a triplet.
        The triplet is defined by [algorithm, transform, metric]

        Parameters
        ----------
        subset : pd.DataFrame
            A subset containing the data for the selected triplet
        y_pred : np.ndarray
            Predictions generated for the triplets using MetricMaker
            y_pred = mm.y_pred
        t_l : int
            Integer label for the transform to analyse
        a_s : str
            String for the hashing algorithm to analys

        Returns
        -------
        _type_
            _description_
        """
        hashes_bit_length = len(self.df_h.iloc[0][a_s])

        # Get int label for column name with 'orig' transform
        orig_label = np.where(self.le["t"].classes_ == "orig")[0][0]

        # Subset each quadrants in Confusion Matrix (CM)
        cm = {}
        cm["FN"] = subset[(subset["class"] == 1) & (subset["class"] != y_pred)]
        cm["TP"] = subset[(subset["class"] == 1) & (subset["class"] == y_pred)]
        cm["FP"] = subset[(subset["class"] == 0) & (subset["class"] != y_pred)]
        cm["TN"] = subset[(subset["class"] == 0) & (subset["class"] == y_pred)]

        # Analyse each quadrant in the CM
        for key, val in cm.items():
            # Only process if quadrant has values
            if len(val) > 0:
                # Bit_stay=True for FN and TP
                if key in ["FN", "TP"]:
                    good_bits = _bit_flip(
                        cm[key], self.df_h, orig_label, t_l, a_s, True
                    )
                    cm[key] = np.row_stack(good_bits).sum(axis=0) / len(val)

                # Else, Bit_stay=False for FP and TN
                else:
                    good_bits = _bit_flip(
                        cm[key], self.df_h, orig_label, t_l, a_s, bit_stay=False
                    )
                    cm[key] = np.row_stack(good_bits).sum(axis=0) / len(val)

            # Else, when the quadrant is empty
            else:
                # No TRUE values, no bit's provide info
                # Set all bits to zero?
                if key in ["TP", "TN"]:
                    cm[key] = np.repeat(0.0, hashes_bit_length)

                else:
                    # No FALSE values, all bit's may be usefull
                    cm[key] = np.repeat(1.0, hashes_bit_length)

        return pd.DataFrame(cm)


# Create a wrapper parallel metrics
class ComputeMetrics:
    def __init__(
        self,
        le: dict,
        df_d: pd.DataFrame,
        df_h: pd.DataFrame,
        analyse_bits=False,
        n_jobs=1,
        backend="loky",
        progress_bar=True,
        decision_thresholds={}
    ) -> None:
        """
        Compute performance metrics for triplets using JobLib for parallel processing

        Parameters
        ----------
        le : dict
            Dictionary containing LabelEncoders
            Required key values for encoders
        df_d : pd.DataFrame
            Dataframe with distances
        df_h : pd.DataFrame
            Dataframe with hash values
        analyse_bits : bool, optional
            Whether to analyse bit frequency, by default False
        n_jobs : int, optional
            JobLib flag, use all cores, by default -1
        backend : str, optional
            JobLib flag, change backend, by default "loky"
        progress_bar : bool, optional
            Show progress bar using TQDM, by default True
        decision_thresholds : dict, optional
            A nested dictionary, outer keys for algorithm, inner keys for metrics, values are similarity decision thresholds in float between 0 and 1.
 """
        self.le = le
        self.df_d = df_d
        self.df_h = df_h
        self.analyse_bits = analyse_bits
        self.n_jobs = n_jobs
        self.backend = backend
        self.progress_bar = progress_bar
        self.decision_thresholds = decision_thresholds
        self.y_pred = None

    def _process_triplet(self, triplet, normalize, weighted):
        # string values for algo, transf, metric
        a_s, t_s, m_s = triplet

        if normalize:
            normalize = "true"
        else:
            normalize = None

        # from string to integer label encoding
        a_l = self.le["a"].transform(np.array(a_s).ravel())[0]
        t_l = self.le["t"].transform(np.array(t_s).ravel())[0]
        m_l = self.le["m"].transform(np.array(m_s).ravel())[0]

        # subset the triplet data
        subset = self.df_d[(self.df_d["algo"] == a_l) & (self.df_d["metric"] == m_l)]

        y_true = subset["class"]
        y_sims = subset[t_s]

        # use the metric maker to
        mm = MetricMaker(y_true, y_sims, weighted=weighted)
        if self.decision_thresholds:
            a_dict = self.decision_thresholds.get(triplet[0]) # Get the algorithm values
            thresh = a_dict.get(triplet[2]) # get the value for this metric
        else: # No threshold provided, use EER threshold.
            thresh = mm.eer_thresh
        cm = mm.get_cm(thresh, normalize, True)

        # Create metricmaker columns. Note, threshold is inverted to similarity in analysis, and here.
        m = [a_s, t_s, m_s, mm.auc, mm.eer_score, mm.eer_thresh, *cm, thresh]

        # set predictions as a property so they can be accessed externally.
        self.ypred = mm.y_pred
        
        if self.analyse_bits:
            BA = BitAnalyzer(df_h=self.df_h, le=self.le)

            # BA.fit() -> pd.DataFrame
            bits = BA.fit(subset, mm.y_pred, t_l, a_s)
            return m, bits

        else:
            return m, None

    def fit(self, triplets, normalize="true", weighted=True):
        """
        Fit object on a list of triplets using JobLib

        Parameters
        ----------
        triplets : list
            List of triplets
        weighted : bool, optional
            Apply weighting='balanced' to ConfusionMatrix, by default False

        Returns
        -------
        metrics, bit-weights
            Returns a tuple
        """
        # Use zip() to return tuple (m, b) from process_triplet
        _m, _b = zip(
            *Parallel(n_jobs=self.n_jobs, backend=self.backend)(
                delayed(self._process_triplet)(t, normalize, weighted)
                # TODO: fix the progress bar :/
                for t in tqdm(triplets, desc="Triplet", disable=not self.progress_bar)
            )
        )

        # post-process metrics
        m = np.row_stack(_m)  # type:ignore
        cols = [
            "Algorithm",
            "Transform",
            "Metric",
            "AUC",
            "EER",
            "EER-Threshold",
            "TN",
            "FP",
            "FN",
            "TP",
            "Decision_Threshold"
        ]
        m = pd.DataFrame(m, columns=cols)
        m[m.columns[3:]] = m[m.columns[3:]].astype(float)

        # post-process bitweights
        if self.analyse_bits:
            b = {}
            for i, t in enumerate(triplets):
                t_str = f"{t[0]}_{t[1]}_{t[2]}"
                b[t_str] = _b[i]
        else:
            b = dict()

        return m, b


def make_bit_weights(bitfreq: dict, le: dict) -> dict:
    """
    Create median bit weights from analysed bit frequency.

    Parameters
    ----------
    bitfreq : dict
        Dictionary containing the bitfrequency dictionary from a the BitAnalyser
    algorithms : list
        Class names from the LabelEncoder used for algorithms
    metrics : list
        Class names from the LabelEncoder used for metrics
    transforms : list
        Class names from the LabelEncoder used for transformations

    Returns
    -------
    dict
        Dictionary with median weights for each pair (algorithm, metric)
    """
    # Dict to keep weights during loops
    bit_weights = {}
    a_s_list = le["a"].classes_
    m_s_list = le["m"].classes_
    t_s_list = le["t"].classes_

    # Outer loop
    for a in a_s_list:
        # Inner loop
        for m in m_s_list:
            pair = f"{a}_{m}"

            # Temp list to store weights per loop
            _w = []

            # Inner-inner loop to process each transform
            for t in t_s_list:
                # Skip bits from orig-orig
                if t == "orig":
                    continue
                else:
                    # Get the bits from the original triplet
                    freq = bitfreq[f"{a}_{t}_{m}"]
                    result = freq.T.median().values
                    _w.append(result)

            bit_weights[pair] = np.median(_w, axis=0)
    # return all the weights
    return bit_weights
