from demo00_conf import *
import matplotlib.pyplot as plt

plt.switch_backend("agg")

from phaser.evaluation import ComputeMetrics, make_bit_weights
from phaser.similarities import IntraDistance, InterDistance, find_inter_samplesize
from phaser.plotting import bit_weights_ax

print("Running script.")
# Load the label encoders
le = load("./demo_outputs/LabelEncoders.bz2")
df_h = load("./demo_outputs/Hashes.df.bz2")
df_d = load("./demo_outputs/Distances.df.bz2")
n_samples = find_inter_samplesize(len(df_h["filename"].unique() * 1))

# Generate triplet combinations without 'orig'
triplets = np.array(
    np.meshgrid(
        le["a"].classes_, [t for t in le["t"].classes_ if t != "orig"], le["m"].classes_
    )
).T.reshape(-1, 3)

# Compute metrics for all available triplets
print(f"Number of triplets to analyse: {len(triplets)}")
cm = ComputeMetrics(le, df_d, df_h, analyse_bits=True, n_jobs=1)
m, b = cm.fit(triplets=triplets)

print(f"Performance without bit weights:")
print(m.groupby(["Algorithm"])[["AUC", "EER"]].agg(["mean", "std"]))
print(m)


# Plot the bit frequency for each triplet ignoring 'orig'
print(f"Plotting bit weights for each triplets")
for triplet in list(b.keys()):
    fig, ax = plt.subplots(1, 1, figsize=(5, 1.5), constrained_layout=True)
    _ = bit_weights_ax(b[triplet], ax=ax)
    fig.savefig(f"./demo_outputs/figs/03-bit_analysis_{triplet}.png")
    plt.close()


# Create bit_weights (algo,metric)
weights = make_bit_weights(b, le)

# Plot the applied bitweights for the pairs (algo,metric)
for pair in list(weights.keys()):
    fig, ax = plt.subplots(1, 1, figsize=(5, 1.5), constrained_layout=True)
    _ = bit_weights_ax(weights[pair].reshape(-1, 1), ax=ax)
    fig.savefig(f"./demo_outputs/figs/03-bit_weights_{pair}.png")
    plt.close()

intra_df_w = IntraDistance(METR_dict, le, 1, weights, progress_bar=True).fit(df_h)
inter_df_w = InterDistance(METR_dict, le, 0, weights, n_samples, progress_bar=True).fit(
    df_h
)
df_d_w = pd.concat([intra_df_w, inter_df_w])

cm_w = ComputeMetrics(le, df_d_w, df_h, analyse_bits=False, n_jobs=1)
m_w, _ = cm_w.fit(triplets=triplets)
print(f"Performance with bit weights:")
print(m_w.groupby(["Algorithm"])[["AUC", "EER"]].agg(["mean", "std"]))
print(m_w)

# Plot the AUC comparison between without and with bit weights
from phaser.plotting import auc_cmp_fig

fig = auc_cmp_fig(m, m_w, metric="Hamming")
fig.savefig("./demo_outputs/figs/03_auc_cmp_w_without_weights.png")
plt.close()
print("Script finished")
