import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/processed/normalised_prices.csv"

RESULTS_DIR = "results/dbscan_final"
FIGURES_DIR = "results/figures/dbscan_final"

FINAL_EPS = 4.037274
FINAL_MIN_SAMPLES = 3

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    FIGURES_DIR,
    exist_ok=True
)


# ============================================================
# Load cleaned normalised prices
# ============================================================

normalised_prices = pd.read_csv(
    DATA_PATH,
    index_col=0,
    parse_dates=True
)

normalised_prices = (
    normalised_prices
    .sort_index()
    .sort_index(axis=1)
)

print("=" * 80)
print("FINAL DBSCAN CLUSTERING")
print("=" * 80)

print(
    f"\nDataset shape: "
    f"{normalised_prices.shape}"
)

print(
    f"Date range: "
    f"{normalised_prices.index.min()} "
    f"to {normalised_prices.index.max()}"
)

print(
    f"Number of companies: "
    f"{normalised_prices.shape[1]}"
)

print(
    f"Number of trading days: "
    f"{normalised_prices.shape[0]}"
)


# ============================================================
# Validate dataset
# ============================================================

if normalised_prices.empty:
    raise ValueError(
        "The normalised-price dataset is empty."
    )

if normalised_prices.isna().any().any():

    companies_with_missing = (
        normalised_prices.columns[
            normalised_prices.isna().any()
        ]
        .tolist()
    )

    raise ValueError(
        "Missing values found for: "
        + ", ".join(companies_with_missing)
    )

if not np.isfinite(
    normalised_prices.to_numpy()
).all():

    raise ValueError(
        "The dataset contains "
        "infinite or invalid values."
    )

print(
    "\nInput validation completed successfully."
)


# ============================================================
# Prepare clustering matrix
# ============================================================

# Before transposing:
# rows = trading days
# columns = companies
#
# After transposing:
# rows = companies
# columns = normalised prices through time

X_original = normalised_prices.T

ticker_names = (
    X_original.index.tolist()
)

print("\nOriginal clustering matrix:")
print(X_original.shape)


# ============================================================
# PCA dimensionality reduction
# ============================================================

# Use exactly the same approach as the DBSCAN evaluation:
# retain enough principal components to explain at least 95%
# of the total variance.

pca_full = PCA()

pca_full.fit(
    X_original
)

cumulative_variance = np.cumsum(
    pca_full.explained_variance_ratio_
)

n_components_95 = int(
    np.argmax(
        cumulative_variance >= 0.95
    ) + 1
)

print("\n" + "=" * 80)
print("PCA DIMENSIONALITY REDUCTION")
print("=" * 80)

print(
    f"Principal components required "
    f"for at least 95% variance: "
    f"{n_components_95}"
)

print(
    f"Variance retained: "
    f"{cumulative_variance[n_components_95 - 1] * 100:.2f}%"
)


# Fit PCA using selected number of components

pca = PCA(
    n_components=n_components_95
)

X_pca = pca.fit_transform(
    X_original
)

print(
    f"Reduced clustering matrix: "
    f"{X_pca.shape}"
)


# Save PCA variance information

pca_summary = pd.DataFrame({
    "Principal Component":
        np.arange(
            1,
            n_components_95 + 1
        ),
    "Explained Variance Ratio":
        pca.explained_variance_ratio_,
    "Cumulative Explained Variance":
        np.cumsum(
            pca.explained_variance_ratio_
        )
})

pca_summary.to_csv(
    os.path.join(
        RESULTS_DIR,
        "dbscan_final_pca_variance.csv"
    ),
    index=False
)


# ============================================================
# Run final DBSCAN model
# ============================================================

print("\n" + "=" * 80)
print("FINAL DBSCAN PARAMETERS")
print("=" * 80)

print(
    f"eps = {FINAL_EPS}"
)

print(
    f"min_samples = "
    f"{FINAL_MIN_SAMPLES}"
)


dbscan = DBSCAN(
    eps=FINAL_EPS,
    min_samples=FINAL_MIN_SAMPLES,
    metric="euclidean"
)

labels = dbscan.fit_predict(
    X_pca
)


# ============================================================
# Convert labels for easier interpretation
# ============================================================

# DBSCAN uses:
# -1 = noise
#  0,1,... = clusters
#
# For output, clusters will be displayed as 1,2,...
# while noise remains labelled as "Noise".

display_labels = []

for label in labels:

    if label == -1:
        display_labels.append(
            "Noise"
        )
    else:
        display_labels.append(
            f"Cluster {label + 1}"
        )


# ============================================================
# Cluster assignments
# ============================================================

assignments = pd.DataFrame({
    "Ticker": ticker_names,
    "DBSCAN Label": labels,
    "Cluster": display_labels
})

assignments = (
    assignments
    .sort_values(
        by=[
            "DBSCAN Label",
            "Ticker"
        ]
    )
    .reset_index(drop=True)
)

assignments.to_csv(
    os.path.join(
        RESULTS_DIR,
        "dbscan_final_assignments.csv"
    ),
    index=False
)


# ============================================================
# Cluster summary
# ============================================================

cluster_labels = sorted(
    [
        label
        for label in np.unique(labels)
        if label != -1
    ]
)

number_of_clusters = len(
    cluster_labels
)

number_of_noise = int(
    np.sum(
        labels == -1
    )
)

noise_percentage = (
    number_of_noise
    / len(labels)
    * 100
)


cluster_summary_rows = []

for label in cluster_labels:

    count = int(
        np.sum(
            labels == label
        )
    )

    cluster_summary_rows.append({
        "Cluster":
            f"Cluster {label + 1}",
        "Company Count":
            count
    })

cluster_summary_rows.append({
    "Cluster": "Noise",
    "Company Count":
        number_of_noise
})

cluster_summary = pd.DataFrame(
    cluster_summary_rows
)

cluster_summary.to_csv(
    os.path.join(
        RESULTS_DIR,
        "dbscan_final_cluster_sizes.csv"
    ),
    index=False
)


# ============================================================
# Silhouette Score
# ============================================================

# Calculate silhouette using only non-noise observations,
# consistent with the evaluation stage.

non_noise_mask = (
    labels != -1
)

non_noise_X = X_pca[
    non_noise_mask
]

non_noise_labels = labels[
    non_noise_mask
]

silhouette = np.nan

if (
    len(
        np.unique(
            non_noise_labels
        )
    ) >= 2
):

    silhouette = silhouette_score(
        non_noise_X,
        non_noise_labels,
        metric="euclidean"
    )


# ============================================================
# Print final results
# ============================================================

print("\n" + "=" * 80)
print("FINAL DBSCAN RESULTS")
print("=" * 80)

print(
    f"Number of clusters: "
    f"{number_of_clusters}"
)

print(
    f"Noise points: "
    f"{number_of_noise}"
)

print(
    f"Noise percentage: "
    f"{noise_percentage:.2f}%"
)

print(
    f"Silhouette Score "
    f"(excluding noise): "
    f"{silhouette:.4f}"
)


print("\nCluster sizes:")
print(
    cluster_summary.to_string(
        index=False
    )
)


# ============================================================
# Print cluster memberships
# ============================================================

for label in cluster_labels:

    members = (
        assignments.loc[
            assignments[
                "DBSCAN Label"
            ]
            == label,
            "Ticker"
        ]
        .tolist()
    )

    print(
        "\n" + "-" * 80
    )

    print(
        f"Cluster {label + 1} "
        f"({len(members)} companies)"
    )

    print(
        "-" * 80
    )

    print(
        ", ".join(members)
    )


noise_members = (
    assignments.loc[
        assignments[
            "DBSCAN Label"
        ]
        == -1,
        "Ticker"
    ]
    .tolist()
)

print(
    "\n" + "-" * 80
)

print(
    f"Noise "
    f"({len(noise_members)} companies)"
)

print(
    "-" * 80
)

print(
    ", ".join(noise_members)
)


# ============================================================
# PCA coordinates for visualisation
# ============================================================

# Use the first two principal components for plotting.
#
# Even though clustering uses all retained PCA components,
# only PC1 and PC2 are used for visualisation.

plot_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Ticker": ticker_names,
    "DBSCAN Label": labels
})

plot_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "dbscan_final_pca_coordinates.csv"
    ),
    index=False
)


# ============================================================
# PCA cluster visualisation
# ============================================================

plt.figure(
    figsize=(11, 8)
)


# Plot regular DBSCAN clusters

for label in cluster_labels:

    subset = plot_df.loc[
        plot_df[
            "DBSCAN Label"
        ]
        == label
    ]

    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        s=80,
        alpha=0.8,
        label=(
            f"Cluster {label + 1} "
            f"(n={len(subset)})"
        )
    )


# Plot noise separately

noise_subset = plot_df.loc[
    plot_df[
        "DBSCAN Label"
    ]
    == -1
]

plt.scatter(
    noise_subset["PC1"],
    noise_subset["PC2"],
    s=90,
    marker="x",
    label=(
        f"Noise "
        f"(n={len(noise_subset)})"
    )
)


# Label each company

for _, row in plot_df.iterrows():

    plt.annotate(
        row["Ticker"],
        (
            row["PC1"],
            row["PC2"]
        ),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=6,
        alpha=0.7
    )


explained_variance = (
    pca.explained_variance_ratio_
    * 100
)

plt.title(
    "DBSCAN Clustering PCA Visualisation"
)

plt.xlabel(
    "Principal Component 1 "
    f"({explained_variance[0]:.2f}%)"
)

plt.ylabel(
    "Principal Component 2 "
    f"({explained_variance[1]:.2f}%)"
)

plt.legend()

plt.grid(
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "dbscan_final_pca.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# Mean cluster trajectories
# ============================================================

cluster_profiles = pd.DataFrame(
    index=normalised_prices.index
)


for label in cluster_labels:

    tickers = (
        assignments.loc[
            assignments[
                "DBSCAN Label"
            ]
            == label,
            "Ticker"
        ]
        .tolist()
    )

    cluster_profiles[
        f"Cluster {label + 1}"
    ] = (
        normalised_prices[
            tickers
        ]
        .mean(
            axis=1
        )
    )


# Also calculate the average trajectory of noise companies

if len(noise_members) > 0:

    cluster_profiles[
        "Noise Mean"
    ] = (
        normalised_prices[
            noise_members
        ]
        .mean(
            axis=1
        )
    )


cluster_profiles.to_csv(
    os.path.join(
        RESULTS_DIR,
        "dbscan_final_mean_trajectories.csv"
    )
)


# ============================================================
# Plot mean trajectories
# ============================================================

plt.figure(
    figsize=(13, 7)
)


for column in cluster_profiles.columns:

    plt.plot(
        cluster_profiles.index,
        cluster_profiles[
            column
        ],
        linewidth=2.5,
        label=column
    )


plt.axhline(
    y=1.0,
    linestyle="--",
    linewidth=1,
    alpha=0.6
)

plt.title(
    "Average Normalised Price Movement "
    "by DBSCAN Cluster"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Mean Normalised Price"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "dbscan_final_mean_trajectories.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# Individual trajectories for each cluster
# ============================================================

for label in cluster_labels:

    tickers = (
        assignments.loc[
            assignments[
                "DBSCAN Label"
            ]
            == label,
            "Ticker"
        ]
        .tolist()
    )

    cluster_mean = (
        normalised_prices[
            tickers
        ]
        .mean(
            axis=1
        )
    )

    plt.figure(
        figsize=(13, 7)
    )

    for ticker in tickers:

        plt.plot(
            normalised_prices.index,
            normalised_prices[
                ticker
            ],
            linewidth=0.8,
            alpha=0.3
        )

    plt.plot(
        cluster_mean.index,
        cluster_mean,
        linewidth=3,
        label="Cluster mean"
    )

    plt.axhline(
        y=1.0,
        linestyle="--",
        linewidth=1,
        alpha=0.6
    )

    plt.title(
        f"DBSCAN Cluster "
        f"{label + 1} "
        f"(n={len(tickers)})"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Normalised Price"
    )

    plt.legend()

    plt.grid(
        alpha=0.2
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            (
                f"dbscan_cluster_"
                f"{label + 1}_"
                "trajectories.png"
            )
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# ============================================================
# Noise trajectory plot
# ============================================================

if len(noise_members) > 0:

    noise_mean = (
        normalised_prices[
            noise_members
        ]
        .mean(
            axis=1
        )
    )

    plt.figure(
        figsize=(13, 7)
    )

    for ticker in noise_members:

        plt.plot(
            normalised_prices.index,
            normalised_prices[
                ticker
            ],
            linewidth=0.8,
            alpha=0.3
        )

    plt.plot(
        noise_mean.index,
        noise_mean,
        linewidth=3,
        label="Noise mean"
    )

    plt.axhline(
        y=1.0,
        linestyle="--",
        linewidth=1,
        alpha=0.6
    )

    plt.title(
        f"DBSCAN Noise Trajectories "
        f"(n={len(noise_members)})"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Normalised Price"
    )

    plt.legend()

    plt.grid(
        alpha=0.2
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            "dbscan_noise_trajectories.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# ============================================================
# Final normalised values
# ============================================================

print("\n" + "=" * 80)
print("FINAL MEAN NORMALISED VALUES")
print("=" * 80)

for column in cluster_profiles.columns:

    final_value = (
        cluster_profiles[
            column
        ]
        .iloc[-1]
    )

    print(
        f"{column}: "
        f"{final_value:.4f}"
    )


# ============================================================
# Save final metrics
# ============================================================

metrics = pd.DataFrame({
    "eps": [
        FINAL_EPS
    ],
    "min_samples": [
        FINAL_MIN_SAMPLES
    ],
    "Number of Clusters": [
        number_of_clusters
    ],
    "Noise Points": [
        number_of_noise
    ],
    "Noise Percentage": [
        noise_percentage
    ],
    "Silhouette Score": [
        silhouette
    ],
    "PCA Components": [
        n_components_95
    ],
    "PCA Variance Retained (%)": [
        (
            pca
            .explained_variance_ratio_
            .sum()
            * 100
        )
    ]
})

metrics.to_csv(
    os.path.join(
        RESULTS_DIR,
        "dbscan_final_metrics.csv"
    ),
    index=False
)


# ============================================================
# Finished
# ============================================================

print("\n" + "=" * 80)
print("FINAL DBSCAN ANALYSIS COMPLETED")
print("=" * 80)

print(
    "\nSaved results to:"
)

print(
    RESULTS_DIR
)

print(
    "\nSaved figures to:"
)

print(
    FIGURES_DIR
)