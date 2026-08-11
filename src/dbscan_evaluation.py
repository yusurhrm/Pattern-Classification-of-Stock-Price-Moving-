import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/processed/normalised_prices.csv"

RESULTS_DIR = "results/dbscan_evaluation"
FIGURES_DIR = "results/figures/dbscan_evaluation"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# Load cleaned normalised data
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
print("DBSCAN CLUSTERING EVALUATION")
print("=" * 80)

print(f"\nDataset shape: {normalised_prices.shape}")
print(
    f"Date range: {normalised_prices.index.min()} "
    f"to {normalised_prices.index.max()}"
)
print(f"Number of companies: {normalised_prices.shape[1]}")
print(f"Trading days: {normalised_prices.shape[0]}")


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
        ].tolist()
    )

    raise ValueError(
        "Missing values were found for: "
        + ", ".join(companies_with_missing)
    )

if not np.isfinite(
    normalised_prices.to_numpy()
).all():
    raise ValueError(
        "The dataset contains infinite or invalid values."
    )

print("\nInput validation completed successfully.")


# ============================================================
# Prepare clustering matrix
# ============================================================

# Before transposing:
# rows = dates
# columns = companies
#
# After transposing:
# rows = companies
# columns = daily normalised prices

X_original = normalised_prices.T

ticker_names = X_original.index.tolist()

print("\nOriginal clustering matrix:")
print(X_original.shape)


# ============================================================
# PCA dimensionality reduction
# ============================================================

# Retain enough principal components to explain at least 95%
# of the variance.
#
# DBSCAN is sensitive to distances, and Euclidean distances
# become less informative in very high-dimensional spaces.
# PCA therefore reduces dimensionality before DBSCAN.

pca_full = PCA()

X_pca_full = pca_full.fit_transform(
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
    f"Variance explained: "
    f"{cumulative_variance[n_components_95 - 1] * 100:.2f}%"
)


# Fit PCA using selected number of components

pca = PCA(
    n_components=n_components_95
)

X = pca.fit_transform(
    X_original
)

print(
    f"\nReduced DBSCAN matrix shape: {X.shape}"
)


# Save PCA summary

pca_summary = pd.DataFrame({
    "Principal Component": np.arange(
        1,
        len(pca.explained_variance_ratio_) + 1
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
        "dbscan_pca_variance_summary.csv"
    ),
    index=False
)


# ============================================================
# Plot cumulative PCA variance
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    np.arange(
        1,
        len(cumulative_variance) + 1
    ),
    cumulative_variance * 100
)

plt.axhline(
    y=95,
    linestyle="--",
    label="95% variance"
)

plt.axvline(
    x=n_components_95,
    linestyle="--",
    label=(
        f"{n_components_95} components"
    )
)

plt.xlabel(
    "Number of Principal Components"
)

plt.ylabel(
    "Cumulative Explained Variance (%)"
)

plt.title(
    "PCA Cumulative Explained Variance"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "dbscan_pca_cumulative_variance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# k-distance plots
# ============================================================

# Several min_samples values are examined because DBSCAN
# results can be sensitive to this parameter.

MIN_SAMPLES_VALUES = [
    3,
    4,
    5,
    6,
    8,
    10
]

k_distance_results = {}


for min_samples in MIN_SAMPLES_VALUES:

    print(
        f"\nCalculating k-distance curve "
        f"for min_samples = {min_samples}"
    )

    nearest_neighbors = NearestNeighbors(
        n_neighbors=min_samples,
        metric="euclidean"
    )

    nearest_neighbors.fit(
        X
    )

    distances, _ = (
        nearest_neighbors.kneighbors(
            X
        )
    )

    # Distance to the kth nearest neighbour
    kth_distances = np.sort(
        distances[:, -1]
    )

    k_distance_results[
        min_samples
    ] = kth_distances


    # --------------------------------------------------------
    # Plot individual k-distance graph
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        np.arange(
            1,
            len(kth_distances) + 1
        ),
        kth_distances
    )

    plt.xlabel(
        "Companies Sorted by Distance"
    )

    plt.ylabel(
        f"Distance to "
        f"{min_samples}th Nearest Neighbour"
    )

    plt.title(
        f"DBSCAN k-Distance Plot "
        f"(min_samples = {min_samples})"
    )

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            (
                f"dbscan_k_distance_"
                f"min_samples_{min_samples}.png"
            )
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# ============================================================
# Save k-distance values
# ============================================================

k_distance_df = pd.DataFrame(
    {
        f"min_samples_{min_samples}":
            values
        for min_samples, values
        in k_distance_results.items()
    }
)

k_distance_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "dbscan_k_distance_values.csv"
    ),
    index=False
)


# ============================================================
# Automatically create candidate eps values
# ============================================================

# Rather than manually choosing only one elbow value,
# evaluate a broad range derived from the k-distance curves.

all_k_distances = np.concatenate(
    list(
        k_distance_results.values()
    )
)

lower_eps = float(
    np.percentile(
        all_k_distances,
        20
    )
)

upper_eps = float(
    np.percentile(
        all_k_distances,
        95
    )
)

EPS_VALUES = np.linspace(
    lower_eps,
    upper_eps,
    25
)

print("\n" + "=" * 80)
print("EPS SEARCH RANGE")
print("=" * 80)

print(
    f"Minimum eps tested: "
    f"{EPS_VALUES[0]:.4f}"
)

print(
    f"Maximum eps tested: "
    f"{EPS_VALUES[-1]:.4f}"
)

print(
    f"Number of eps values tested: "
    f"{len(EPS_VALUES)}"
)


# ============================================================
# Evaluate DBSCAN parameter combinations
# ============================================================

evaluation_rows = []

assignment_rows = []


for min_samples in MIN_SAMPLES_VALUES:

    for eps in EPS_VALUES:

        model = DBSCAN(
            eps=float(eps),
            min_samples=min_samples,
            metric="euclidean"
        )

        labels = model.fit_predict(
            X
        )

        unique_labels = set(
            labels
        )

        # Noise label = -1
        cluster_labels = [
            label
            for label in unique_labels
            if label != -1
        ]

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

        cluster_sizes = pd.Series(
            labels[
                labels != -1
            ]
        ).value_counts()

        if not cluster_sizes.empty:

            smallest_cluster = int(
                cluster_sizes.min()
            )

            largest_cluster = int(
                cluster_sizes.max()
            )

            singleton_clusters = int(
                (
                    cluster_sizes == 1
                ).sum()
            )

        else:

            smallest_cluster = 0
            largest_cluster = 0
            singleton_clusters = 0


        # ----------------------------------------------------
        # Silhouette Score
        # ----------------------------------------------------

        silhouette = np.nan

        non_noise_mask = (
            labels != -1
        )

        non_noise_labels = labels[
            non_noise_mask
        ]

        non_noise_X = X[
            non_noise_mask
        ]

        unique_non_noise = np.unique(
            non_noise_labels
        )

        # Silhouette requires at least 2 clusters
        # and fewer clusters than observations.
        if (
            len(unique_non_noise) >= 2
            and
            len(unique_non_noise)
            < len(non_noise_X)
        ):

            silhouette = silhouette_score(
                non_noise_X,
                non_noise_labels,
                metric="euclidean"
            )


        evaluation_rows.append({
            "eps": float(eps),
            "min_samples": min_samples,
            "Number of Clusters":
                number_of_clusters,
            "Noise Points":
                number_of_noise,
            "Noise Percentage":
                noise_percentage,
            "Silhouette Score":
                silhouette,
            "Smallest Cluster":
                smallest_cluster,
            "Largest Cluster":
                largest_cluster,
            "Singleton Clusters":
                singleton_clusters
        })


        # Save assignments for later inspection

        for ticker, label in zip(
            ticker_names,
            labels
        ):

            assignment_rows.append({
                "eps": float(eps),
                "min_samples":
                    min_samples,
                "Ticker":
                    ticker,
                "Cluster":
                    int(label)
            })


# ============================================================
# Convert evaluation results to DataFrame
# ============================================================

evaluation_df = pd.DataFrame(
    evaluation_rows
)

assignments_df = pd.DataFrame(
    assignment_rows
)

evaluation_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "dbscan_evaluation_metrics.csv"
    ),
    index=False
)

assignments_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "dbscan_all_assignments.csv"
    ),
    index=False
)


# ============================================================
# Keep useful candidate solutions
# ============================================================

# Require:
# - at least 2 clusters
# - no more than 10 clusters
# - at least 50% of observations assigned to clusters
# - a valid silhouette score

candidate_df = evaluation_df.loc[
    (
        evaluation_df[
            "Number of Clusters"
        ] >= 2
    )
    &
    (
        evaluation_df[
            "Number of Clusters"
        ] <= 10
    )
    &
    (
        evaluation_df[
            "Noise Percentage"
        ] <= 50
    )
    &
    (
        evaluation_df[
            "Silhouette Score"
        ].notna()
    )
].copy()


# ============================================================
# Rank candidate solutions
# ============================================================

if not candidate_df.empty:

    # Higher silhouette = better
    candidate_df[
        "Silhouette Rank"
    ] = (
        candidate_df[
            "Silhouette Score"
        ]
        .rank(
            ascending=False,
            method="min"
        )
    )

    # Lower noise percentage = better
    candidate_df[
        "Noise Rank"
    ] = (
        candidate_df[
            "Noise Percentage"
        ]
        .rank(
            ascending=True,
            method="min"
        )
    )

    # Fewer singleton clusters = better
    candidate_df[
        "Singleton Rank"
    ] = (
        candidate_df[
            "Singleton Clusters"
        ]
        .rank(
            ascending=True,
            method="min"
        )
    )

    candidate_df[
        "Mean Rank"
    ] = (
        candidate_df[
            [
                "Silhouette Rank",
                "Noise Rank",
                "Singleton Rank"
            ]
        ]
        .mean(
            axis=1
        )
    )

    candidate_df = (
        candidate_df
        .sort_values(
            by=[
                "Mean Rank",
                "Silhouette Score",
                "Noise Percentage"
            ],
            ascending=[
                True,
                False,
                True
            ]
        )
    )

    candidate_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "dbscan_ranked_candidates.csv"
        ),
        index=False
    )


# ============================================================
# Print full evaluation summary
# ============================================================

print("\n" + "=" * 100)
print("DBSCAN EVALUATION RESULTS")
print("=" * 100)

print(
    evaluation_df[
        [
            "eps",
            "min_samples",
            "Number of Clusters",
            "Noise Points",
            "Noise Percentage",
            "Silhouette Score",
            "Smallest Cluster",
            "Largest Cluster",
            "Singleton Clusters"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Print strongest candidate solutions
# ============================================================

print("\n" + "=" * 100)
print("TOP DBSCAN CANDIDATE SOLUTIONS")
print("=" * 100)

if candidate_df.empty:

    print(
        "No candidate DBSCAN solution met "
        "the current selection criteria."
    )

else:

    print(
        candidate_df[
            [
                "eps",
                "min_samples",
                "Number of Clusters",
                "Noise Points",
                "Noise Percentage",
                "Silhouette Score",
                "Smallest Cluster",
                "Largest Cluster",
                "Singleton Clusters",
                "Mean Rank"
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )


# ============================================================
# Plot silhouette score by eps
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for min_samples in MIN_SAMPLES_VALUES:

    subset = evaluation_df.loc[
        evaluation_df[
            "min_samples"
        ]
        == min_samples
    ]

    plt.plot(
        subset["eps"],
        subset[
            "Silhouette Score"
        ],
        marker="o",
        label=(
            f"min_samples="
            f"{min_samples}"
        )
    )

plt.xlabel(
    "eps"
)

plt.ylabel(
    "Silhouette Score"
)

plt.title(
    "DBSCAN Silhouette Score "
    "Across Parameter Values"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "dbscan_silhouette_by_eps.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# Plot number of clusters by eps
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for min_samples in MIN_SAMPLES_VALUES:

    subset = evaluation_df.loc[
        evaluation_df[
            "min_samples"
        ]
        == min_samples
    ]

    plt.plot(
        subset["eps"],
        subset[
            "Number of Clusters"
        ],
        marker="o",
        label=(
            f"min_samples="
            f"{min_samples}"
        )
    )

plt.xlabel(
    "eps"
)

plt.ylabel(
    "Number of Clusters"
)

plt.title(
    "DBSCAN Number of Clusters "
    "Across Parameter Values"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "dbscan_clusters_by_eps.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# Plot noise percentage by eps
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for min_samples in MIN_SAMPLES_VALUES:

    subset = evaluation_df.loc[
        evaluation_df[
            "min_samples"
        ]
        == min_samples
    ]

    plt.plot(
        subset["eps"],
        subset[
            "Noise Percentage"
        ],
        marker="o",
        label=(
            f"min_samples="
            f"{min_samples}"
        )
    )

plt.xlabel(
    "eps"
)

plt.ylabel(
    "Noise Points (%)"
)

plt.title(
    "DBSCAN Noise Percentage "
    "Across Parameter Values"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "dbscan_noise_percentage_by_eps.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 100)
print("DBSCAN EVALUATION COMPLETED")
print("=" * 100)

print(
    f"\nCompanies analysed: "
    f"{X_original.shape[0]}"
)

print(
    f"Original dimensions: "
    f"{X_original.shape[1]}"
)

print(
    f"PCA dimensions retained: "
    f"{X.shape[1]}"
)

print(
    f"Variance retained: "
    f"{pca.explained_variance_ratio_.sum() * 100:.2f}%"
)

print(
    f"min_samples values evaluated: "
    f"{MIN_SAMPLES_VALUES}"
)

print(
    f"eps values evaluated: "
    f"{len(EPS_VALUES)}"
)

print(
    "\nResults saved to:"
)

print(
    RESULTS_DIR
)

print(
    "\nFigures saved to:"
)

print(
    FIGURES_DIR
)