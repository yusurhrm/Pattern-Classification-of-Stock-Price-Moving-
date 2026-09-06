import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)



DATA_PATH = "data/processed/normalised_prices.csv"

RESULTS_DIR = "results/kmeans_higher_k"
FIGURES_DIR = "results/figures/kmeans_higher_k"

K_VALUES = [3, 5, 7, 10]

RANDOM_STATE = 42
N_INIT = 20

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    FIGURES_DIR,
    exist_ok=True
)


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
print("K-MEANS HIGHER-CLUSTER ANALYSIS")
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
    f"Companies: "
    f"{normalised_prices.shape[1]}"
)

print(
    f"Trading days: "
    f"{normalised_prices.shape[0]}"
)


if normalised_prices.empty:
    raise ValueError(
        "The normalised-price dataset is empty."
    )

if normalised_prices.isna().any().any():

    missing_companies = (
        normalised_prices.columns[
            normalised_prices.isna().any()
        ]
        .tolist()
    )

    raise ValueError(
        "Missing values found for: "
        + ", ".join(missing_companies)
    )

if not np.isfinite(
    normalised_prices.to_numpy()
).all():

    raise ValueError(
        "The dataset contains invalid or infinite values."
    )


X = normalised_prices.T

ticker_names = X.index.tolist()

print(
    f"\nClustering matrix shape: "
    f"{X.shape}"
)


pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X
)

explained_variance = (
    pca.explained_variance_ratio_ * 100
)

print("\nPCA explained variance:")

print(
    f"PC1: "
    f"{explained_variance[0]:.2f}%"
)

print(
    f"PC2: "
    f"{explained_variance[1]:.2f}%"
)

print(
    f"Combined: "
    f"{explained_variance.sum():.2f}%"
)


comparison_rows = []

all_assignments = []

cluster_size_rows = []


for k in K_VALUES:

    print("\n" + "=" * 80)
    print(
        f"K-MEANS ANALYSIS FOR k = {k}"
    )
    print("=" * 80)


    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=N_INIT
    )

    labels = model.fit_predict(
        X
    )

    labels_for_output = (
        labels + 1
    )


    assignments = pd.DataFrame({
        "Ticker": ticker_names,
        "Cluster": labels_for_output
    })

    assignments = (
        assignments
        .sort_values(
            [
                "Cluster",
                "Ticker"
            ]
        )
        .reset_index(drop=True)
    )

    assignments["k"] = k

    all_assignments.append(
        assignments.copy()
    )

    assignments.to_csv(
        os.path.join(
            RESULTS_DIR,
            f"kmeans_clusters_k{k}.csv"
        ),
        index=False
    )


    cluster_sizes = (
        assignments[
            "Cluster"
        ]
        .value_counts()
        .sort_index()
    )

    print("\nCluster sizes:")

    print(
        cluster_sizes
    )

    for (
        cluster_number,
        size
    ) in cluster_sizes.items():

        cluster_size_rows.append({
            "k": k,
            "Cluster": cluster_number,
            "Company Count": int(size)
        })



    for cluster_number in sorted(
        assignments[
            "Cluster"
        ].unique()
    ):

        members = (
            assignments.loc[
                assignments[
                    "Cluster"
                ]
                == cluster_number,
                "Ticker"
            ]
            .tolist()
        )

        print(
            f"\nCluster {cluster_number} "
            f"({len(members)} companies):"
        )

        print(
            ", ".join(members)
        )



    silhouette = silhouette_score(
        X,
        labels,
        metric="euclidean"
    )

    davies_bouldin = (
        davies_bouldin_score(
            X,
            labels
        )
    )

    calinski_harabasz = (
        calinski_harabasz_score(
            X,
            labels
        )
    )

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

    comparison_rows.append({
        "k": k,
        "WCSS": model.inertia_,
        "Silhouette Score": silhouette,
        "Davies-Bouldin Index": davies_bouldin,
        "Calinski-Harabasz Index":
            calinski_harabasz,
        "Smallest Cluster":
            smallest_cluster,
        "Largest Cluster":
            largest_cluster,
        "Singleton Clusters":
            singleton_clusters
    })

    print("\nValidation metrics:")

    print(
        f"Silhouette Score: "
        f"{silhouette:.4f}"
    )

    print(
        f"Davies-Bouldin Index: "
        f"{davies_bouldin:.4f}"
    )

    print(
        f"Calinski-Harabasz Index: "
        f"{calinski_harabasz:.2f}"
    )

    print(
        f"Singleton clusters: "
        f"{singleton_clusters}"
    )

    plot_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Cluster": labels_for_output,
        "Ticker": ticker_names
    })

    plt.figure(
        figsize=(11, 8)
    )

    for cluster_number in sorted(
        plot_df[
            "Cluster"
        ].unique()
    ):

        subset = (
            plot_df.loc[
                plot_df[
                    "Cluster"
                ]
                == cluster_number
            ]
        )

        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            s=80,
            alpha=0.8,
            label=(
                f"Cluster "
                f"{cluster_number} "
                f"(n={len(subset)})"
            )
        )

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

    plt.title(
        f"K-Means PCA Visualisation "
        f"(k = {k})"
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
            f"kmeans_pca_k{k}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


    cluster_profiles = pd.DataFrame(
        index=normalised_prices.index
    )

    for cluster_number in sorted(
        assignments[
            "Cluster"
        ].unique()
    ):

        tickers = (
            assignments.loc[
                assignments[
                    "Cluster"
                ]
                == cluster_number,
                "Ticker"
            ]
            .tolist()
        )

        cluster_profiles[
            f"Cluster {cluster_number}"
        ] = (
            normalised_prices[
                tickers
            ]
            .mean(
                axis=1
            )
        )

    cluster_profiles.to_csv(
        os.path.join(
            RESULTS_DIR,
            f"kmeans_cluster_means_k{k}.csv"
        )
    )


    plt.figure(
        figsize=(13, 7)
    )

    for cluster_number in sorted(
        assignments[
            "Cluster"
        ].unique()
    ):

        cluster_size = int(
            cluster_sizes.loc[
                cluster_number
            ]
        )

        column = (
            f"Cluster "
            f"{cluster_number}"
        )

        plt.plot(
            cluster_profiles.index,
            cluster_profiles[
                column
            ],
            linewidth=2.5,
            label=(
                f"{column} "
                f"(n={cluster_size})"
            )
        )

    plt.axhline(
        y=1.0,
        linestyle="--",
        linewidth=1,
        alpha=0.6
    )

    plt.title(
        "Average Normalised "
        "Price Movement "
        f"(k = {k})"
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
            f"kmeans_average_patterns_k{k}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    for cluster_number in sorted(
        assignments[
            "Cluster"
        ].unique()
    ):

        tickers = (
            assignments.loc[
                assignments[
                    "Cluster"
                ]
                == cluster_number,
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
                alpha=0.25
            )

        plt.plot(
            cluster_mean.index,
            cluster_mean,
            linewidth=3.5,
            label="Cluster mean"
        )

        plt.axhline(
            y=1.0,
            linestyle="--",
            linewidth=1,
            alpha=0.6
        )

        plt.title(
            f"K-Means k={k}, "
            f"Cluster {cluster_number} "
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
                    f"kmeans_k{k}_"
                    f"cluster_"
                    f"{cluster_number}_"
                    "trajectories.png"
                )
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()


    print(
        "\nFinal normalised values "
        "of cluster means:"
    )

    for column in (
        cluster_profiles.columns
    ):

        final_value = float(
            cluster_profiles[
                column
            ]
            .iloc[-1]
        )

        print(
            f"{column}: "
            f"{final_value:.4f}"
        )


combined_assignments = pd.concat(
    all_assignments,
    ignore_index=True
)

combined_assignments.to_csv(
    os.path.join(
        RESULTS_DIR,
        "kmeans_all_k_assignments.csv"
    ),
    index=False
)


cluster_sizes_df = pd.DataFrame(
    cluster_size_rows
)

cluster_sizes_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "kmeans_all_k_cluster_sizes.csv"
    ),
    index=False
)


comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "kmeans_k3_k5_k7_k10_comparison.csv"
    ),
    index=False
)

print("\n" + "=" * 100)
print(
    "COMPARISON OF k = 3, 5, 7 AND 10"
)
print("=" * 100)

print(
    comparison_df.to_string(
        index=False
    )
)


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    comparison_df["k"],
    comparison_df[
        "Silhouette Score"
    ],
    marker="o"
)

plt.xlabel(
    "Number of Clusters (k)"
)

plt.ylabel(
    "Silhouette Score"
)

plt.title(
    "Silhouette Score for "
    "Selected Cluster Numbers"
)

plt.xticks(
    K_VALUES
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "higher_k_silhouette_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    comparison_df["k"],
    comparison_df[
        "Largest Cluster"
    ],
    marker="o",
    label="Largest cluster"
)

plt.plot(
    comparison_df["k"],
    comparison_df[
        "Smallest Cluster"
    ],
    marker="o",
    label="Smallest cluster"
)

plt.xlabel(
    "Number of Clusters (k)"
)

plt.ylabel(
    "Number of Companies"
)

plt.title(
    "Cluster-Size Balance Across "
    "Selected Values of k"
)

plt.xticks(
    K_VALUES
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "higher_k_cluster_size_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    comparison_df["k"],
    comparison_df[
        "Singleton Clusters"
    ],
    marker="o"
)

plt.xlabel(
    "Number of Clusters (k)"
)

plt.ylabel(
    "Number of Singleton Clusters"
)

plt.title(
    "Cluster Fragmentation "
    "Across Values of k"
)

plt.xticks(
    K_VALUES
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "higher_k_singleton_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


print("\n" + "=" * 100)
print("HIGHER-k ANALYSIS COMPLETED")
print("=" * 100)

print(
    f"\nCompanies analysed: "
    f"{X.shape[0]}"
)

print(
    f"Trading days analysed: "
    f"{X.shape[1]}"
)

print(
    f"Values of k analysed: "
    f"{K_VALUES}"
)

print(
    "\nThe following were generated "
    "for each value of k:"
)

print(
    "- Cluster assignments"
)

print(
    "- Cluster sizes"
)

print(
    "- PCA visualisation"
)

print(
    "- Mean cluster trajectories"
)

print(
    "- Individual stock trajectories"
)

print(
    "- Internal validation metrics"
)

print(
    "\nResults directory:"
)

print(
    RESULTS_DIR
)

print(
    "\nFigures directory:"
)

print(
    FIGURES_DIR
)