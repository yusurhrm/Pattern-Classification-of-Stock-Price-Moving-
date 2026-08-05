import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


# ============================================================
# File paths and settings
# ============================================================

# Cleaned and normalised dataset containing the 98 companies
# retained from the FTSE 100 dataset.
DATA_PATH = "data/processed/normalised_prices.csv"

RESULTS_DIR = "results/hierarchical_evaluation"
FIGURES_DIR = "results/figures/hierarchical_evaluation"

MIN_CLUSTERS = 2
MAX_CLUSTERS = 10

LINKAGE_METHODS = [
    "ward",
    "complete",
    "average",
    "single",
]

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# Load cleaned normalised stock-price data
# ============================================================

normalised_prices = pd.read_csv(
    DATA_PATH,
    index_col=0,
    parse_dates=True,
)

normalised_prices = normalised_prices.sort_index()
normalised_prices = normalised_prices.sort_index(axis=1)

print("=" * 80)
print("HIERARCHICAL CLUSTERING EVALUATION")
print("=" * 80)

print(f"\nDataset shape: {normalised_prices.shape}")

print(
    f"Date range: {normalised_prices.index.min()} "
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
# Validate input data
# ============================================================

if normalised_prices.empty:
    raise ValueError(
        "The normalised-price dataset is empty."
    )

missing_values = int(
    normalised_prices.isna().sum().sum()
)

print(f"\nTotal missing values: {missing_values}")

if missing_values > 0:

    companies_with_missing = (
        normalised_prices.columns[
            normalised_prices.isna().any()
        ]
        .tolist()
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

constant_companies = (
    normalised_prices.columns[
        normalised_prices.nunique() <= 1
    ]
    .tolist()
)

if constant_companies:

    raise ValueError(
        "Constant price series were found for: "
        + ", ".join(constant_companies)
    )

print("Input validation completed successfully.")


# ============================================================
# Prepare data for clustering
# ============================================================

# Before transposing:
# rows = trading dates
# columns = companies
#
# After transposing:
# rows = companies
# columns = normalised prices across time

X = normalised_prices.T

ticker_names = X.index.tolist()

print("\nClustering matrix shape:")
print(X.shape)

print(
    f"\nEach of the {X.shape[0]} companies is represented by "
    f"{X.shape[1]} trading-day observations."
)


# ============================================================
# Select valid values of k
# ============================================================

maximum_k = min(
    MAX_CLUSTERS,
    X.shape[0] - 1,
)

if maximum_k < MIN_CLUSTERS:
    raise ValueError(
        "There are not enough companies to evaluate clustering."
    )

k_values = list(
    range(
        MIN_CLUSTERS,
        maximum_k + 1,
    )
)

print("\nValues of k to evaluate:")
print(k_values)


# ============================================================
# Evaluate hierarchical clustering
# ============================================================

evaluation_results = []
cluster_size_results = []
cluster_assignment_results = []
linkage_matrices = {}

for linkage_method in LINKAGE_METHODS:

    print("\n" + "=" * 80)
    print(
        f"EVALUATING LINKAGE METHOD: "
        f"{linkage_method.upper()}"
    )
    print("=" * 80)

    # All linkage methods use Euclidean distance so that the
    # comparison remains consistent with the K-Means analysis.
    #
    # Ward linkage specifically requires Euclidean distance.
    linkage_matrix = linkage(
        X.to_numpy(),
        method=linkage_method,
        metric="euclidean",
    )

    linkage_matrices[
        linkage_method
    ] = linkage_matrix

    for k in k_values:

        labels = fcluster(
            linkage_matrix,
            t=k,
            criterion="maxclust",
        )

        unique_labels = np.unique(
            labels
        )

        actual_clusters = len(
            unique_labels
        )

        if actual_clusters < 2:

            print(
                f"k={k}: skipped because only "
                f"{actual_clusters} cluster was produced."
            )

            continue

        if actual_clusters >= len(X):

            print(
                f"k={k}: skipped because each company "
                "would form its own cluster."
            )

            continue

        cluster_sizes = (
            pd.Series(
                labels
            )
            .value_counts()
            .sort_index()
        )

        silhouette = silhouette_score(
            X.to_numpy(),
            labels,
            metric="euclidean",
        )

        davies_bouldin = davies_bouldin_score(
            X.to_numpy(),
            labels,
        )

        calinski_harabasz = calinski_harabasz_score(
            X.to_numpy(),
            labels,
        )

        smallest_cluster = int(
            cluster_sizes.min()
        )

        largest_cluster = int(
            cluster_sizes.max()
        )

        singleton_clusters = int(
            (cluster_sizes == 1).sum()
        )

        evaluation_results.append(
            {
                "Linkage": linkage_method,
                "Requested k": k,
                "Actual Clusters": actual_clusters,
                "Silhouette Score": silhouette,
                "Davies-Bouldin Index": davies_bouldin,
                "Calinski-Harabasz Index": calinski_harabasz,
                "Smallest Cluster": smallest_cluster,
                "Largest Cluster": largest_cluster,
                "Singleton Clusters": singleton_clusters,
            }
        )

        for (
            cluster_number,
            cluster_size,
        ) in cluster_sizes.items():

            cluster_size_results.append(
                {
                    "Linkage": linkage_method,
                    "k": k,
                    "Cluster": int(
                        cluster_number
                    ),
                    "Cluster Size": int(
                        cluster_size
                    ),
                }
            )

        for (
            ticker,
            cluster_number,
        ) in zip(
            ticker_names,
            labels,
        ):

            cluster_assignment_results.append(
                {
                    "Linkage": linkage_method,
                    "k": k,
                    "Ticker": ticker,
                    "Cluster": int(
                        cluster_number
                    ),
                }
            )

        print(
            f"k={k:<2} | "
            f"Actual={actual_clusters:<2} | "
            f"Silhouette={silhouette:.4f} | "
            f"DB={davies_bouldin:.4f} | "
            f"CH={calinski_harabasz:.2f} | "
            f"Smallest={smallest_cluster:<2} | "
            f"Largest={largest_cluster:<2} | "
            f"Singletons={singleton_clusters}"
        )


# ============================================================
# Convert results to DataFrames
# ============================================================

evaluation_df = pd.DataFrame(
    evaluation_results
)

cluster_sizes_df = pd.DataFrame(
    cluster_size_results
)

cluster_assignments_df = pd.DataFrame(
    cluster_assignment_results
)

if evaluation_df.empty:
    raise ValueError(
        "No valid hierarchical clustering results were produced."
    )


# ============================================================
# Save full results
# ============================================================

evaluation_path = os.path.join(
    RESULTS_DIR,
    "hierarchical_evaluation_metrics.csv",
)

cluster_sizes_path = os.path.join(
    RESULTS_DIR,
    "hierarchical_cluster_sizes.csv",
)

cluster_assignments_path = os.path.join(
    RESULTS_DIR,
    "hierarchical_all_assignments.csv",
)

evaluation_df.to_csv(
    evaluation_path,
    index=False,
)

cluster_sizes_df.to_csv(
    cluster_sizes_path,
    index=False,
)

cluster_assignments_df.to_csv(
    cluster_assignments_path,
    index=False,
)


# ============================================================
# Save linkage matrices
# ============================================================

for (
    linkage_method,
    linkage_matrix,
) in linkage_matrices.items():

    linkage_table = pd.DataFrame(
        linkage_matrix,
        columns=[
            "Cluster 1",
            "Cluster 2",
            "Linkage Distance",
            "Cluster Size",
        ],
    )

    linkage_table.to_csv(
        os.path.join(
            RESULTS_DIR,
            f"{linkage_method}_linkage_matrix.csv",
        ),
        index=False,
    )


# ============================================================
# Identify best result for each linkage method
# ============================================================

best_by_silhouette = (
    evaluation_df
    .sort_values(
        by=[
            "Linkage",
            "Silhouette Score",
            "Singleton Clusters",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )
    .groupby(
        "Linkage",
        as_index=False,
    )
    .first()
)

best_by_davies_bouldin = (
    evaluation_df
    .sort_values(
        by=[
            "Linkage",
            "Davies-Bouldin Index",
            "Singleton Clusters",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    )
    .groupby(
        "Linkage",
        as_index=False,
    )
    .first()
)

best_by_calinski_harabasz = (
    evaluation_df
    .sort_values(
        by=[
            "Linkage",
            "Calinski-Harabasz Index",
            "Singleton Clusters",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )
    .groupby(
        "Linkage",
        as_index=False,
    )
    .first()
)

best_by_silhouette.to_csv(
    os.path.join(
        RESULTS_DIR,
        "best_hierarchical_by_silhouette.csv",
    ),
    index=False,
)

best_by_davies_bouldin.to_csv(
    os.path.join(
        RESULTS_DIR,
        "best_hierarchical_by_davies_bouldin.csv",
    ),
    index=False,
)

best_by_calinski_harabasz.to_csv(
    os.path.join(
        RESULTS_DIR,
        "best_hierarchical_by_calinski_harabasz.csv",
    ),
    index=False,
)


# ============================================================
# Identify overall metric optima
# ============================================================

overall_best_silhouette = evaluation_df.loc[
    evaluation_df[
        "Silhouette Score"
    ].idxmax()
]

overall_best_davies_bouldin = evaluation_df.loc[
    evaluation_df[
        "Davies-Bouldin Index"
    ].idxmin()
]

overall_best_calinski_harabasz = evaluation_df.loc[
    evaluation_df[
        "Calinski-Harabasz Index"
    ].idxmax()
]

overall_metric_summary = pd.DataFrame(
    [
        {
            "Metric": "Silhouette Score",
            "Best Linkage": overall_best_silhouette[
                "Linkage"
            ],
            "Best k": int(
                overall_best_silhouette[
                    "Requested k"
                ]
            ),
            "Metric Value": overall_best_silhouette[
                "Silhouette Score"
            ],
            "Singleton Clusters": int(
                overall_best_silhouette[
                    "Singleton Clusters"
                ]
            ),
        },
        {
            "Metric": "Davies-Bouldin Index",
            "Best Linkage": overall_best_davies_bouldin[
                "Linkage"
            ],
            "Best k": int(
                overall_best_davies_bouldin[
                    "Requested k"
                ]
            ),
            "Metric Value": overall_best_davies_bouldin[
                "Davies-Bouldin Index"
            ],
            "Singleton Clusters": int(
                overall_best_davies_bouldin[
                    "Singleton Clusters"
                ]
            ),
        },
        {
            "Metric": "Calinski-Harabasz Index",
            "Best Linkage": overall_best_calinski_harabasz[
                "Linkage"
            ],
            "Best k": int(
                overall_best_calinski_harabasz[
                    "Requested k"
                ]
            ),
            "Metric Value": overall_best_calinski_harabasz[
                "Calinski-Harabasz Index"
            ],
            "Singleton Clusters": int(
                overall_best_calinski_harabasz[
                    "Singleton Clusters"
                ]
            ),
        },
    ]
)

overall_metric_summary.to_csv(
    os.path.join(
        RESULTS_DIR,
        "hierarchical_overall_metric_optima.csv",
    ),
    index=False,
)


# ============================================================
# Calculate overall rankings
# ============================================================

# Higher silhouette score is better.
evaluation_df["Silhouette Rank"] = (
    evaluation_df[
        "Silhouette Score"
    ]
    .rank(
        ascending=False,
        method="min",
    )
)

# Lower Davies-Bouldin index is better.
evaluation_df["Davies-Bouldin Rank"] = (
    evaluation_df[
        "Davies-Bouldin Index"
    ]
    .rank(
        ascending=True,
        method="min",
    )
)

# Higher Calinski-Harabasz index is better.
evaluation_df["Calinski-Harabasz Rank"] = (
    evaluation_df[
        "Calinski-Harabasz Index"
    ]
    .rank(
        ascending=False,
        method="min",
    )
)

evaluation_df["Mean Metric Rank"] = (
    evaluation_df[
        [
            "Silhouette Rank",
            "Davies-Bouldin Rank",
            "Calinski-Harabasz Rank",
        ]
    ]
    .mean(axis=1)
)

ranked_results = evaluation_df.sort_values(
    by=[
        "Mean Metric Rank",
        "Singleton Clusters",
        "Smallest Cluster",
        "Silhouette Score",
    ],
    ascending=[
        True,
        True,
        False,
        False,
    ],
)

ranked_results_path = os.path.join(
    RESULTS_DIR,
    "hierarchical_ranked_results.csv",
)

ranked_results.to_csv(
    ranked_results_path,
    index=False,
)

# Save the evaluation file again with the ranking columns.
evaluation_df.to_csv(
    evaluation_path,
    index=False,
)


# ============================================================
# Create cluster-balance score
# ============================================================

evaluation_df["Cluster Size Ratio"] = (
    evaluation_df[
        "Largest Cluster"
    ]
    / evaluation_df[
        "Smallest Cluster"
    ]
)

cluster_balance_df = evaluation_df[
    [
        "Linkage",
        "Requested k",
        "Smallest Cluster",
        "Largest Cluster",
        "Cluster Size Ratio",
        "Singleton Clusters",
    ]
].copy()

cluster_balance_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "hierarchical_cluster_balance.csv",
    ),
    index=False,
)


# ============================================================
# Plot 1: Silhouette scores
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for linkage_method in LINKAGE_METHODS:

    method_results = (
        evaluation_df.loc[
            evaluation_df["Linkage"]
            == linkage_method
        ]
        .sort_values(
            "Requested k"
        )
    )

    plt.plot(
        method_results["Requested k"],
        method_results["Silhouette Score"],
        marker="o",
        label=linkage_method.capitalize(),
    )

plt.xlabel(
    "Number of Clusters (k)"
)

plt.ylabel(
    "Silhouette Score"
)

plt.title(
    f"Hierarchical Clustering Silhouette Scores "
    f"({X.shape[0]} Companies)"
)

plt.xticks(
    k_values
)

plt.grid(
    alpha=0.3
)

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "hierarchical_silhouette_scores.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()


# ============================================================
# Plot 2: Davies-Bouldin indices
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for linkage_method in LINKAGE_METHODS:

    method_results = (
        evaluation_df.loc[
            evaluation_df["Linkage"]
            == linkage_method
        ]
        .sort_values(
            "Requested k"
        )
    )

    plt.plot(
        method_results["Requested k"],
        method_results["Davies-Bouldin Index"],
        marker="o",
        label=linkage_method.capitalize(),
    )

plt.xlabel(
    "Number of Clusters (k)"
)

plt.ylabel(
    "Davies-Bouldin Index"
)

plt.title(
    f"Hierarchical Clustering Davies-Bouldin Indices "
    f"({X.shape[0]} Companies)"
)

plt.xticks(
    k_values
)

plt.grid(
    alpha=0.3
)

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "hierarchical_davies_bouldin_indices.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()


# ============================================================
# Plot 3: Calinski-Harabasz indices
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for linkage_method in LINKAGE_METHODS:

    method_results = (
        evaluation_df.loc[
            evaluation_df["Linkage"]
            == linkage_method
        ]
        .sort_values(
            "Requested k"
        )
    )

    plt.plot(
        method_results["Requested k"],
        method_results["Calinski-Harabasz Index"],
        marker="o",
        label=linkage_method.capitalize(),
    )

plt.xlabel(
    "Number of Clusters (k)"
)

plt.ylabel(
    "Calinski-Harabasz Index"
)

plt.title(
    f"Hierarchical Clustering Calinski-Harabasz Indices "
    f"({X.shape[0]} Companies)"
)

plt.xticks(
    k_values
)

plt.grid(
    alpha=0.3
)

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "hierarchical_calinski_harabasz_indices.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()


# ============================================================
# Plot 4: Number of singleton clusters
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for linkage_method in LINKAGE_METHODS:

    method_results = (
        evaluation_df.loc[
            evaluation_df["Linkage"]
            == linkage_method
        ]
        .sort_values(
            "Requested k"
        )
    )

    plt.plot(
        method_results["Requested k"],
        method_results["Singleton Clusters"],
        marker="o",
        label=linkage_method.capitalize(),
    )

plt.xlabel(
    "Number of Clusters (k)"
)

plt.ylabel(
    "Number of Singleton Clusters"
)

plt.title(
    f"Singleton Clusters by Linkage Method "
    f"({X.shape[0]} Companies)"
)

plt.xticks(
    k_values
)

plt.grid(
    alpha=0.3
)

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "hierarchical_singleton_clusters.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()


# ============================================================
# Plot 5: Smallest cluster sizes
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for linkage_method in LINKAGE_METHODS:

    method_results = (
        evaluation_df.loc[
            evaluation_df["Linkage"]
            == linkage_method
        ]
        .sort_values(
            "Requested k"
        )
    )

    plt.plot(
        method_results["Requested k"],
        method_results["Smallest Cluster"],
        marker="o",
        label=linkage_method.capitalize(),
    )

plt.xlabel(
    "Number of Clusters (k)"
)

plt.ylabel(
    "Smallest Cluster Size"
)

plt.title(
    f"Smallest Hierarchical Cluster by Linkage Method "
    f"({X.shape[0]} Companies)"
)

plt.xticks(
    k_values
)

plt.grid(
    alpha=0.3
)

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "hierarchical_smallest_cluster_sizes.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()


# ============================================================
# Plot 6: Largest cluster sizes
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for linkage_method in LINKAGE_METHODS:

    method_results = (
        evaluation_df.loc[
            evaluation_df["Linkage"]
            == linkage_method
        ]
        .sort_values(
            "Requested k"
        )
    )

    plt.plot(
        method_results["Requested k"],
        method_results["Largest Cluster"],
        marker="o",
        label=linkage_method.capitalize(),
    )

plt.xlabel(
    "Number of Clusters (k)"
)

plt.ylabel(
    "Largest Cluster Size"
)

plt.title(
    f"Largest Hierarchical Cluster by Linkage Method "
    f"({X.shape[0]} Companies)"
)

plt.xticks(
    k_values
)

plt.grid(
    alpha=0.3
)

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "hierarchical_largest_cluster_sizes.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close()


# ============================================================
# Save separate results for each linkage method
# ============================================================

for linkage_method in LINKAGE_METHODS:

    method_results = (
        evaluation_df.loc[
            evaluation_df["Linkage"]
            == linkage_method
        ]
        .sort_values(
            "Requested k"
        )
        .copy()
    )

    method_results.to_csv(
        os.path.join(
            RESULTS_DIR,
            f"{linkage_method}_evaluation_metrics.csv",
        ),
        index=False,
    )


# ============================================================
# Save candidate assignments for k = 2, 3 and 4
# ============================================================

candidate_k_values = [
    k
    for k in [2, 3, 4]
    if k in k_values
]

candidate_assignments_df = (
    cluster_assignments_df.loc[
        cluster_assignments_df[
            "k"
        ].isin(
            candidate_k_values
        )
    ]
    .copy()
)

candidate_assignments_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "hierarchical_candidate_assignments.csv",
    ),
    index=False,
)


# ============================================================
# Create candidate-cluster summary
# ============================================================

candidate_summary = (
    evaluation_df.loc[
        evaluation_df[
            "Requested k"
        ].isin(
            candidate_k_values
        )
    ][
        [
            "Linkage",
            "Requested k",
            "Actual Clusters",
            "Silhouette Score",
            "Davies-Bouldin Index",
            "Calinski-Harabasz Index",
            "Smallest Cluster",
            "Largest Cluster",
            "Singleton Clusters",
            "Mean Metric Rank",
        ]
    ]
    .sort_values(
        by=[
            "Requested k",
            "Mean Metric Rank",
        ]
    )
)

candidate_summary.to_csv(
    os.path.join(
        RESULTS_DIR,
        "hierarchical_candidate_summary.csv",
    ),
    index=False,
)


# ============================================================
# Print full evaluation results
# ============================================================

print("\n" + "=" * 100)
print("FULL HIERARCHICAL EVALUATION RESULTS")
print("=" * 100)

print(
    evaluation_df[
        [
            "Linkage",
            "Requested k",
            "Actual Clusters",
            "Silhouette Score",
            "Davies-Bouldin Index",
            "Calinski-Harabasz Index",
            "Smallest Cluster",
            "Largest Cluster",
            "Singleton Clusters",
        ]
    ]
    .sort_values(
        by=[
            "Linkage",
            "Requested k",
        ]
    )
    .to_string(
        index=False
    )
)


# ============================================================
# Print best results for each linkage
# ============================================================

print("\n" + "=" * 100)
print("BEST RESULT FOR EACH LINKAGE: SILHOUETTE SCORE")
print("=" * 100)

print(
    best_by_silhouette[
        [
            "Linkage",
            "Requested k",
            "Silhouette Score",
            "Davies-Bouldin Index",
            "Calinski-Harabasz Index",
            "Smallest Cluster",
            "Largest Cluster",
            "Singleton Clusters",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("BEST RESULT FOR EACH LINKAGE: DAVIES-BOULDIN INDEX")
print("=" * 100)

print(
    best_by_davies_bouldin[
        [
            "Linkage",
            "Requested k",
            "Silhouette Score",
            "Davies-Bouldin Index",
            "Calinski-Harabasz Index",
            "Smallest Cluster",
            "Largest Cluster",
            "Singleton Clusters",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 100)
print("BEST RESULT FOR EACH LINKAGE: CALINSKI-HARABASZ INDEX")
print("=" * 100)

print(
    best_by_calinski_harabasz[
        [
            "Linkage",
            "Requested k",
            "Silhouette Score",
            "Davies-Bouldin Index",
            "Calinski-Harabasz Index",
            "Smallest Cluster",
            "Largest Cluster",
            "Singleton Clusters",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Print overall metric optima
# ============================================================

print("\n" + "=" * 100)
print("OVERALL BEST RESULT FOR EACH METRIC")
print("=" * 100)

print(
    overall_metric_summary.to_string(
        index=False
    )
)


# ============================================================
# Print top-ranked results
# ============================================================

print("\n" + "=" * 100)
print("TOP 10 OVERALL RESULTS BY MEAN METRIC RANK")
print("=" * 100)

print(
    ranked_results[
        [
            "Linkage",
            "Requested k",
            "Silhouette Score",
            "Davies-Bouldin Index",
            "Calinski-Harabasz Index",
            "Smallest Cluster",
            "Largest Cluster",
            "Singleton Clusters",
            "Mean Metric Rank",
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# Print cluster memberships for k = 2, 3 and 4
# ============================================================

for linkage_method in LINKAGE_METHODS:

    print("\n" + "=" * 100)

    print(
        f"{linkage_method.upper()} LINKAGE "
        "CLUSTER MEMBERSHIPS"
    )

    print("=" * 100)

    for k in candidate_k_values:

        selected_assignments = (
            cluster_assignments_df.loc[
                (
                    cluster_assignments_df[
                        "Linkage"
                    ]
                    == linkage_method
                )
                & (
                    cluster_assignments_df[
                        "k"
                    ]
                    == k
                )
            ]
        )

        if selected_assignments.empty:
            continue

        print(f"\n{k} clusters:")

        grouped_tickers = (
            selected_assignments
            .groupby(
                "Cluster"
            )["Ticker"]
            .apply(
                lambda values: ", ".join(
                    sorted(values)
                )
            )
        )

        for (
            cluster_number,
            tickers,
        ) in grouped_tickers.items():

            cluster_size = int(
                (
                    selected_assignments[
                        "Cluster"
                    ]
                    == cluster_number
                )
                .sum()
            )

            print(
                f"\nCluster {cluster_number} "
                f"({cluster_size} companies):"
            )

            print(tickers)


# ============================================================
# Final validation
# ============================================================

expected_combinations = (
    len(LINKAGE_METHODS)
    * len(k_values)
)

actual_combinations = len(
    evaluation_df
)

print("\n" + "=" * 100)
print("HIERARCHICAL EVALUATION SUMMARY")
print("=" * 100)

print(
    f"Companies analysed: "
    f"{X.shape[0]}"
)

print(
    f"Trading days analysed: "
    f"{X.shape[1]}"
)

print(
    f"Linkage methods evaluated: "
    f"{', '.join(LINKAGE_METHODS)}"
)

print(
    f"k values evaluated: "
    f"{min(k_values)} to {max(k_values)}"
)

print(
    f"Expected method-k combinations: "
    f"{expected_combinations}"
)

print(
    f"Valid method-k combinations: "
    f"{actual_combinations}"
)


# ============================================================
# Finished
# ============================================================

print("\n" + "=" * 100)
print("HIERARCHICAL EVALUATION COMPLETED")
print("=" * 100)

print("\nResults saved to:")

saved_result_files = [
    evaluation_path,
    cluster_sizes_path,
    cluster_assignments_path,
    ranked_results_path,
    os.path.join(
        RESULTS_DIR,
        "hierarchical_cluster_balance.csv",
    ),
    os.path.join(
        RESULTS_DIR,
        "hierarchical_candidate_assignments.csv",
    ),
    os.path.join(
        RESULTS_DIR,
        "hierarchical_candidate_summary.csv",
    ),
    os.path.join(
        RESULTS_DIR,
        "hierarchical_overall_metric_optima.csv",
    ),
    os.path.join(
        RESULTS_DIR,
        "best_hierarchical_by_silhouette.csv",
    ),
    os.path.join(
        RESULTS_DIR,
        "best_hierarchical_by_davies_bouldin.csv",
    ),
    os.path.join(
        RESULTS_DIR,
        "best_hierarchical_by_calinski_harabasz.csv",
    ),
]

for file_path in saved_result_files:
    print(file_path)

print("\nFigures saved to:")

saved_figure_files = [
    "hierarchical_silhouette_scores.png",
    "hierarchical_davies_bouldin_indices.png",
    "hierarchical_calinski_harabasz_indices.png",
    "hierarchical_singleton_clusters.png",
    "hierarchical_smallest_cluster_sizes.png",
    "hierarchical_largest_cluster_sizes.png",
]

for filename in saved_figure_files:

    print(
        os.path.join(
            FIGURES_DIR,
            filename,
        )
    )