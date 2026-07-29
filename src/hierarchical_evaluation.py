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

DATA_PATH = "data/raw/ftse100_40_companies.csv"
RESULTS_DIR = "results/hierarchical_evaluation"
FIGURES_DIR = "figures/hierarchical_evaluation"

MIN_CLUSTERS = 2
MAX_CLUSTERS = 10

# Linkage methods to compare
LINKAGE_METHODS = [
    "ward",
    "complete",
    "average",
    "single",
]

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# Load stock-price data
# ============================================================

data = pd.read_csv(
    DATA_PATH,
    header=[0, 1],
    index_col=0,
    parse_dates=True,
)

print("\nOriginal dataset shape:")
print(data.shape)

print("\nColumn level names:")
print(data.columns.names)


# ============================================================
# Extract adjusted closing prices
# ============================================================

try:
    prices = data.xs(
        "Adj Close",
        level="Price",
        axis=1,
    )
except (KeyError, ValueError):
    # Fallback in case the column level is unnamed
    prices = data.xs(
        "Adj Close",
        level=0,
        axis=1,
    )

prices = prices.apply(
    pd.to_numeric,
    errors="coerce",
)

prices = prices.sort_index()

# Remove duplicated dates if present
prices = prices.loc[
    ~prices.index.duplicated(keep="first")
]

print("\nAdjusted closing-price shape:")
print(prices.shape)

print("\nCompanies included:")
print(prices.columns.tolist())


# ============================================================
# Check and handle missing values
# ============================================================

missing_before = prices.isna().sum()

missing_before = missing_before[
    missing_before > 0
].sort_values(ascending=False)

print("\nMissing values before filling:")

if missing_before.empty:
    print("No missing values found.")
else:
    print(missing_before)

# Fill isolated missing observations
prices = prices.ffill().bfill()

# Replace infinite values
prices = prices.replace(
    [np.inf, -np.inf],
    np.nan,
)

if prices.isna().any().any():
    unresolved = prices.isna().sum()

    unresolved = unresolved[
        unresolved > 0
    ]

    raise ValueError(
        "Missing values remain after preprocessing:\n"
        f"{unresolved}"
    )


# ============================================================
# Normalise stock prices
# ============================================================

# Each company begins at 1.0
normalised_prices = prices.div(
    prices.iloc[0],
    axis="columns",
)

normalised_prices = normalised_prices.replace(
    [np.inf, -np.inf],
    np.nan,
)

if normalised_prices.isna().any().any():
    raise ValueError(
        "Invalid values were produced during normalisation."
    )

print("\nNormalised dataset shape:")
print(normalised_prices.shape)


# ============================================================
# Prepare data for clustering
# ============================================================

# Rows must represent companies
# Columns represent trading dates
X = normalised_prices.T

print("\nClustering matrix shape:")
print(X.shape)

print(
    f"\nEach of the {X.shape[0]} companies is represented by "
    f"{X.shape[1]} trading-day observations."
)

ticker_names = X.index.tolist()


# ============================================================
# Evaluate hierarchical clustering
# ============================================================

evaluation_results = []
cluster_size_results = []
cluster_assignment_results = []

for linkage_method in LINKAGE_METHODS:

    print("\n" + "=" * 80)
    print(
        f"EVALUATING LINKAGE METHOD: "
        f"{linkage_method.upper()}"
    )
    print("=" * 80)

    # Ward linkage is defined using Euclidean distance.
    # The other methods also use Euclidean distance here to keep
    # the algorithm comparison consistent with K-Means.
    linkage_matrix = linkage(
        X.values,
        method=linkage_method,
        metric="euclidean",
    )

    for k in range(
        MIN_CLUSTERS,
        MAX_CLUSTERS + 1,
    ):

        # fcluster labels begin at 1
        labels = fcluster(
            linkage_matrix,
            t=k,
            criterion="maxclust",
        )

        unique_labels = np.unique(labels)
        actual_clusters = len(unique_labels)

        # Some linkage configurations may return fewer than k clusters
        # because of tied distances.
        if actual_clusters < 2:
            print(
                f"k={k}: skipped because only "
                f"{actual_clusters} cluster was produced."
            )
            continue

        if actual_clusters >= len(X):
            print(
                f"k={k}: skipped because each observation "
                "would form its own cluster."
            )
            continue

        silhouette = silhouette_score(
            X.values,
            labels,
            metric="euclidean",
        )

        davies_bouldin = davies_bouldin_score(
            X.values,
            labels,
        )

        calinski_harabasz = calinski_harabasz_score(
            X.values,
            labels,
        )

        cluster_sizes = pd.Series(
            labels
        ).value_counts().sort_index()

        evaluation_results.append(
            {
                "Linkage": linkage_method,
                "Requested k": k,
                "Actual Clusters": actual_clusters,
                "Silhouette Score": silhouette,
                "Davies-Bouldin Index": davies_bouldin,
                "Calinski-Harabasz Index": calinski_harabasz,
                "Smallest Cluster": cluster_sizes.min(),
                "Largest Cluster": cluster_sizes.max(),
                "Singleton Clusters": int(
                    (cluster_sizes == 1).sum()
                ),
            }
        )

        for cluster_number, cluster_size in (
            cluster_sizes.items()
        ):
            cluster_size_results.append(
                {
                    "Linkage": linkage_method,
                    "k": k,
                    "Cluster": cluster_number,
                    "Cluster Size": cluster_size,
                }
            )

        for ticker, cluster_number in zip(
            ticker_names,
            labels,
        ):
            cluster_assignment_results.append(
                {
                    "Linkage": linkage_method,
                    "k": k,
                    "Ticker": ticker,
                    "Cluster": cluster_number,
                }
            )

        print(
            f"k={k:<2} | "
            f"Silhouette={silhouette:.4f} | "
            f"DB={davies_bouldin:.4f} | "
            f"CH={calinski_harabasz:.2f} | "
            f"Sizes={cluster_sizes.to_dict()}"
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
# Identify best result for each linkage method
# ============================================================

best_by_silhouette = (
    evaluation_df
    .sort_values(
        by=[
            "Linkage",
            "Silhouette Score",
        ],
        ascending=[
            True,
            False,
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
        ],
        ascending=[
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
        ],
        ascending=[
            True,
            False,
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
# Overall rankings
# ============================================================

# Higher silhouette is better
evaluation_df["Silhouette Rank"] = (
    evaluation_df["Silhouette Score"]
    .rank(
        ascending=False,
        method="min",
    )
)

# Lower Davies-Bouldin is better
evaluation_df["Davies-Bouldin Rank"] = (
    evaluation_df["Davies-Bouldin Index"]
    .rank(
        ascending=True,
        method="min",
    )
)

# Higher Calinski-Harabasz is better
evaluation_df["Calinski-Harabasz Rank"] = (
    evaluation_df["Calinski-Harabasz Index"]
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
        "Silhouette Score",
    ],
    ascending=[
        True,
        True,
        False,
    ],
)

ranked_results.to_csv(
    os.path.join(
        RESULTS_DIR,
        "hierarchical_ranked_results.csv",
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

    method_results = evaluation_df.loc[
        evaluation_df["Linkage"]
        == linkage_method
    ]

    plt.plot(
        method_results["Requested k"],
        method_results["Silhouette Score"],
        marker="o",
        label=linkage_method.capitalize(),
    )

plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette score")
plt.title(
    "Hierarchical Clustering Silhouette Scores"
)
plt.xticks(
    range(
        MIN_CLUSTERS,
        MAX_CLUSTERS + 1,
    )
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

    method_results = evaluation_df.loc[
        evaluation_df["Linkage"]
        == linkage_method
    ]

    plt.plot(
        method_results["Requested k"],
        method_results["Davies-Bouldin Index"],
        marker="o",
        label=linkage_method.capitalize(),
    )

plt.xlabel("Number of clusters (k)")
plt.ylabel("Davies-Bouldin index")
plt.title(
    "Hierarchical Clustering Davies-Bouldin Indices"
)
plt.xticks(
    range(
        MIN_CLUSTERS,
        MAX_CLUSTERS + 1,
    )
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

    method_results = evaluation_df.loc[
        evaluation_df["Linkage"]
        == linkage_method
    ]

    plt.plot(
        method_results["Requested k"],
        method_results["Calinski-Harabasz Index"],
        marker="o",
        label=linkage_method.capitalize(),
    )

plt.xlabel("Number of clusters (k)")
plt.ylabel("Calinski-Harabasz index")
plt.title(
    "Hierarchical Clustering Calinski-Harabasz Indices"
)
plt.xticks(
    range(
        MIN_CLUSTERS,
        MAX_CLUSTERS + 1,
    )
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

    method_results = evaluation_df.loc[
        evaluation_df["Linkage"]
        == linkage_method
    ]

    plt.plot(
        method_results["Requested k"],
        method_results["Singleton Clusters"],
        marker="o",
        label=linkage_method.capitalize(),
    )

plt.xlabel("Number of clusters (k)")
plt.ylabel("Number of singleton clusters")
plt.title(
    "Singleton Clusters by Linkage Method"
)
plt.xticks(
    range(
        MIN_CLUSTERS,
        MAX_CLUSTERS + 1,
    )
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
# Save separate results for each linkage method
# ============================================================

for linkage_method in LINKAGE_METHODS:

    method_results = evaluation_df.loc[
        evaluation_df["Linkage"]
        == linkage_method
    ].copy()

    method_results.to_csv(
        os.path.join(
            RESULTS_DIR,
            f"{linkage_method}_evaluation_metrics.csv",
        ),
        index=False,
    )


# ============================================================
# Print summaries
# ============================================================

print("\n" + "=" * 90)
print("FULL HIERARCHICAL EVALUATION RESULTS")
print("=" * 90)

print(
    evaluation_df[
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


print("\n" + "=" * 90)
print("BEST RESULT FOR EACH LINKAGE: SILHOUETTE")
print("=" * 90)

print(
    best_by_silhouette[
        [
            "Linkage",
            "Requested k",
            "Silhouette Score",
            "Davies-Bouldin Index",
            "Calinski-Harabasz Index",
            "Singleton Clusters",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 90)
print("BEST RESULT FOR EACH LINKAGE: DAVIES-BOULDIN")
print("=" * 90)

print(
    best_by_davies_bouldin[
        [
            "Linkage",
            "Requested k",
            "Silhouette Score",
            "Davies-Bouldin Index",
            "Calinski-Harabasz Index",
            "Singleton Clusters",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 90)
print("BEST RESULT FOR EACH LINKAGE: CALINSKI-HARABASZ")
print("=" * 90)

print(
    best_by_calinski_harabasz[
        [
            "Linkage",
            "Requested k",
            "Silhouette Score",
            "Davies-Bouldin Index",
            "Calinski-Harabasz Index",
            "Singleton Clusters",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 90)
print("TOP 10 OVERALL RESULTS BY MEAN METRIC RANK")
print("=" * 90)

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

    print("\n" + "=" * 90)
    print(
        f"{linkage_method.upper()} LINKAGE "
        "CLUSTER MEMBERSHIPS"
    )
    print("=" * 90)

    for k in [2, 3, 4]:

        selected_assignments = (
            cluster_assignments_df.loc[
                (
                    cluster_assignments_df["Linkage"]
                    == linkage_method
                )
                & (
                    cluster_assignments_df["k"]
                    == k
                )
            ]
        )

        if selected_assignments.empty:
            continue

        print(f"\n{k} clusters:")

        grouped_tickers = (
            selected_assignments
            .groupby("Cluster")["Ticker"]
            .apply(
                lambda values: ", ".join(
                    sorted(values)
                )
            )
        )

        for cluster_number, tickers in (
            grouped_tickers.items()
        ):

            cluster_size = (
                selected_assignments[
                    "Cluster"
                ]
                == cluster_number
            ).sum()

            print(
                f"Cluster {cluster_number} "
                f"({cluster_size} companies):"
            )

            print(tickers)


# ============================================================
# Finished
# ============================================================

print("\n" + "=" * 90)
print("HIERARCHICAL EVALUATION COMPLETED")
print("=" * 90)

print("\nResults saved to:")

saved_result_files = [
    evaluation_path,
    cluster_sizes_path,
    cluster_assignments_path,
    os.path.join(
        RESULTS_DIR,
        "hierarchical_ranked_results.csv",
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
]

for filename in saved_figure_files:
    print(
        os.path.join(
            FIGURES_DIR,
            filename,
        )
    )