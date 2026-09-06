import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score
)


DATA_PATH = "data/processed/normalised_prices.csv"
RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"

RANDOM_STATE = 42
N_INIT = 20
MAX_K = 10

# Final value selected after considering:
# - Elbow Method
# - Silhouette Score
# - Davies-Bouldin Index
# - Calinski-Harabasz Index
# - Cluster-size balance
# - Interpretability
FINAL_K = 3

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


normalised_prices = pd.read_csv(
    DATA_PATH,
    index_col=0,
    parse_dates=True
)

normalised_prices = normalised_prices.sort_index()
normalised_prices = normalised_prices.sort_index(axis=1)

print("=" * 60)
print("K-MEANS CLUSTERING ANALYSIS")
print("=" * 60)

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

constant_companies = normalised_prices.columns[
    normalised_prices.nunique() <= 1
].tolist()

if constant_companies:
    raise ValueError(
        "Constant price series were found for: "
        + ", ".join(constant_companies)
    )

print("Input validation completed successfully.")


X = normalised_prices.T

print("\nFinal clustering data shape:")
print(X.shape)

print(f"Rows (companies): {X.shape[0]}")
print(f"Columns (trading days): {X.shape[1]}")


maximum_k = min(
    MAX_K,
    X.shape[0] - 1
)

if maximum_k < 2:
    raise ValueError(
        "There are not enough companies for clustering."
    )

k_values = list(
    range(2, maximum_k + 1)
)

print("\nValues of k to evaluate:")
print(k_values)



evaluation_rows = []

for k in k_values:

    print(f"\nEvaluating k = {k}")

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=N_INIT
    )

    labels = model.fit_predict(X)

    cluster_sizes = pd.Series(
        labels
    ).value_counts()

    smallest_cluster = int(
        cluster_sizes.min()
    )

    largest_cluster = int(
        cluster_sizes.max()
    )

    singleton_clusters = int(
        (cluster_sizes == 1).sum()
    )

    silhouette = silhouette_score(
        X,
        labels,
        metric="euclidean"
    )

    davies_bouldin = davies_bouldin_score(
        X,
        labels
    )

    calinski_harabasz = calinski_harabasz_score(
        X,
        labels
    )

    evaluation_rows.append({
        "k": k,
        "WCSS": model.inertia_,
        "Silhouette Score": silhouette,
        "Davies-Bouldin Index": davies_bouldin,
        "Calinski-Harabasz Index": calinski_harabasz,
        "Smallest Cluster": smallest_cluster,
        "Largest Cluster": largest_cluster,
        "Singleton Clusters": singleton_clusters
    })



results = pd.DataFrame(
    evaluation_rows
)

print("\n" + "=" * 60)
print("K-MEANS EVALUATION RESULTS")
print("=" * 60)

print(
    results.to_string(
        index=False
    )
)

results.to_csv(
    os.path.join(
        RESULTS_DIR,
        "kmeans_evaluation_metrics.csv"
    ),
    index=False
)



best_silhouette_k = int(
    results.loc[
        results["Silhouette Score"].idxmax(),
        "k"
    ]
)

best_davies_bouldin_k = int(
    results.loc[
        results["Davies-Bouldin Index"].idxmin(),
        "k"
    ]
)

best_calinski_harabasz_k = int(
    results.loc[
        results["Calinski-Harabasz Index"].idxmax(),
        "k"
    ]
)

print("\n" + "=" * 60)
print("BEST k ACCORDING TO EACH METRIC")
print("=" * 60)

print(
    f"Highest Silhouette Score: "
    f"k = {best_silhouette_k}"
)

print(
    f"Lowest Davies-Bouldin Index: "
    f"k = {best_davies_bouldin_k}"
)

print(
    f"Highest Calinski-Harabasz Index: "
    f"k = {best_calinski_harabasz_k}"
)

print(
    f"Final selected k: "
    f"k = {FINAL_K}"
)



selection_summary = pd.DataFrame({
    "Method": [
        "Silhouette Score",
        "Davies-Bouldin Index",
        "Calinski-Harabasz Index",
        "Final Selected k"
    ],
    "Selected k": [
        best_silhouette_k,
        best_davies_bouldin_k,
        best_calinski_harabasz_k,
        FINAL_K
    ],
    "Reason": [
        "Highest Silhouette Score",
        "Lowest Davies-Bouldin Index",
        "Highest Calinski-Harabasz Index",
        (
            "Selected using all metrics, the elbow method, "
            "cluster-size balance and interpretability"
        )
    ]
})

selection_summary.to_csv(
    os.path.join(
        RESULTS_DIR,
        "kmeans_k_selection_summary.csv"
    ),
    index=False
)

print("\nK-selection summary:")
print(
    selection_summary.to_string(
        index=False
    )
)


plt.figure(figsize=(8, 5))

plt.plot(
    results["k"],
    results["WCSS"],
    marker="o"
)

plt.axvline(
    FINAL_K,
    linestyle="--",
    label=f"Selected k = {FINAL_K}"
)

plt.title(
    f"K-Means Elbow Method "
    f"({X.shape[0]} Companies)"
)

plt.xlabel(
    "Number of Clusters (k)"
)

plt.ylabel(
    "Within-Cluster Sum of Squares"
)

plt.xticks(
    results["k"]
)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "kmeans_elbow_method.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


plt.figure(figsize=(8, 5))

plt.plot(
    results["k"],
    results["Silhouette Score"],
    marker="o"
)

plt.axvline(
    best_silhouette_k,
    linestyle="--",
    label=f"Metric optimum k = {best_silhouette_k}"
)

plt.axvline(
    FINAL_K,
    linestyle=":",
    label=f"Final selected k = {FINAL_K}"
)

plt.title(
    f"K-Means Silhouette Scores "
    f"({X.shape[0]} Companies)"
)

plt.xlabel(
    "Number of Clusters (k)"
)

plt.ylabel(
    "Silhouette Score"
)

plt.xticks(
    results["k"]
)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "kmeans_silhouette_score.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


plt.figure(figsize=(8, 5))

plt.plot(
    results["k"],
    results["Davies-Bouldin Index"],
    marker="o"
)

plt.axvline(
    best_davies_bouldin_k,
    linestyle="--",
    label=f"Metric optimum k = {best_davies_bouldin_k}"
)

plt.axvline(
    FINAL_K,
    linestyle=":",
    label=f"Final selected k = {FINAL_K}"
)

plt.title(
    f"K-Means Davies-Bouldin Index "
    f"({X.shape[0]} Companies)"
)

plt.xlabel(
    "Number of Clusters (k)"
)

plt.ylabel(
    "Davies-Bouldin Index"
)

plt.xticks(
    results["k"]
)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "kmeans_davies_bouldin_index.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


plt.figure(figsize=(8, 5))

plt.plot(
    results["k"],
    results["Calinski-Harabasz Index"],
    marker="o"
)

plt.axvline(
    best_calinski_harabasz_k,
    linestyle="--",
    label=f"Metric optimum k = {best_calinski_harabasz_k}"
)

plt.axvline(
    FINAL_K,
    linestyle=":",
    label=f"Final selected k = {FINAL_K}"
)

plt.title(
    f"K-Means Calinski-Harabasz Index "
    f"({X.shape[0]} Companies)"
)

plt.xlabel(
    "Number of Clusters (k)"
)

plt.ylabel(
    "Calinski-Harabasz Index"
)

plt.xticks(
    results["k"]
)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "kmeans_calinski_harabasz_index.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


plt.figure(figsize=(8, 5))

plt.plot(
    results["k"],
    results["Smallest Cluster"],
    marker="o",
    label="Smallest cluster"
)

plt.plot(
    results["k"],
    results["Largest Cluster"],
    marker="o",
    label="Largest cluster"
)

plt.axvline(
    FINAL_K,
    linestyle="--",
    label=f"Selected k = {FINAL_K}"
)

plt.title(
    "K-Means Cluster-Size Balance"
)

plt.xlabel(
    "Number of Clusters (k)"
)

plt.ylabel(
    "Number of Companies"
)

plt.xticks(
    results["k"]
)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "kmeans_cluster_size_balance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


candidate_k_values = sorted(
    set(
        [
            2,
            3,
            4,
            best_silhouette_k,
            best_davies_bouldin_k,
            best_calinski_harabasz_k,
            FINAL_K
        ]
    )
)

candidate_k_values = [
    k
    for k in candidate_k_values
    if 2 <= k <= maximum_k
]

print("\nCandidate values of k:")
print(candidate_k_values)

all_cluster_assignments = []
candidate_summary_rows = []

for k in candidate_k_values:

    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=N_INIT
    )

    labels = model.fit_predict(X)

    assignments = pd.DataFrame({
        "Ticker": X.index,
        "Cluster": labels + 1
    })

    assignments = assignments.sort_values(
        by=["Cluster", "Ticker"]
    )

    cluster_sizes = (
        assignments["Cluster"]
        .value_counts()
        .sort_index()
    )

    print("\n" + "=" * 60)
    print(f"K-MEANS RESULTS FOR k = {k}")
    print("=" * 60)

    print("\nCluster sizes:")
    print(cluster_sizes)

    for cluster_number in sorted(
        assignments["Cluster"].unique()
    ):

        members = assignments.loc[
            assignments["Cluster"] == cluster_number,
            "Ticker"
        ].tolist()

        print(
            f"\nCluster {cluster_number} "
            f"({len(members)} companies):"
        )

        print(
            ", ".join(members)
        )

    candidate_summary_rows.append({
        "k": k,
        "Smallest Cluster": int(
            cluster_sizes.min()
        ),
        "Largest Cluster": int(
            cluster_sizes.max()
        ),
        "Singleton Clusters": int(
            (cluster_sizes == 1).sum()
        )
    })

    assignments["k"] = k

    all_cluster_assignments.append(
        assignments
    )

    assignments.to_csv(
        os.path.join(
            RESULTS_DIR,
            f"kmeans_cluster_assignments_k{k}.csv"
        ),
        index=False
    )


combined_assignments = pd.concat(
    all_cluster_assignments,
    ignore_index=True
)

combined_assignments.to_csv(
    os.path.join(
        RESULTS_DIR,
        "kmeans_candidate_cluster_assignments.csv"
    ),
    index=False
)

candidate_summary = pd.DataFrame(
    candidate_summary_rows
)

candidate_summary.to_csv(
    os.path.join(
        RESULTS_DIR,
        "kmeans_candidate_cluster_summary.csv"
    ),
    index=False
)


print("\n" + "=" * 60)
print("K-MEANS EVALUATION SUMMARY")
print("=" * 60)

print(f"Companies analysed: {X.shape[0]}")
print(f"Trading days analysed: {X.shape[1]}")
print(f"k values evaluated: {min(k_values)} to {max(k_values)}")

print(f"Highest Silhouette Score: k = {best_silhouette_k}")
print(f"Lowest Davies-Bouldin Index: k = {best_davies_bouldin_k}")
print(f"Highest Calinski-Harabasz Index: k = {best_calinski_harabasz_k}")
print(f"Interpretation-selected candidate k: {FINAL_K}")

print("\nSaved result files:")
print(os.path.join(RESULTS_DIR, "kmeans_evaluation_metrics.csv"))
print(os.path.join(RESULTS_DIR, "kmeans_k_selection_summary.csv"))
print(os.path.join(RESULTS_DIR, "kmeans_candidate_cluster_assignments.csv"))
print(os.path.join(RESULTS_DIR, "kmeans_candidate_cluster_summary.csv"))

print("\nSaved figures:")
print(os.path.join(FIGURES_DIR, "kmeans_elbow_method.png"))
print(os.path.join(FIGURES_DIR, "kmeans_silhouette_score.png"))
print(os.path.join(FIGURES_DIR, "kmeans_davies_bouldin_index.png"))
print(os.path.join(FIGURES_DIR, "kmeans_calinski_harabasz_index.png"))
print(os.path.join(FIGURES_DIR, "kmeans_cluster_size_balance.png"))

print("\nK-Means evaluation completed successfully.")
