import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


# ============================================================
# File paths
# ============================================================

DATA_PATH = "data/raw/ftse100_40_companies.csv"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# Load the dataset
# ============================================================

data = pd.read_csv(
    DATA_PATH,
    header=[0, 1],
    index_col=0,
    parse_dates=True
)

print("\nOriginal dataset shape:")
print(data.shape)

print("\nColumn level names:")
print(data.columns.names)


# ============================================================
# Extract adjusted closing prices
# ============================================================

prices = data.xs(
    "Adj Close",
    level="Price",
    axis=1
)

# Convert all values to numeric.
# Any invalid values are converted to NaN.
prices = prices.apply(
    pd.to_numeric,
    errors="coerce"
)

# Sort dates in chronological order.
prices = prices.sort_index()

# Remove duplicated dates, keeping the first occurrence.
prices = prices.loc[
    ~prices.index.duplicated(keep="first")
]

print("\nAdjusted closing price shape:")
print(prices.shape)

print("\nCompanies included:")
print(prices.columns.tolist())


# ============================================================
# Inspect missing values
# ============================================================

missing_before = prices.isna().sum()
missing_before = missing_before[
    missing_before > 0
].sort_values(ascending=False)

print("\nMissing values before preprocessing:")

if missing_before.empty:
    print("No missing values found.")
else:
    print(missing_before)


# Save the missing-value summary.
missing_before.rename(
    "Missing values before filling"
).to_csv(
    os.path.join(
        RESULTS_DIR,
        "kmeans_missing_values_before_filling.csv"
    )
)


# ============================================================
# Handle missing values
# ============================================================

# Forward-fill missing values using the previous available price.
prices = prices.ffill()

# Backward-fill values missing at the beginning of a series.
prices = prices.bfill()


# Check whether any company still contains missing values.
remaining_missing = prices.isna().sum()
companies_with_remaining_nan = remaining_missing[
    remaining_missing > 0
].index.tolist()

if companies_with_remaining_nan:
    print(
        "\nRemoving companies with unresolved missing values:"
    )
    print(companies_with_remaining_nan)

    prices = prices.drop(
        columns=companies_with_remaining_nan
    )


# Remove companies whose initial price is zero.
# Dividing by zero during normalisation would create infinity.
zero_initial_price_companies = prices.columns[
    prices.iloc[0] == 0
].tolist()

if zero_initial_price_companies:
    print(
        "\nRemoving companies with an initial price of zero:"
    )
    print(zero_initial_price_companies)

    prices = prices.drop(
        columns=zero_initial_price_companies
    )


# Remove columns containing infinite values, if any.
prices = prices.replace(
    [np.inf, -np.inf],
    np.nan
)

infinite_or_missing_companies = prices.columns[
    prices.isna().any()
].tolist()

if infinite_or_missing_companies:
    print(
        "\nRemoving companies with invalid values after preprocessing:"
    )
    print(infinite_or_missing_companies)

    prices = prices.drop(
        columns=infinite_or_missing_companies
    )


# ============================================================
# Normalise prices
# ============================================================

# Each stock begins at 1.
normalised_prices = prices.div(
    prices.iloc[0],
    axis="columns"
)

normalised_prices = normalised_prices.replace(
    [np.inf, -np.inf],
    np.nan
)


# Final validation.
total_missing = int(
    normalised_prices.isna().sum().sum()
)

if total_missing > 0:
    raise ValueError(
        f"{total_missing} NaN values remain after preprocessing."
    )

if not np.isfinite(normalised_prices.to_numpy()).all():
    raise ValueError(
        "The normalised dataset contains infinite or invalid values."
    )


# Save the cleaned normalised data.
normalised_prices.to_csv(
    os.path.join(
        RESULTS_DIR,
        "normalised_prices_for_kmeans.csv"
    )
)


# ============================================================
# Prepare data for K-Means
# ============================================================

# Before transposing:
# Rows = dates
# Columns = companies
#
# After transposing:
# Rows = companies
# Columns = daily normalised prices

X = normalised_prices.T

# ------------------------------------------------
# Sensitivity analysis
# Remove Rolls-Royce (RR.L)
# ------------------------------------------------

X = X.drop(index="RR.L")

print("\nSensitivity analysis")
print("Rolls-Royce removed.")

print("\nCompanies remaining:")
print(X.shape[0])

print("\nFinal clustering data shape:")
print(X.shape)

print("\nNumber of companies:")
print(X.shape[0])

print("\nNumber of trading days:")
print(X.shape[1])

print("\nTotal missing values in X:")
print(X.isna().sum().sum())


# Ensure the requested k values are valid.
maximum_k = min(10, X.shape[0] - 1)

if maximum_k < 2:
    raise ValueError(
        "There are not enough companies to evaluate clustering."
    )

k_values = range(2, maximum_k + 1)


# ============================================================
# Evaluate K-Means for different k values
# ============================================================

wcss_values = []
silhouette_scores = []
davies_bouldin_scores = []
calinski_harabasz_scores = []

for k in k_values:

    print(f"Evaluating k = {k}")

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(X)

    # Within-cluster sum of squares.
    wcss_values.append(
        model.inertia_
    )

    # Higher values indicate better-defined clusters.
    silhouette_scores.append(
        silhouette_score(
            X,
            labels,
            metric="euclidean"
        )
    )

    # Lower values indicate better clustering.
    davies_bouldin_scores.append(
        davies_bouldin_score(
            X,
            labels
        )
    )

    # Higher values indicate better clustering.
    calinski_harabasz_scores.append(
        calinski_harabasz_score(
            X,
            labels
        )
    )


# ============================================================
# Create and save evaluation table
# ============================================================

results = pd.DataFrame({
    "k": list(k_values),
    "WCSS": wcss_values,
    "Silhouette Score": silhouette_scores,
    "Davies-Bouldin Index": davies_bouldin_scores,
    "Calinski-Harabasz Index": calinski_harabasz_scores
})

print("\nK-Means evaluation results:")
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


# ============================================================
# Elbow method plot
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    wcss_values,
    marker="o"
)

plt.title(
    "K-Means Elbow Method"
)

plt.xlabel(
    "Number of Clusters (k)"
)

plt.ylabel(
    "Within-Cluster Sum of Squares (WCSS)"
)

plt.xticks(
    list(k_values)
)

plt.grid(True)
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


# ============================================================
# Silhouette score plot
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o"
)

plt.title(
    "K-Means Silhouette Score"
)

plt.xlabel(
    "Number of Clusters (k)"
)

plt.ylabel(
    "Silhouette Score"
)

plt.xticks(
    list(k_values)
)

plt.grid(True)
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


# ============================================================
# Davies-Bouldin index plot
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    davies_bouldin_scores,
    marker="o"
)

plt.title(
    "K-Means Davies-Bouldin Index"
)

plt.xlabel(
    "Number of Clusters (k)"
)

plt.ylabel(
    "Davies-Bouldin Index"
)

plt.xticks(
    list(k_values)
)

plt.grid(True)
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


# ============================================================
# Calinski-Harabasz index plot
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    calinski_harabasz_scores,
    marker="o"
)

plt.title(
    "K-Means Calinski-Harabasz Index"
)

plt.xlabel(
    "Number of Clusters (k)"
)

plt.ylabel(
    "Calinski-Harabasz Index"
)

plt.xticks(
    list(k_values)
)

plt.grid(True)
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


print("\nK-Means evaluation completed successfully.")

print("\nSaved results:")
print(
    os.path.join(
        RESULTS_DIR,
        "kmeans_evaluation_metrics.csv"
    )
)

print("\nSaved figures:")
print(
    os.path.join(
        FIGURES_DIR,
        "kmeans_elbow_method.png"
    )
)
print(
    os.path.join(
        FIGURES_DIR,
        "kmeans_silhouette_score.png"
    )
)
print(
    os.path.join(
        FIGURES_DIR,
        "kmeans_davies_bouldin_index.png"
    )
)
print(
    os.path.join(
        FIGURES_DIR,
        "kmeans_calinski_harabasz_index.png"
    )
)

# ============================================================
# Inspect candidate K-Means solutions
# ============================================================

candidate_k_values = [2, 3, 4]

all_cluster_assignments = []

for k in candidate_k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(X)

    assignments = pd.DataFrame({
        "Ticker": X.index,
        "Cluster": labels + 1
    })

    assignments = assignments.sort_values(
        by=["Cluster", "Ticker"]
    )

    cluster_sizes = assignments[
        "Cluster"
    ].value_counts().sort_index()

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
        print(", ".join(members))

    assignments["k"] = k
    all_cluster_assignments.append(assignments)

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