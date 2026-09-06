import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import pairwise_distances



NORMALISED_DATA_PATH = "data/processed/normalised_prices.csv"
PROCESSED_DATA_DIR = "data/processed"
FIGURES_DIR = "results/figures"

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)



normalised_prices = pd.read_csv(
    NORMALISED_DATA_PATH,
    index_col=0,
    parse_dates=True
)

normalised_prices = normalised_prices.sort_index()
normalised_prices = normalised_prices.sort_index(axis=1)

print("=" * 60)
print("EUCLIDEAN DISTANCE ANALYSIS")
print("=" * 60)

print(f"Dataset shape: {normalised_prices.shape}")
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
        "The normalised dataset contains missing values for: "
        + ", ".join(companies_with_missing)
    )

if not np.isfinite(
    normalised_prices.to_numpy()
).all():
    raise ValueError(
        "The normalised dataset contains infinite or invalid values."
    )

if normalised_prices.empty:
    raise ValueError(
        "The normalised dataset is empty."
    )

print("Dataset validation completed successfully.")



stock_series = normalised_prices.T

print("\nClustering data shape:")
print(stock_series.shape)

print(
    f"Rows (companies): {stock_series.shape[0]}"
)
print(
    f"Columns (trading days): {stock_series.shape[1]}"
)




euclidean_array = pairwise_distances(
    stock_series,
    metric="euclidean"
)

euclidean_distances = pd.DataFrame(
    euclidean_array,
    index=stock_series.index,
    columns=stock_series.index
)

euclidean_distances.index.name = "Ticker"
euclidean_distances.columns.name = "Ticker"

print("\n" + "=" * 60)
print("EUCLIDEAN DISTANCE MATRIX")
print("=" * 60)

print(euclidean_distances.round(3))

euclidean_distances.to_csv(
    f"{PROCESSED_DATA_DIR}/euclidean_distance_matrix.csv"
)




distance_matrix = euclidean_distances.rename_axis(
    index=None,
    columns=None
)

pairs = (
    distance_matrix
    .stack()
    .reset_index()
)

pairs.columns = [
    "Stock 1",
    "Stock 2",
    "Distance"
]

# Remove self-comparisons.
pairs = pairs[
    pairs["Stock 1"] != pairs["Stock 2"]
].copy()

pairs["Pair"] = pairs.apply(
    lambda row: tuple(
        sorted(
            [
                row["Stock 1"],
                row["Stock 2"]
            ]
        )
    ),
    axis=1
)

pairs = (
    pairs
    .drop_duplicates(subset="Pair")
    .drop(columns="Pair")
    .reset_index(drop=True)
)

expected_pairs = (
    stock_series.shape[0]
    * (stock_series.shape[0] - 1)
    // 2
)

print(
    f"\nUnique stock pairs: {len(pairs)}"
)
print(
    f"Expected unique pairs: {expected_pairs}"
)

if len(pairs) != expected_pairs:
    raise ValueError(
        "The number of unique pairs is incorrect."
    )




closest_pairs = (
    pairs
    .nsmallest(10, "Distance")
    .reset_index(drop=True)
)

furthest_pairs = (
    pairs
    .nlargest(10, "Distance")
    .reset_index(drop=True)
)

print("\n" + "=" * 60)
print("10 MOST SIMILAR STOCK PAIRS")
print("=" * 60)

print(
    closest_pairs
    .round(3)
    .to_string(index=False)
)

print("\n" + "=" * 60)
print("10 MOST DISSIMILAR STOCK PAIRS")
print("=" * 60)

print(
    furthest_pairs
    .round(3)
    .to_string(index=False)
)

closest_pairs.to_csv(
    f"{PROCESSED_DATA_DIR}/closest_euclidean_pairs.csv",
    index=False
)

furthest_pairs.to_csv(
    f"{PROCESSED_DATA_DIR}/furthest_euclidean_pairs.csv",
    index=False
)




distance_summary = pairs["Distance"].describe()

print("\n" + "=" * 60)
print("EUCLIDEAN DISTANCE SUMMARY")
print("=" * 60)

print(distance_summary)

distance_summary.to_csv(
    f"{PROCESSED_DATA_DIR}/euclidean_distance_summary.csv",
    header=["Value"]
)




plt.figure(figsize=(16, 14))

heatmap = plt.imshow(
    euclidean_distances.to_numpy(),
    aspect="auto"
)

plt.colorbar(
    heatmap,
    label="Euclidean Distance"
)

plt.xticks(
    ticks=range(
        len(euclidean_distances.columns)
    ),
    labels=euclidean_distances.columns,
    rotation=90,
    fontsize=5
)

plt.yticks(
    ticks=range(
        len(euclidean_distances.index)
    ),
    labels=euclidean_distances.index,
    fontsize=5
)

plt.title(
    "Euclidean Distances Between Normalised Price "
    f"Trajectories ({stock_series.shape[0]} Companies)"
)

plt.xlabel("Company")
plt.ylabel("Company")
plt.tight_layout()

plt.savefig(
    f"{FIGURES_DIR}/euclidean_distance_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()




# Ward linkage is based on Euclidean geometry and therefore
# should be used with Euclidean distance.
linked = linkage(
    stock_series.to_numpy(),
    method="ward",
    metric="euclidean"
)

print("\n" + "=" * 60)
print("HIERARCHICAL CLUSTERING")
print("=" * 60)

print(f"Linkage matrix shape: {linked.shape}")
print(
    f"Maximum linkage distance: "
    f"{linked[:, 2].max():.3f}"
)



plt.figure(figsize=(20, 9))

dendrogram(
    linked,
    labels=stock_series.index.tolist(),
    leaf_rotation=90,
    leaf_font_size=7
)

plt.title(
    "Ward-Linkage Hierarchical Dendrogram of "
    f"{stock_series.shape[0]} FTSE 100 Constituents"
)

plt.xlabel("Company")
plt.ylabel("Ward Linkage Distance")
plt.tight_layout()

plt.savefig(
    f"{FIGURES_DIR}/ward_hierarchical_dendrogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()




linkage_table = pd.DataFrame(
    linked,
    columns=[
        "Cluster 1",
        "Cluster 2",
        "Linkage Distance",
        "Cluster Size"
    ]
)

linkage_table.to_csv(
    f"{PROCESSED_DATA_DIR}/ward_linkage_matrix.csv",
    index=False
)




print("\n" + "=" * 60)
print("ANALYSIS SUMMARY")
print("=" * 60)

print(
    f"Companies analysed: "
    f"{stock_series.shape[0]}"
)

print(
    f"Trading days analysed: "
    f"{stock_series.shape[1]}"
)

print(
    f"Unique company pairs compared: "
    f"{len(pairs)}"
)

print(
    "Distance metric: Euclidean distance"
)

print(
    "Hierarchical linkage method: Ward"
)

print("\nSaved processed files:")

print(
    f"- {PROCESSED_DATA_DIR}/"
    "euclidean_distance_matrix.csv"
)

print(
    f"- {PROCESSED_DATA_DIR}/"
    "closest_euclidean_pairs.csv"
)

print(
    f"- {PROCESSED_DATA_DIR}/"
    "furthest_euclidean_pairs.csv"
)

print(
    f"- {PROCESSED_DATA_DIR}/"
    "euclidean_distance_summary.csv"
)

print(
    f"- {PROCESSED_DATA_DIR}/"
    "ward_linkage_matrix.csv"
)

print("\nSaved figures:")

print(
    f"- {FIGURES_DIR}/"
    "euclidean_distance_heatmap.png"
)

print(
    f"- {FIGURES_DIR}/"
    "ward_hierarchical_dendrogram.png"
)

print("\nEuclidean-distance analysis completed successfully.")