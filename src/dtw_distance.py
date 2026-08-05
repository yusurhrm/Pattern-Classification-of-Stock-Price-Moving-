import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from tslearn.metrics import cdist_dtw


# ============================================================
# Configuration
# ============================================================

NORMALISED_DATA_PATH = "data/processed/normalised_prices.csv"
PROCESSED_DATA_DIR = "data/processed"
FIGURES_DIR = "results/figures"

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# Load cleaned normalised stock-price data
# ============================================================

normalised_prices = pd.read_csv(
    NORMALISED_DATA_PATH,
    index_col=0,
    parse_dates=True
)

normalised_prices = normalised_prices.sort_index()
normalised_prices = normalised_prices.sort_index(axis=1)

print("=" * 60)
print("DYNAMIC TIME WARPING DISTANCE ANALYSIS")
print("=" * 60)

print(f"Dataset shape: {normalised_prices.shape}")
print(
    f"Date range: {normalised_prices.index.min()} "
    f"to {normalised_prices.index.max()}"
)
print(f"Number of companies: {normalised_prices.shape[1]}")
print(f"Number of trading days: {normalised_prices.shape[0]}")


# ============================================================
# Validate the processed dataset
# ============================================================

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


# ============================================================
# Prepare data for DTW
# ============================================================

# tslearn expects:
# number of series × number of time steps × number of features
#
# Each stock is a univariate time series, so features = 1.

stock_names = normalised_prices.columns.tolist()

stock_series = normalised_prices.T.to_numpy(
    dtype=float
)

stock_series_3d = stock_series[:, :, np.newaxis]

print("\nDTW input shape:")
print(stock_series_3d.shape)

print(f"Series: {stock_series_3d.shape[0]}")
print(f"Time steps: {stock_series_3d.shape[1]}")
print(f"Features: {stock_series_3d.shape[2]}")


# ============================================================
# Calculate pairwise DTW distances
# ============================================================

print("\nCalculating DTW distance matrix...")
print(
    "This may take considerably longer than Euclidean or "
    "correlation distance."
)

start_time = time.perf_counter()

dtw_array = cdist_dtw(
    stock_series_3d,
    n_jobs=-1
)

elapsed_time = time.perf_counter() - start_time

print(
    f"DTW calculation completed in "
    f"{elapsed_time:.2f} seconds."
)

dtw_distances = pd.DataFrame(
    dtw_array,
    index=stock_names,
    columns=stock_names
)

dtw_distances.index.name = "Ticker"
dtw_distances.columns.name = "Ticker"

print("\n" + "=" * 60)
print("DTW DISTANCE MATRIX")
print("=" * 60)

print(dtw_distances.round(3))

dtw_distances.to_csv(
    f"{PROCESSED_DATA_DIR}/dtw_distance_matrix.csv"
)


# ============================================================
# Validate DTW matrix
# ============================================================

if not np.isfinite(dtw_array).all():
    raise ValueError(
        "The DTW distance matrix contains invalid values."
    )

if not np.allclose(
    dtw_array,
    dtw_array.T,
    atol=1e-8
):
    raise ValueError(
        "The DTW distance matrix is not symmetric."
    )

if not np.allclose(
    np.diag(dtw_array),
    0,
    atol=1e-8
):
    raise ValueError(
        "The diagonal of the DTW matrix is not zero."
    )

print("\nDTW matrix validation completed successfully.")


# ============================================================
# Convert distance matrix into unique stock pairs
# ============================================================

distance_matrix = dtw_distances.rename_axis(
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

# Treat A-B and B-A as one pair.
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
    len(stock_names)
    * (len(stock_names) - 1)
    // 2
)

print(f"\nUnique stock pairs: {len(pairs)}")
print(f"Expected unique pairs: {expected_pairs}")

if len(pairs) != expected_pairs:
    raise ValueError(
        "The number of unique pairs is incorrect."
    )


# ============================================================
# Find most similar and most dissimilar pairs
# ============================================================

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
    f"{PROCESSED_DATA_DIR}/closest_dtw_pairs.csv",
    index=False
)

furthest_pairs.to_csv(
    f"{PROCESSED_DATA_DIR}/furthest_dtw_pairs.csv",
    index=False
)


# ============================================================
# DTW distance summary
# ============================================================

distance_summary = pairs["Distance"].describe()

print("\n" + "=" * 60)
print("DTW DISTANCE SUMMARY")
print("=" * 60)

print(distance_summary)

distance_summary.to_csv(
    f"{PROCESSED_DATA_DIR}/dtw_distance_summary.csv",
    header=["Value"]
)


# ============================================================
# Plot DTW distance heatmap
# ============================================================

plt.figure(figsize=(16, 14))

heatmap = plt.imshow(
    dtw_distances.to_numpy(),
    aspect="auto"
)

plt.colorbar(
    heatmap,
    label="DTW Distance"
)

plt.xticks(
    ticks=range(len(stock_names)),
    labels=stock_names,
    rotation=90,
    fontsize=5
)

plt.yticks(
    ticks=range(len(stock_names)),
    labels=stock_names,
    fontsize=5
)

plt.title(
    "Dynamic Time Warping Distances Between Normalised "
    f"Price Trajectories ({len(stock_names)} Companies)"
)

plt.xlabel("Company")
plt.ylabel("Company")
plt.tight_layout()

plt.savefig(
    f"{FIGURES_DIR}/dtw_distance_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# Hierarchical clustering from precomputed DTW distances
# ============================================================

# Convert the full square matrix into condensed form.
condensed_dtw = squareform(
    dtw_distances.to_numpy(),
    checks=True
)

# Average linkage can use a precomputed distance matrix.
linked = linkage(
    condensed_dtw,
    method="average"
)

print("\n" + "=" * 60)
print("HIERARCHICAL CLUSTERING")
print("=" * 60)

print(f"Linkage matrix shape: {linked.shape}")
print(
    f"Maximum linkage distance: "
    f"{linked[:, 2].max():.3f}"
)


# ============================================================
# Plot DTW dendrogram
# ============================================================

plt.figure(figsize=(20, 9))

dendrogram(
    linked,
    labels=stock_names,
    leaf_rotation=90,
    leaf_font_size=7
)

plt.title(
    "Average-Linkage Hierarchical Dendrogram Using "
    f"Dynamic Time Warping ({len(stock_names)} Companies)"
)

plt.xlabel("Company")
plt.ylabel("DTW Distance")
plt.tight_layout()

plt.savefig(
    f"{FIGURES_DIR}/dtw_distance_dendrogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# Save linkage matrix
# ============================================================

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
    f"{PROCESSED_DATA_DIR}/dtw_average_linkage_matrix.csv",
    index=False
)


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 60)
print("ANALYSIS SUMMARY")
print("=" * 60)

print(f"Companies analysed: {len(stock_names)}")
print(f"Trading days analysed: {normalised_prices.shape[0]}")
print(f"Unique company pairs compared: {len(pairs)}")
print("Distance metric: Dynamic Time Warping")
print("Hierarchical linkage method: Average")
print(f"DTW computation time: {elapsed_time:.2f} seconds")

print("\nSaved processed files:")
print(
    f"- {PROCESSED_DATA_DIR}/dtw_distance_matrix.csv"
)
print(
    f"- {PROCESSED_DATA_DIR}/closest_dtw_pairs.csv"
)
print(
    f"- {PROCESSED_DATA_DIR}/furthest_dtw_pairs.csv"
)
print(
    f"- {PROCESSED_DATA_DIR}/dtw_distance_summary.csv"
)
print(
    f"- {PROCESSED_DATA_DIR}/dtw_average_linkage_matrix.csv"
)

print("\nSaved figures:")
print(
    f"- {FIGURES_DIR}/dtw_distance_heatmap.png"
)
print(
    f"- {FIGURES_DIR}/dtw_distance_dendrogram.png"
)

print("\nDTW-distance analysis completed successfully.")