import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from tslearn.metrics import cdist_dtw


# --------------------------------------------------
# 1. Create output folders
# --------------------------------------------------

os.makedirs("data/processed", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)


# --------------------------------------------------
# 2. Load the stock dataset
# --------------------------------------------------

data = pd.read_csv(
    "data/raw/ftse100_40_companies.csv",
    header=[0, 1],
    index_col=0,
    parse_dates=True
)


# --------------------------------------------------
# 3. Extract adjusted closing prices
# --------------------------------------------------

prices = data.xs(
    "Adj Close",
    level="Price",
    axis=1
)

# Remove companies with no data
prices = prices.dropna(axis=1, how="all")

# Fill occasional missing values
prices = prices.ffill().bfill()

print("\nAdjusted closing prices:")
print(prices.head())

print("\nPrice data shape:")
print(prices.shape)


# --------------------------------------------------
# 4. Normalise prices by their starting value
# --------------------------------------------------

normalised_prices = prices / prices.iloc[0]

normalised_prices.to_csv(
    "data/processed/normalised_stock_prices.csv"
)

print("\nNormalised prices:")
print(normalised_prices.head())


# --------------------------------------------------
# 5. Prepare the data for DTW
# --------------------------------------------------

# tslearn expects data in the format:
# number of stocks × number of dates × number of features
#
# Each stock is a univariate time series, so the final
# feature dimension is 1.

stock_names = normalised_prices.columns

stock_series = normalised_prices.T.to_numpy()

stock_series_3d = stock_series[:, :, np.newaxis]

print("\nDTW input shape:")
print(stock_series_3d.shape)


# --------------------------------------------------
# 6. Calculate the DTW distance matrix
# --------------------------------------------------

dtw_array = cdist_dtw(
    stock_series_3d,
    n_jobs=-1
)

dtw_distances = pd.DataFrame(
    dtw_array,
    index=stock_names,
    columns=stock_names
)

print("\nDTW Distance Matrix:")
print(dtw_distances.round(2))

dtw_distances.to_csv(
    "data/processed/dtw_distance_matrix.csv"
)


# --------------------------------------------------
# 7. Convert the matrix into unique stock pairs
# --------------------------------------------------

distance_matrix = dtw_distances.rename_axis(
    index=None,
    columns=None
)

pairs = distance_matrix.stack().reset_index()

pairs.columns = [
    "Stock 1",
    "Stock 2",
    "Distance"
]

# Remove comparisons of each stock with itself
pairs = pairs[
    pairs["Stock 1"] != pairs["Stock 2"]
].copy()

# Treat A-B and B-A as the same pair
pairs["Pair"] = pairs.apply(
    lambda row: tuple(
        sorted([row["Stock 1"], row["Stock 2"]])
    ),
    axis=1
)

pairs = (
    pairs
    .drop_duplicates(subset="Pair")
    .drop(columns="Pair")
)


# --------------------------------------------------
# 8. Find the closest and furthest stock pairs
# --------------------------------------------------

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

print("\n10 Most Similar Stock Pairs:")
print(
    closest_pairs
    .round(3)
    .to_string(index=False)
)

print("\n10 Most Dissimilar Stock Pairs:")
print(
    furthest_pairs
    .round(3)
    .to_string(index=False)
)

closest_pairs.to_csv(
    "data/processed/closest_dtw_pairs.csv",
    index=False
)

furthest_pairs.to_csv(
    "data/processed/furthest_dtw_pairs.csv",
    index=False
)


# --------------------------------------------------
# 9. Plot the DTW distance heatmap
# --------------------------------------------------

plt.figure(figsize=(14, 12))

plt.imshow(
    dtw_distances,
    aspect="auto"
)

plt.colorbar(
    label="DTW Distance"
)

plt.xticks(
    ticks=range(len(stock_names)),
    labels=stock_names,
    rotation=90,
    fontsize=8
)

plt.yticks(
    ticks=range(len(stock_names)),
    labels=stock_names,
    fontsize=8
)

plt.title(
    "Dynamic Time Warping Distance Between Normalised Stock Prices"
)

plt.xlabel("Stock")
plt.ylabel("Stock")

plt.tight_layout()

plt.savefig(
    "results/figures/dtw_distance_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# --------------------------------------------------
# 10. Create hierarchical dendrogram from DTW matrix
# --------------------------------------------------

# Convert the square DTW matrix into condensed form.
# checks=False avoids minor floating-point symmetry issues.
condensed_dtw = squareform(
    dtw_distances.to_numpy(),
    checks=False
)

# Average linkage accepts a precomputed distance matrix.
linked = linkage(
    condensed_dtw,
    method="average"
)

plt.figure(figsize=(16, 8))

dendrogram(
    linked,
    labels=stock_names,
    leaf_rotation=90,
    leaf_font_size=8
)

plt.title(
    "Hierarchical Dendrogram Using Dynamic Time Warping"
)

plt.xlabel("Stock")
plt.ylabel("DTW Distance")

plt.tight_layout()

plt.savefig(
    "results/figures/dtw_distance_dendrogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()