import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, dendrogram


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

print("\nOriginal column structure:")
print(data.columns)


# --------------------------------------------------
# 3. Extract adjusted closing prices
# --------------------------------------------------

# The CSV columns are organised as:
# first level = ticker
# second level = price type
prices = data.xs(
    "Adj Close",
    level="Price",
    axis=1
)

# Remove columns containing no valid observations
prices = prices.dropna(axis=1, how="all")

# Fill occasional missing values
prices = prices.ffill().bfill()

print("\nAdjusted closing prices:")
print(prices.head())

print("\nPrice data shape:")
print(prices.shape)


# --------------------------------------------------
# 4. Normalise each stock by its initial value
# --------------------------------------------------

normalised_prices = prices / prices.iloc[0]

normalised_prices.to_csv(
    "data/processed/normalised_stock_prices.csv"
)

print("\nNormalised prices:")
print(normalised_prices.head())


# --------------------------------------------------
# 5. Prepare the data for distance calculation
# --------------------------------------------------

# pairwise_distances expects each row to be one observation.
# Therefore, transpose the DataFrame so:
# rows = stocks
# columns = trading dates
stock_series = normalised_prices.T


# --------------------------------------------------
# 6. Calculate Euclidean distances
# --------------------------------------------------

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

print("\nEuclidean Distance Matrix:")
print(euclidean_distances.round(2))

euclidean_distances.to_csv(
    "data/processed/euclidean_distance_matrix.csv"
)


# --------------------------------------------------
# 7. Convert the matrix into unique stock pairs
# --------------------------------------------------

# Remove index/column names to avoid duplicate column names
distance_matrix = euclidean_distances.rename_axis(index=None, columns=None)

# Convert to long format
pairs = distance_matrix.stack().reset_index()
pairs.columns = ["Stock 1", "Stock 2", "Distance"]

# Remove comparisons where a stock is compared with itself
pairs = pairs[
    pairs["Stock 1"] != pairs["Stock 2"]
].copy()

# Create a sorted pair so A-B and B-A are treated as duplicates
pairs["Pair"] = pairs.apply(
    lambda row: tuple(
        sorted([row["Stock 1"], row["Stock 2"]])
    ),
    axis=1
)

# Keep only one copy of each stock pair
pairs = pairs.drop_duplicates(
    subset="Pair"
).drop(
    columns="Pair"
)


# --------------------------------------------------
# 8. Find closest and furthest stock pairs
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
print(closest_pairs.round(2).to_string(index=False))

print("\n10 Most Dissimilar Stock Pairs:")
print(furthest_pairs.round(2).to_string(index=False))

closest_pairs.to_csv(
    "data/processed/closest_euclidean_pairs.csv",
    index=False
)

furthest_pairs.to_csv(
    "data/processed/furthest_euclidean_pairs.csv",
    index=False
)


# --------------------------------------------------
# 9. Plot the Euclidean distance heatmap
# --------------------------------------------------

plt.figure(figsize=(14, 12))

plt.imshow(
    euclidean_distances,
    aspect="auto"
)

plt.colorbar(
    label="Euclidean Distance"
)

plt.xticks(
    ticks=range(len(euclidean_distances.columns)),
    labels=euclidean_distances.columns,
    rotation=90,
    fontsize=8
)

plt.yticks(
    ticks=range(len(euclidean_distances.index)),
    labels=euclidean_distances.index,
    fontsize=8
)

plt.title(
    "Euclidean Distance Between Normalised Stock Prices"
)

plt.xlabel("Stock")
plt.ylabel("Stock")

plt.tight_layout()

plt.savefig(
    "results/figures/euclidean_distance_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# --------------------------------------------------
# 10. Hierarchical clustering dendrogram
# --------------------------------------------------

# Perform hierarchical clustering using Ward linkage
linked = linkage(
    stock_series,
    method="ward",
    metric="euclidean"
)

plt.figure(figsize=(16, 8))

dendrogram(
    linked,
    labels=stock_series.index,
    leaf_rotation=90,
    leaf_font_size=8,
    color_threshold=35
)

plt.title("Hierarchical Dendrogram of FTSE 100 Stocks")
plt.xlabel("Stock")
plt.ylabel("Euclidean Distance")

plt.tight_layout()

plt.savefig(
    "results/figures/hierarchical_dendrogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()