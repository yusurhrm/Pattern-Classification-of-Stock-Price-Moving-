import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import pairwise_distances


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

# CSV structure:
# level 0 = ticker
# level 1 = price type
prices = data.xs(
    "Adj Close",
    level="Price",
    axis=1
)

# Remove companies with no valid observations
prices = prices.dropna(axis=1, how="all")

# Fill occasional missing observations
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
# 5. Prepare the data for distance calculations
# --------------------------------------------------

# Rows must represent stocks and columns must represent dates
stock_series = normalised_prices.T


# --------------------------------------------------
# 6. Calculate correlation distances
# --------------------------------------------------

correlation_array = pairwise_distances(
    stock_series,
    metric="correlation"
)

correlation_distances = pd.DataFrame(
    correlation_array,
    index=stock_series.index,
    columns=stock_series.index
)

print("\nCorrelation Distance Matrix:")
print(correlation_distances.round(3))

correlation_distances.to_csv(
    "data/processed/correlation_distance_matrix.csv"
)


# --------------------------------------------------
# 7. Convert the matrix into unique stock pairs
# --------------------------------------------------

# Remove axis names to avoid duplicate-column errors
distance_matrix = correlation_distances.rename_axis(
    index=None,
    columns=None
)

pairs = distance_matrix.stack().reset_index()
pairs.columns = ["Stock 1", "Stock 2", "Distance"]

# Remove self-comparisons
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

pairs = pairs.drop_duplicates(
    subset="Pair"
).drop(
    columns="Pair"
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
    "data/processed/closest_correlation_pairs.csv",
    index=False
)

furthest_pairs.to_csv(
    "data/processed/furthest_correlation_pairs.csv",
    index=False
)


# --------------------------------------------------
# 9. Plot the correlation distance heatmap
# --------------------------------------------------

plt.figure(figsize=(14, 12))

plt.imshow(
    correlation_distances,
    aspect="auto"
)

plt.colorbar(
    label="Correlation Distance"
)

plt.xticks(
    ticks=range(len(correlation_distances.columns)),
    labels=correlation_distances.columns,
    rotation=90,
    fontsize=8
)

plt.yticks(
    ticks=range(len(correlation_distances.index)),
    labels=correlation_distances.index,
    fontsize=8
)

plt.title(
    "Correlation Distance Between Normalised Stock Prices"
)

plt.xlabel("Stock")
plt.ylabel("Stock")

plt.tight_layout()

plt.savefig(
    "results/figures/correlation_distance_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# --------------------------------------------------
# 10. Create hierarchical dendrogram
# --------------------------------------------------

# Ward linkage cannot be used with correlation distance.
# Average linkage is therefore used instead.
linked = linkage(
    stock_series,
    method="average",
    metric="correlation"
)

plt.figure(figsize=(16, 8))

dendrogram(
    linked,
    labels=stock_series.index,
    leaf_rotation=90,
    leaf_font_size=8
)

plt.title(
    "Hierarchical Dendrogram Using Correlation Distance"
)

plt.xlabel("Stock")
plt.ylabel("Correlation Distance")

plt.tight_layout()

plt.savefig(
    "results/figures/correlation_distance_dendrogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()