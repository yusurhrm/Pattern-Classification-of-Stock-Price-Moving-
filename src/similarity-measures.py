import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

# Create output folder
os.makedirs("results/figures", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Load stock data
data = pd.read_csv(
    "data/raw/ftse100_40_companies.csv",
    header=[0, 1],
    index_col=0,
    parse_dates=True
)

# Extract adjusted closing prices
prices = data["Adj Close"]

# Remove stocks with no valid observations
prices = prices.dropna(axis=1, how="all")

# Fill occasional missing values
prices = prices.ffill().bfill()

# Normalise each stock by its initial value
normalised_prices = prices / prices.iloc[0]

# Transpose so each row represents one stock
stock_series = normalised_prices.T

# Calculate pairwise Euclidean distances
euclidean_array = pairwise_distances(
    stock_series,
    metric="euclidean"
)

# Convert the result into a labelled DataFrame
euclidean_distances = pd.DataFrame(
    euclidean_array,
    index=stock_series.index,
    columns=stock_series.index
)

print("\nEuclidean Distance Matrix:")
print(euclidean_distances.round(2))

# Save the full matrix
euclidean_distances.to_csv(
    "data/processed/euclidean_distance_matrix.csv"
)

# Plot the distance matrix
plt.figure(figsize=(14, 12))
plt.imshow(euclidean_distances, aspect="auto")
plt.colorbar(label="Euclidean Distance")

plt.xticks(
    range(len(euclidean_distances.columns)),
    euclidean_distances.columns,
    rotation=90,
    fontsize=8
)

plt.yticks(
    range(len(euclidean_distances.index)),
    euclidean_distances.index,
    fontsize=8
)

plt.title("Euclidean Distance Between Normalised Stock Prices")
plt.xlabel("Stock")
plt.ylabel("Stock")
plt.tight_layout()

plt.savefig(
    "results/figures/euclidean_distance_heatmap.png",
    dpi=300
)

plt.show()