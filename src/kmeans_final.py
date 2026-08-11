import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ==========================================================
# Directories
# ==========================================================

DATA_PATH = "data/processed/normalised_prices.csv"

FIGURES_DIR = "results/figures"
RESULTS_DIR = "results"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================================
# Load cleaned normalised data
# ==========================================================

normalised_prices = pd.read_csv(
    DATA_PATH,
    index_col=0,
    parse_dates=True
)

normalised_prices = normalised_prices.sort_index()
normalised_prices = normalised_prices.sort_index(axis=1)

print("=" * 60)
print("FINAL K-MEANS CLUSTERING")
print("=" * 60)

print(f"\nDataset shape: {normalised_prices.shape}")
print(f"Companies: {normalised_prices.shape[1]}")
print(f"Trading days: {normalised_prices.shape[0]}")

# ==========================================================
# Prepare data
# ==========================================================

X = normalised_prices.T

# ==========================================================
# Final K-Means (k = 3)
# ==========================================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=20
)

labels = kmeans.fit_predict(X)

clusters = pd.DataFrame({
    "Ticker": X.index,
    "Cluster": labels + 1
})

clusters = clusters.sort_values(
    ["Cluster", "Ticker"]
)

print("\nFinal cluster assignments:")
print(clusters)

clusters.to_csv(
    os.path.join(
        RESULTS_DIR,
        "kmeans_final_clusters.csv"
    ),
    index=False
)

# ==========================================================
# PCA visualisation
# ==========================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

plot_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Cluster": labels + 1,
    "Ticker": X.index
})

plt.figure(figsize=(10, 8))

for cluster in sorted(plot_df.Cluster.unique()):

    subset = plot_df[
        plot_df.Cluster == cluster
    ]

    plt.scatter(
        subset.PC1,
        subset.PC2,
        s=90,
        label=f"Cluster {cluster}"
    )

for _, row in plot_df.iterrows():

    plt.text(
        row.PC1,
        row.PC2,
        row.Ticker,
        fontsize=7
    )

plt.title("K-Means Clusters (k = 3, 98 Companies)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "kmeans_pca.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

# ==========================================================
# Average movement pattern
# ==========================================================

cluster_profiles = pd.DataFrame(
    index=normalised_prices.index
)

for cluster in sorted(clusters.Cluster.unique()):

    tickers = clusters.loc[
        clusters.Cluster == cluster,
        "Ticker"
    ]

    cluster_profiles[
        f"Cluster {cluster}"
    ] = normalised_prices[
        tickers
    ].mean(axis=1)

plt.figure(figsize=(12, 6))

for column in cluster_profiles.columns:

    plt.plot(
        cluster_profiles.index,
        cluster_profiles[column],
        linewidth=2,
        label=column
    )

plt.title(
    "Average Normalised Price Movement by Cluster (98 Companies)"
)

plt.xlabel("Year")
plt.ylabel("Normalised Price")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "cluster_average_patterns.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

cluster_profiles.to_csv(
    os.path.join(
        RESULTS_DIR,
        "cluster_average_patterns.csv"
    )
)

# ==========================================================
# Individual stock trajectories within each cluster
# ==========================================================

for cluster in sorted(clusters.Cluster.unique()):

    tickers = clusters.loc[
        clusters.Cluster == cluster,
        "Ticker"
    ].tolist()

    plt.figure(figsize=(12, 7))

    for ticker in tickers:

        plt.plot(
            normalised_prices.index,
            normalised_prices[ticker],
            linewidth=1,
            alpha=0.40
        )

    cluster_mean = normalised_prices[
        tickers
    ].mean(axis=1)

    plt.plot(
        normalised_prices.index,
        cluster_mean,
        linewidth=3,
        label="Cluster mean"
    )

    plt.title(
        f"Normalised Price Trajectories - Cluster {cluster}"
    )

    plt.xlabel("Year")
    plt.ylabel("Normalised Price")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            f"cluster_{cluster}_individual_trajectories.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

print("\nFinished.")