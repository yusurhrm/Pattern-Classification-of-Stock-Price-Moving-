import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ==========================================================
# Directories
# ==========================================================

DATA_PATH = "data/raw/ftse100_40_companies.csv"

FIGURES_DIR = "figures"
RESULTS_DIR = "results"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================================
# Load data
# ==========================================================

data = pd.read_csv(
    DATA_PATH,
    header=[0, 1],
    index_col=0,
    parse_dates=True
)

prices = data.xs(
    "Adj Close",
    level="Price",
    axis=1
)

prices = prices.ffill().bfill()

# ==========================================================
# Normalise
# ==========================================================

normalised_prices = prices / prices.iloc[0]

# ==========================================================
# K-Means
# ==========================================================

X = normalised_prices.T

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

print(clusters)

clusters.to_csv(
    os.path.join(
        RESULTS_DIR,
        "kmeans_final_clusters.csv"
    ),
    index=False
)

# ==========================================================
# PCA
# ==========================================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

plot_df = pd.DataFrame({
    "PC1": X_pca[:,0],
    "PC2": X_pca[:,1],
    "Cluster": labels + 1,
    "Ticker": X.index
})

plt.figure(figsize=(10,8))

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
        fontsize=8
    )

plt.title("K-Means Clusters (k=3)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "kmeans_pca.png"
    ),
    dpi=300
)

plt.show()

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

plt.figure(figsize=(12,6))

for column in cluster_profiles.columns:

    plt.plot(
        cluster_profiles.index,
        cluster_profiles[column],
        label=column,
        linewidth=2
    )

plt.title("Average Normalised Price Movement by Cluster")

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
    dpi=300
)

plt.show()

cluster_profiles.to_csv(
    os.path.join(
        RESULTS_DIR,
        "cluster_average_patterns.csv"
    )
)

# ==========================================================
# Individual stock trajectories within each cluster
# ==========================================================

for cluster in sorted(clusters["Cluster"].unique()):

    tickers = clusters.loc[
        clusters["Cluster"] == cluster,
        "Ticker"
    ].tolist()

    plt.figure(figsize=(12, 7))

    # Plot every stock in the cluster
    for ticker in tickers:
        plt.plot(
            normalised_prices.index,
            normalised_prices[ticker],
            linewidth=1,
            alpha=0.45
        )

    # Plot the cluster mean more prominently
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
        f"Normalised Price Trajectories for Cluster {cluster}"
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