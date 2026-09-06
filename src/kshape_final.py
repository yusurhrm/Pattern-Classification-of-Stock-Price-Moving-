import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import cdist_dtw


DATA_PATH = "data/processed/normalised_prices.csv"

FIGURES_DIR = "results/figures/kshape"
RESULTS_DIR = "results"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


normalised_prices = pd.read_csv(
    DATA_PATH,
    index_col=0,
    parse_dates=True
)

normalised_prices = normalised_prices.sort_index()
normalised_prices = normalised_prices.sort_index(axis=1)

print("=" * 60)
print("K-SHAPE CLUSTER INSPECTION")
print("=" * 60)

print(f"\nDataset shape: {normalised_prices.shape}")
print(f"Companies: {normalised_prices.shape[1]}")
print(f"Trading days: {normalised_prices.shape[0]}")


X = normalised_prices.T.values

# K-Shape expects:
# (n_samples, n_timesteps, n_features)
X = X[:, :, np.newaxis]

# K-Shape uses shape-based similarity, so each time series
# should be z-normalised.
X_scaled = TimeSeriesScalerMeanVariance(
    mu=0.0,
    std=1.0
).fit_transform(X)

tickers = normalised_prices.columns.to_numpy()


K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10]

summary_rows = []

for k in K_VALUES:

    print("\n" + "=" * 60)
    print(f"K-SHAPE: k = {k}")
    print("=" * 60)

    model = KShape(
        n_clusters=k,
        n_init=10,
        random_state=42,
        verbose=False
    )

    labels = model.fit_predict(X_scaled)


    assignments = pd.DataFrame({
        "Ticker": tickers,
        "Cluster": labels + 1
    })

    assignments = assignments.sort_values(
        ["Cluster", "Ticker"]
    )

    assignments.to_csv(
        os.path.join(
            RESULTS_DIR,
            f"kshape_k{k}_clusters.csv"
        ),
        index=False
    )


    cluster_sizes = (
        assignments
        .groupby("Cluster")
        .size()
        .sort_index()
    )

    smallest_cluster = cluster_sizes.min()
    largest_cluster = cluster_sizes.max()
    singleton_clusters = int(
        (cluster_sizes == 1).sum()
    )

    print("\nCluster sizes:")
    print(cluster_sizes)

    print(
        f"\nSmallest cluster: {smallest_cluster}"
    )
    print(
        f"Largest cluster: {largest_cluster}"
    )
    print(
        f"Singleton clusters: {singleton_clusters}"
    )


    cluster_profiles = pd.DataFrame(
        index=normalised_prices.index
    )

    for cluster in sorted(
        assignments.Cluster.unique()
    ):

        members = assignments.loc[
            assignments.Cluster == cluster,
            "Ticker"
        ].tolist()

        cluster_profiles[
            f"Cluster {cluster}"
        ] = normalised_prices[
            members
        ].mean(axis=1)


    plt.figure(figsize=(12, 6))

    for column in cluster_profiles.columns:

        plt.plot(
            cluster_profiles.index,
            cluster_profiles[column],
            linewidth=2,
            label=column
        )

    plt.axhline(
        y=1,
        linestyle="--",
        linewidth=1,
        alpha=0.6
    )

    plt.title(
        f"Average Normalised Price Movement - K-Shape (k = {k})"
    )

    plt.xlabel("Date")
    plt.ylabel("Mean Normalised Price")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            f"kshape_k{k}_average_patterns.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


    for cluster in sorted(
        assignments.Cluster.unique()
    ):

        members = assignments.loc[
            assignments.Cluster == cluster,
            "Ticker"
        ].tolist()

        plt.figure(figsize=(12, 7))

        for ticker in members:

            plt.plot(
                normalised_prices.index,
                normalised_prices[ticker],
                linewidth=1,
                alpha=0.30
            )

        cluster_mean = normalised_prices[
            members
        ].mean(axis=1)

        plt.plot(
            normalised_prices.index,
            cluster_mean,
            linewidth=3,
            label="Cluster mean"
        )

        plt.axhline(
            y=1,
            linestyle="--",
            linewidth=1,
            alpha=0.6
        )

        plt.title(
            f"K-Shape k={k}, Cluster {cluster} "
            f"(n={len(members)})"
        )

        plt.xlabel("Date")
        plt.ylabel("Normalised Price")

        plt.grid(alpha=0.3)
        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                FIGURES_DIR,
                f"kshape_k{k}_cluster_{cluster}.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()


    cluster_profiles.to_csv(
        os.path.join(
            RESULTS_DIR,
            f"kshape_k{k}_average_patterns.csv"
        )
    )


    summary_rows.append({
        "k": k,
        "Smallest Cluster": smallest_cluster,
        "Largest Cluster": largest_cluster,
        "Singleton Clusters": singleton_clusters,
        "Inertia": model.inertia_
    })


summary = pd.DataFrame(summary_rows)

summary.to_csv(
    os.path.join(
        RESULTS_DIR,
        "kshape_cluster_inspection_summary.csv"
    ),
    index=False
)

print("\n" + "=" * 60)
print("K-SHAPE SUMMARY")
print("=" * 60)

print(summary.to_string(index=False))

print("\nFinished.")