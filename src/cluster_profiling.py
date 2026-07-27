import os

import numpy as np
import pandas as pd


# ============================================================
# File paths
# ============================================================

DATA_PATH = "data/raw/ftse100_40_companies.csv"
CLUSTERS_PATH = "results/kmeans_final_clusters.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Cluster names
# ============================================================

CLUSTER_NAMES = {
    1: "Moderate Sustained Growth",
    2: "Stable or Low Growth",
    3: "Exceptional Growth",
}


# ============================================================
# Sector classifications
# ============================================================

SECTOR_MAP = {
    "AAL.L": "Mining",
    "ADM.L": "Financials",
    "AUTO.L": "Technology and Media",
    "AV.L": "Financials",
    "AZN.L": "Healthcare",
    "BARC.L": "Financials",
    "BEZ.L": "Financials",
    "BME.L": "Retail",
    "BP.L": "Energy",
    "CCH.L": "Consumer Goods",
    "CNA.L": "Utilities",
    "CTEC.L": "Healthcare",
    "DGE.L": "Consumer Goods",
    "GLEN.L": "Mining",
    "GSK.L": "Healthcare",
    "HIK.L": "Healthcare",
    "HSBA.L": "Financials",
    "IMB.L": "Consumer Goods",
    "JD.L": "Retail",
    "LGEN.L": "Financials",
    "LLOY.L": "Financials",
    "MKS.L": "Retail",
    "NG.L": "Utilities",
    "NWG.L": "Financials",
    "PRU.L": "Financials",
    "REL.L": "Technology and Media",
    "RIO.L": "Mining",
    "RKT.L": "Consumer Goods",
    "RR.L": "Industrials",
    "SBRY.L": "Retail",
    "SHEL.L": "Energy",
    "SMIN.L": "Industrials",
    "SN.L": "Healthcare",
    "SSE.L": "Utilities",
    "STAN.L": "Financials",
    "SVT.L": "Utilities",
    "TSCO.L": "Retail",
    "ULVR.L": "Consumer Goods",
    "UU.L": "Utilities",
    "WEIR.L": "Industrials",
}


# ============================================================
# Cluster interpretations
# ============================================================

INTERPRETATION_MAP = {
    1: "Moderate sustained growth",
    2: "Stable or low growth",
    3: "Exceptional growth",
}


# ============================================================
# Load the stock-price dataset
# ============================================================

data = pd.read_csv(
    DATA_PATH,
    header=[0, 1],
    index_col=0,
    parse_dates=True,
)

print("\nOriginal dataset shape:")
print(data.shape)

print("\nColumn levels:")
print(data.columns.names)


# ============================================================
# Extract adjusted closing prices
# ============================================================

prices = data.xs(
    "Adj Close",
    level="Price",
    axis=1,
)

prices = prices.apply(
    pd.to_numeric,
    errors="coerce",
)

prices = prices.sort_index()

# Remove duplicated dates, if any
prices = prices.loc[
    ~prices.index.duplicated(keep="first")
]

print("\nAdjusted closing-price shape:")
print(prices.shape)


# ============================================================
# Handle missing values
# ============================================================

missing_before = prices.isna().sum()
missing_before = missing_before[
    missing_before > 0
].sort_values(ascending=False)

print("\nMissing values before filling:")

if missing_before.empty:
    print("No missing values found.")
else:
    print(missing_before)

# Appropriate for isolated missing trading observations
prices = prices.ffill().bfill()

prices = prices.replace(
    [np.inf, -np.inf],
    np.nan,
)

if prices.isna().any().any():
    unresolved = prices.isna().sum()
    unresolved = unresolved[
        unresolved > 0
    ]

    raise ValueError(
        "Unresolved missing values remain:\n"
        f"{unresolved}"
    )


# ============================================================
# Normalise the stock prices
# ============================================================

# Each company begins at 1.0
normalised_prices = prices.div(
    prices.iloc[0],
    axis="columns",
)

normalised_prices = normalised_prices.replace(
    [np.inf, -np.inf],
    np.nan,
)

if normalised_prices.isna().any().any():
    raise ValueError(
        "Invalid values were produced during normalisation."
    )


# ============================================================
# Load K-Means cluster assignments
# ============================================================

clusters = pd.read_csv(
    CLUSTERS_PATH
)

required_columns = {
    "Ticker",
    "Cluster",
}

if not required_columns.issubset(clusters.columns):
    raise ValueError(
        "The cluster file must contain "
        "'Ticker' and 'Cluster' columns."
    )

clusters["Ticker"] = (
    clusters["Ticker"]
    .astype(str)
    .str.strip()
)

clusters["Cluster"] = pd.to_numeric(
    clusters["Cluster"],
    errors="raise",
).astype(int)

clusters["Cluster Name"] = clusters[
    "Cluster"
].map(CLUSTER_NAMES)

clusters["Sector"] = clusters[
    "Ticker"
].map(SECTOR_MAP)


# ============================================================
# Validate ticker and sector information
# ============================================================

missing_price_tickers = sorted(
    set(clusters["Ticker"])
    - set(normalised_prices.columns)
)

if missing_price_tickers:
    raise ValueError(
        "These cluster tickers are missing from the price data: "
        f"{missing_price_tickers}"
    )

missing_cluster_names = clusters.loc[
    clusters["Cluster Name"].isna(),
    "Cluster",
].unique()

if len(missing_cluster_names) > 0:
    raise ValueError(
        "No cluster name has been defined for: "
        f"{missing_cluster_names.tolist()}"
    )

missing_sector_tickers = clusters.loc[
    clusters["Sector"].isna(),
    "Ticker",
].tolist()

if missing_sector_tickers:
    raise ValueError(
        "No sector has been defined for: "
        f"{missing_sector_tickers}"
    )


# ============================================================
# Calculate company-level financial statistics
# ============================================================

TRADING_DAYS_PER_YEAR = 252

daily_returns = prices.pct_change(
    fill_method=None
)

study_years = (
    prices.index[-1] - prices.index[0]
).days / 365.25

company_statistics = []

for ticker in clusters["Ticker"]:

    ticker_prices = prices[ticker]
    ticker_normalised = normalised_prices[ticker]
    ticker_returns = daily_returns[ticker].dropna()

    initial_price = ticker_prices.iloc[0]
    final_price = ticker_prices.iloc[-1]

    final_normalised_value = (
        ticker_normalised.iloc[-1]
    )

    total_return = (
        final_price / initial_price
    ) - 1

    if study_years > 0 and initial_price > 0:
        annualised_return = (
            final_price / initial_price
        ) ** (1 / study_years) - 1
    else:
        annualised_return = np.nan

    annualised_volatility = (
        ticker_returns.std(ddof=1)
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    running_maximum = ticker_normalised.cummax()

    drawdown = (
        ticker_normalised / running_maximum
    ) - 1

    maximum_drawdown = drawdown.min()

    company_statistics.append(
        {
            "Ticker": ticker,
            "Initial Adjusted Close": initial_price,
            "Final Adjusted Close": final_price,
            "Final Normalised Value": final_normalised_value,
            "Total Return": total_return,
            "Annualised Return": annualised_return,
            "Annualised Volatility": annualised_volatility,
            "Maximum Drawdown": maximum_drawdown,
        }
    )

company_statistics = pd.DataFrame(
    company_statistics
)


# ============================================================
# Combine company, sector and cluster information
# ============================================================

company_profiles = clusters.merge(
    company_statistics,
    on="Ticker",
    how="left",
)

company_profiles = company_profiles.sort_values(
    by=[
        "Cluster",
        "Ticker",
    ]
).reset_index(drop=True)

company_profiles.to_csv(
    os.path.join(
        RESULTS_DIR,
        "company_cluster_profiles.csv",
    ),
    index=False,
)


# ============================================================
# Calculate cluster-level statistics
# ============================================================

cluster_statistics = (
    company_profiles
    .groupby(
        [
            "Cluster",
            "Cluster Name",
        ],
        as_index=False,
    )
    .agg(
        Companies=(
            "Ticker",
            "count",
        ),
        Mean_Final_Value=(
            "Final Normalised Value",
            "mean",
        ),
        Median_Final_Value=(
            "Final Normalised Value",
            "median",
        ),
        Mean_Total_Return=(
            "Total Return",
            "mean",
        ),
        Mean_Annualised_Return=(
            "Annualised Return",
            "mean",
        ),
        Mean_Annualised_Volatility=(
            "Annualised Volatility",
            "mean",
        ),
        Mean_Maximum_Drawdown=(
            "Maximum Drawdown",
            "mean",
        ),
    )
)


# ============================================================
# Calculate sector composition
# ============================================================

sector_counts = (
    company_profiles
    .groupby(
        [
            "Cluster",
            "Sector",
        ]
    )
    .size()
    .reset_index(
        name="Company Count"
    )
)

sector_counts = sector_counts.sort_values(
    by=[
        "Cluster",
        "Company Count",
        "Sector",
    ],
    ascending=[
        True,
        False,
        True,
    ],
)


# ============================================================
# Identify dominant sector in each cluster
# ============================================================

dominant_sectors = (
    sector_counts
    .drop_duplicates(
        subset="Cluster",
        keep="first",
    )
    .rename(
        columns={
            "Sector": "Dominant Sector",
            "Company Count": "Dominant Sector Count",
        }
    )
)

cluster_statistics = cluster_statistics.merge(
    dominant_sectors[
        [
            "Cluster",
            "Dominant Sector",
            "Dominant Sector Count",
        ]
    ],
    on="Cluster",
    how="left",
)

cluster_statistics["Interpretation"] = (
    cluster_statistics["Cluster"]
    .map(INTERPRETATION_MAP)
)


# ============================================================
# TABLE 1: Concise cluster summary
# ============================================================

cluster_summary_table = cluster_statistics[
    [
        "Cluster",
        "Cluster Name",
        "Companies",
        "Mean_Final_Value",
        "Dominant Sector",
        "Interpretation",
    ]
].copy()

cluster_summary_table = cluster_summary_table.rename(
    columns={
        "Mean_Final_Value":
            "Mean Final Normalised Value",
    }
)

cluster_summary_table[
    "Mean Final Normalised Value"
] = (
    cluster_summary_table[
        "Mean Final Normalised Value"
    ]
    .round(2)
)

cluster_summary_table.to_csv(
    os.path.join(
        RESULTS_DIR,
        "table_1_cluster_summary.csv",
    ),
    index=False,
)


# ============================================================
# TABLE 2: Financial characteristics
# ============================================================

financial_characteristics_table = (
    cluster_statistics[
        [
            "Cluster",
            "Mean_Annualised_Return",
            "Mean_Annualised_Volatility",
            "Mean_Maximum_Drawdown",
        ]
    ]
    .copy()
)

financial_characteristics_table[
    "Mean Annualised Return (%)"
] = (
    financial_characteristics_table[
        "Mean_Annualised_Return"
    ] * 100
)

financial_characteristics_table[
    "Mean Annualised Volatility (%)"
] = (
    financial_characteristics_table[
        "Mean_Annualised_Volatility"
    ] * 100
)

financial_characteristics_table[
    "Mean Maximum Drawdown (%)"
] = (
    financial_characteristics_table[
        "Mean_Maximum_Drawdown"
    ] * 100
)

financial_characteristics_table = (
    financial_characteristics_table.drop(
        columns=[
            "Mean_Annualised_Return",
            "Mean_Annualised_Volatility",
            "Mean_Maximum_Drawdown",
        ]
    )
)

percentage_columns = [
    "Mean Annualised Return (%)",
    "Mean Annualised Volatility (%)",
    "Mean Maximum Drawdown (%)",
]

financial_characteristics_table[
    percentage_columns
] = (
    financial_characteristics_table[
        percentage_columns
    ]
    .round(2)
)

financial_characteristics_table.to_csv(
    os.path.join(
        RESULTS_DIR,
        "table_2_financial_characteristics.csv",
    ),
    index=False,
)


# ============================================================
# TABLE 3: Sector composition
# ============================================================

sector_breakdowns = []

for cluster_number in sorted(
    company_profiles["Cluster"].unique()
):

    current_cluster_sectors = sector_counts.loc[
        sector_counts["Cluster"]
        == cluster_number
    ]

    breakdown = ", ".join(
        f"{row['Sector']} ({row['Company Count']})"
        for _, row in current_cluster_sectors.iterrows()
    )

    sector_breakdowns.append(
        {
            "Cluster": cluster_number,
            "Sector Breakdown": breakdown,
        }
    )

sector_composition_table = pd.DataFrame(
    sector_breakdowns
)

sector_composition_table.to_csv(
    os.path.join(
        RESULTS_DIR,
        "table_3_sector_composition.csv",
    ),
    index=False,
)


# ============================================================
# Optional table: companies in each cluster
# ============================================================

company_lists = (
    company_profiles
    .groupby("Cluster")["Ticker"]
    .apply(
        lambda tickers: ", ".join(
            sorted(tickers)
        )
    )
    .reset_index(
        name="Companies Included"
    )
)

company_lists.to_csv(
    os.path.join(
        RESULTS_DIR,
        "cluster_company_lists.csv",
    ),
    index=False,
)


# ============================================================
# Optional sector pivot table
# ============================================================

sector_pivot = pd.pivot_table(
    company_profiles,
    index="Cluster",
    columns="Sector",
    values="Ticker",
    aggfunc="count",
    fill_value=0,
)

sector_pivot.to_csv(
    os.path.join(
        RESULTS_DIR,
        "cluster_sector_pivot.csv",
    )
)


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 70)
print("TABLE 1: CLUSTER SUMMARY")
print("=" * 70)

print(
    cluster_summary_table.to_string(
        index=False
    )
)

print("\n" + "=" * 70)
print("TABLE 2: FINANCIAL CHARACTERISTICS")
print("=" * 70)

print(
    financial_characteristics_table.to_string(
        index=False
    )
)

print("\n" + "=" * 70)
print("TABLE 3: SECTOR COMPOSITION")
print("=" * 70)

print(
    sector_composition_table.to_string(
        index=False
    )
)

print("\n" + "=" * 70)
print("COMPANIES IN EACH CLUSTER")
print("=" * 70)

print(
    company_lists.to_string(
        index=False
    )
)


# ============================================================
# Finished
# ============================================================

print("\nCluster profiling completed successfully.")

print("\nSaved files:")

saved_files = [
    "company_cluster_profiles.csv",
    "table_1_cluster_summary.csv",
    "table_2_financial_characteristics.csv",
    "table_3_sector_composition.csv",
    "cluster_company_lists.csv",
    "cluster_sector_pivot.csv",
]

for filename in saved_files:
    print(
        os.path.join(
            RESULTS_DIR,
            filename,
        )
    )