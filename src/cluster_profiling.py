import os

import numpy as np
import pandas as pd


NORMALISED_DATA_PATH = "data/processed/normalised_prices.csv"
CLUSTERS_PATH = "results/kmeans_final_clusters.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)



CLUSTER_NAMES = {
    1: "Stable or Modest Growth",
    2: "Exceptional Growth",
    3: "Strong Sustained Growth",
}


INTERPRETATION_MAP = {
    1: (
        "Relatively stable trajectories with limited or "
        "modest long-term appreciation"
    ),
    2: (
        "Exceptional proportional growth, represented by "
        "BGEO.L and RR.L"
    ),
    3: (
        "Stronger sustained long-term appreciation than "
        "the majority cluster"
    ),
}


SECTOR_MAP = {
    "AAF.L": "Telecommunications",
    "AAL.L": "Mining",
    "ABDN.L": "Financial Services",
    "ABF.L": "Consumer Goods",
    "ADM.L": "Insurance",
    "ALW.L": "Investment Trust",
    "ANTO.L": "Mining",
    "AUTO.L": "Technology and Media",
    "AV.L": "Insurance",
    "AZN.L": "Healthcare",

    "BA.L": "Aerospace and Defence",
    "BAB.L": "Aerospace and Defence",
    "BARC.L": "Banking",
    "BATS.L": "Consumer Goods",
    "BBOX.L": "Property",
    "BEZ.L": "Insurance",
    "BGEO.L": "Banking",
    "BKG.L": "Housebuilding",
    "BLND.L": "Property",
    "BNZL.L": "Distribution",
    "BP.L": "Energy",
    "BRBY.L": "Retail",
    "BT-A.L": "Telecommunications",
    "BTRW.L": "Housebuilding",

    "CCEP.L": "Consumer Goods",
    "CCH.L": "Consumer Goods",
    "CNA.L": "Utilities",
    "CPG.L": "Consumer Services",
    "CRDA.L": "Chemicals",
    "CTEC.L": "Healthcare",

    "DCC.L": "Distribution",
    "DGE.L": "Consumer Goods",
    "DPLM.L": "Industrials",

    "EDV.L": "Mining",
    "ENT.L": "Consumer Services",
    "EXPN.L": "Technology and Media",

    "FCIT.L": "Investment Trust",
    "FRES.L": "Mining",

    "GAW.L": "Consumer Goods",
    "GLEN.L": "Mining",
    "GSK.L": "Healthcare",

    "HLMA.L": "Industrials",
    "HSBA.L": "Banking",
    "HSX.L": "Insurance",
    "HWDN.L": "Retail",

    "IAG.L": "Airlines",
    "ICG.L": "Financial Services",
    "IGG.L": "Financial Services",
    "IHG.L": "Travel and Leisure",
    "III.L": "Investment Trust",
    "IMB.L": "Consumer Goods",
    "IMI.L": "Industrials",
    "INF.L": "Technology and Media",
    "ITRK.L": "Industrials",

    "JD.L": "Retail",

    "KGF.L": "Retail",

    "LAND.L": "Property",
    "LGEN.L": "Insurance",
    "LLOY.L": "Banking",
    "LMP.L": "Property",
    "LSEG.L": "Financial Services",

    "MKS.L": "Retail",
    "MNDI.L": "Packaging",
    "MNG.L": "Financial Services",
    "MRO.L": "Industrials",

    "NG.L": "Utilities",
    "NWG.L": "Banking",
    "NXT.L": "Retail",

    "PCT.L": "Investment Trust",
    "PRU.L": "Insurance",
    "PSH.L": "Investment Trust",
    "PSN.L": "Housebuilding",
    "PSON.L": "Technology and Media",

    "REL.L": "Technology and Media",
    "RIO.L": "Mining",
    "RKT.L": "Consumer Goods",
    "RMV.L": "Technology and Media",
    "RR.L": "Aerospace and Defence",
    "RTO.L": "Consumer Services",

    "SBRY.L": "Retail",
    "SDR.L": "Financial Services",
    "SGE.L": "Technology and Media",
    "SGRO.L": "Property",
    "SHEL.L": "Energy",
    "SMIN.L": "Industrials",
    "SMT.L": "Investment Trust",
    "SN.L": "Healthcare",
    "SPX.L": "Industrials",
    "SSE.L": "Utilities",
    "STAN.L": "Banking",
    "STJ.L": "Financial Services",
    "SVT.L": "Utilities",

    "TSCO.L": "Retail",

    "ULVR.L": "Consumer Goods",
    "UU.L": "Utilities",

    "VOD.L": "Telecommunications",

    "WEIR.L": "Industrials",
    "WTB.L": "Travel and Leisure",
}


normalised_prices = pd.read_csv(
    NORMALISED_DATA_PATH,
    index_col=0,
    parse_dates=True,
)

normalised_prices = normalised_prices.sort_index()
normalised_prices = normalised_prices.sort_index(axis=1)

print("=" * 70)
print("CLUSTER PROFILING ANALYSIS")
print("=" * 70)

print(f"\nNormalised dataset shape: {normalised_prices.shape}")
print(
    f"Date range: {normalised_prices.index.min()} "
    f"to {normalised_prices.index.max()}"
)
print(f"Number of companies: {normalised_prices.shape[1]}")
print(f"Number of trading days: {normalised_prices.shape[0]}")


if normalised_prices.empty:
    raise ValueError(
        "The normalised-price dataset is empty."
    )

if normalised_prices.isna().any().any():
    companies_with_missing = normalised_prices.columns[
        normalised_prices.isna().any()
    ].tolist()

    raise ValueError(
        "Missing values were found for: "
        + ", ".join(companies_with_missing)
    )

if not np.isfinite(
    normalised_prices.to_numpy()
).all():
    raise ValueError(
        "The normalised dataset contains infinite "
        "or invalid values."
    )



prices = normalised_prices.copy()

daily_returns = prices.pct_change(
    fill_method=None
)


clusters = pd.read_csv(
    CLUSTERS_PATH
)

required_columns = {
    "Ticker",
    "Cluster",
}

if not required_columns.issubset(
    clusters.columns
):
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

clusters = clusters.sort_values(
    by=[
        "Cluster",
        "Ticker",
    ]
).reset_index(drop=True)

print("\nCluster assignment counts:")
print(
    clusters["Cluster"]
    .value_counts()
    .sort_index()
)


clusters["Cluster Name"] = (
    clusters["Cluster"]
    .map(CLUSTER_NAMES)
)

clusters["Sector"] = (
    clusters["Ticker"]
    .map(SECTOR_MAP)
)


missing_price_tickers = sorted(
    set(clusters["Ticker"])
    - set(normalised_prices.columns)
)

if missing_price_tickers:
    raise ValueError(
        "These cluster tickers are missing from "
        "the normalised-price data: "
        f"{missing_price_tickers}"
    )

extra_price_tickers = sorted(
    set(normalised_prices.columns)
    - set(clusters["Ticker"])
)

if extra_price_tickers:
    raise ValueError(
        "These companies are present in the price data "
        "but missing from the cluster assignments: "
        f"{extra_price_tickers}"
    )

missing_cluster_names = (
    clusters.loc[
        clusters["Cluster Name"].isna(),
        "Cluster",
    ]
    .unique()
    .tolist()
)

if missing_cluster_names:
    raise ValueError(
        "No cluster name has been defined for: "
        f"{missing_cluster_names}"
    )

missing_sector_tickers = (
    clusters.loc[
        clusters["Sector"].isna(),
        "Ticker",
    ]
    .tolist()
)

if missing_sector_tickers:
    raise ValueError(
        "No sector has been defined for: "
        f"{missing_sector_tickers}"
    )

print("\nTicker and sector validation completed successfully.")


TRADING_DAYS_PER_YEAR = 252

study_years = (
    prices.index[-1] - prices.index[0]
).days / 365.25

print(f"\nApproximate study duration: {study_years:.2f} years")


company_statistics = []

for ticker in clusters["Ticker"]:

    ticker_prices = prices[ticker].dropna()
    ticker_returns = daily_returns[ticker].dropna()

    if ticker_prices.empty:
        raise ValueError(
            f"No valid prices are available for {ticker}."
        )

    initial_value = float(
        ticker_prices.iloc[0]
    )

    final_value = float(
        ticker_prices.iloc[-1]
    )

    total_return = (
        final_value / initial_value
    ) - 1

    if (
        study_years > 0
        and initial_value > 0
        and final_value > 0
    ):
        annualised_return = (
            final_value / initial_value
        ) ** (1 / study_years) - 1
    else:
        annualised_return = np.nan

    annualised_volatility = (
        ticker_returns.std(ddof=1)
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    running_maximum = ticker_prices.cummax()

    drawdown = (
        ticker_prices
        / running_maximum
    ) - 1

    maximum_drawdown = float(
        drawdown.min()
    )

    company_statistics.append(
        {
            "Ticker": ticker,
            "Initial Normalised Value": initial_value,
            "Final Normalised Value": final_value,
            "Total Return": total_return,
            "Annualised Return": annualised_return,
            "Annualised Volatility": annualised_volatility,
            "Maximum Drawdown": maximum_drawdown,
        }
    )

company_statistics = pd.DataFrame(
    company_statistics
)


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
        Minimum_Final_Value=(
            "Final Normalised Value",
            "min",
        ),
        Maximum_Final_Value=(
            "Final Normalised Value",
            "max",
        ),
        Mean_Total_Return=(
            "Total Return",
            "mean",
        ),
        Median_Total_Return=(
            "Total Return",
            "median",
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


cluster_company_counts = (
    company_profiles
    .groupby("Cluster")
    .size()
    .rename("Cluster Size")
    .reset_index()
)

sector_counts = sector_counts.merge(
    cluster_company_counts,
    on="Cluster",
    how="left",
)

sector_counts["Cluster Percentage"] = (
    sector_counts["Company Count"]
    / sector_counts["Cluster Size"]
    * 100
)



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
            "Cluster Percentage": "Dominant Sector Percentage",
        }
    )
)

cluster_statistics = cluster_statistics.merge(
    dominant_sectors[
        [
            "Cluster",
            "Dominant Sector",
            "Dominant Sector Count",
            "Dominant Sector Percentage",
        ]
    ],
    on="Cluster",
    how="left",
)

cluster_statistics["Interpretation"] = (
    cluster_statistics["Cluster"]
    .map(INTERPRETATION_MAP)
)



cluster_summary_table = cluster_statistics[
    [
        "Cluster",
        "Cluster Name",
        "Companies",
        "Mean_Final_Value",
        "Median_Final_Value",
        "Dominant Sector",
        "Dominant Sector Count",
        "Interpretation",
    ]
].copy()

cluster_summary_table = cluster_summary_table.rename(
    columns={
        "Mean_Final_Value":
            "Mean Final Normalised Value",
        "Median_Final_Value":
            "Median Final Normalised Value",
    }
)

cluster_summary_table[
    [
        "Mean Final Normalised Value",
        "Median Final Normalised Value",
    ]
] = (
    cluster_summary_table[
        [
            "Mean Final Normalised Value",
            "Median Final Normalised Value",
        ]
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


financial_characteristics_table = (
    cluster_statistics[
        [
            "Cluster",
            "Cluster Name",
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

sector_breakdowns = []

for cluster_number in sorted(
    company_profiles["Cluster"].unique()
):

    current_cluster_sectors = (
        sector_counts.loc[
            sector_counts["Cluster"]
            == cluster_number
        ]
    )

    breakdown = ", ".join(
        (
            f"{row['Sector']} "
            f"({int(row['Company Count'])}, "
            f"{row['Cluster Percentage']:.1f}%)"
        )
        for _, row
        in current_cluster_sectors.iterrows()
    )

    sector_breakdowns.append(
        {
            "Cluster": cluster_number,
            "Cluster Name": CLUSTER_NAMES[
                cluster_number
            ],
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



company_lists = (
    company_profiles
    .groupby(
        [
            "Cluster",
            "Cluster Name",
        ]
    )["Ticker"]
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


sector_pivot = pd.pivot_table(
    company_profiles,
    index=[
        "Cluster",
        "Cluster Name",
    ],
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


cluster_statistics.to_csv(
    os.path.join(
        RESULTS_DIR,
        "cluster_statistics_full.csv",
    ),
    index=False,
)

sector_counts.to_csv(
    os.path.join(
        RESULTS_DIR,
        "cluster_sector_counts.csv",
    ),
    index=False,
)


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

expected_companies = normalised_prices.shape[1]
profiled_companies = len(company_profiles)

if profiled_companies != expected_companies:
    raise ValueError(
        f"Expected {expected_companies} companies, "
        f"but profiled {profiled_companies}."
    )

print(
    f"\nSuccessfully profiled all "
    f"{profiled_companies} companies."
)

print("\nCluster profiling completed successfully.")

print("\nSaved files:")

saved_files = [
    "company_cluster_profiles.csv",
    "cluster_statistics_full.csv",
    "cluster_sector_counts.csv",
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