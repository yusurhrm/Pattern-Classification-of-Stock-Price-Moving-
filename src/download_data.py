from pathlib import Path

import pandas as pd
import yfinance as yf


# --------------------------------------------------
# File paths
# --------------------------------------------------

# Project root:
# Pattern-Classification-of-Stock-Price-Moving-/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MAPPING_FILE = PROJECT_ROOT / "data" / "ftse100_ticker_mapping.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "ftse100_100_companies.csv"


# --------------------------------------------------
# Download settings
# --------------------------------------------------

START_DATE = "2021-01-01"

# yfinance treats the end date as exclusive.
# This therefore downloads through 31 December 2025.
END_DATE = "2026-01-01"


def load_tickers() -> tuple[pd.DataFrame, list[str]]:
    """Load and validate company names and Yahoo Finance tickers."""

    mapping = pd.read_excel(MAPPING_FILE)

    required_columns = {"Company", "Ticker"}
    missing_columns = required_columns - set(mapping.columns)

    if missing_columns:
        raise ValueError(
            f"Mapping file is missing these columns: {sorted(missing_columns)}"
        )

    # Remove accidental spaces from names and tickers.
    mapping["Company"] = mapping["Company"].astype("string").str.strip()
    mapping["Ticker"] = mapping["Ticker"].astype("string").str.strip().str.upper()

    # Remove completely empty rows.
    mapping = mapping.dropna(subset=["Company"], how="all")

    # Check for missing tickers.
    missing_tickers = mapping[
        mapping["Ticker"].isna() | mapping["Ticker"].eq("")
    ]

    if not missing_tickers.empty:
        companies = missing_tickers["Company"].tolist()
        raise ValueError(
            "The following companies have no ticker:\n"
            + "\n".join(f"- {company}" for company in companies)
        )

    # Check for duplicate tickers.
    duplicate_tickers = mapping[
        mapping["Ticker"].duplicated(keep=False)
    ].sort_values("Ticker")

    if not duplicate_tickers.empty:
        raise ValueError(
            "Duplicate tickers found:\n"
            + duplicate_tickers[["Company", "Ticker"]].to_string(index=False)
        )

    tickers = mapping["Ticker"].tolist()

    print(f"Loaded {len(mapping)} companies.")
    print(f"Loaded {len(tickers)} unique tickers.")

    if len(tickers) != 100:
        print(
            f"Warning: expected 100 tickers, but the file contains {len(tickers)}."
        )

    return mapping, tickers


def download_prices(tickers: list[str]) -> pd.DataFrame:
    """Download daily historical prices for all tickers."""

    print(f"\nDownloading data from {START_DATE} to {END_DATE}...")

    data = yf.download(
        tickers=tickers,
        start=START_DATE,
        end=END_DATE,
        group_by="ticker",
        auto_adjust=False,
        actions=False,
        progress=True,
        threads=True,
    )

    if data.empty:
        raise RuntimeError("Yahoo Finance returned no data.")

    return data


def identify_failed_tickers(
    data: pd.DataFrame,
    tickers: list[str],
) -> list[str]:
    """Identify tickers with no downloaded closing-price data."""

    failed = []

    for ticker in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                ticker_data = data[ticker]

                price_column = (
                    "Adj Close"
                    if "Adj Close" in ticker_data.columns
                    else "Close"
                )

                if ticker_data[price_column].dropna().empty:
                    failed.append(ticker)

            else:
                # This normally only applies when one ticker is downloaded.
                price_column = (
                    "Adj Close"
                    if "Adj Close" in data.columns
                    else "Close"
                )

                if data[price_column].dropna().empty:
                    failed.append(ticker)

        except (KeyError, TypeError):
            failed.append(ticker)

    return failed


def main() -> None:
    """Run the complete download process."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mapping, tickers = load_tickers()
    data = download_prices(tickers)

    failed_tickers = identify_failed_tickers(data, tickers)

    if failed_tickers:
        print("\nThese tickers returned no usable data:")
        for ticker in failed_tickers:
            company = mapping.loc[
                mapping["Ticker"].eq(ticker), "Company"
            ].iloc[0]
            print(f"- {company}: {ticker}")
    else:
        print("\nAll tickers returned data.")

    data.to_csv(OUTPUT_FILE)

    print(f"\nDataset shape: {data.shape}")
    print(f"Data saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()