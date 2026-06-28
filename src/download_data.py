import yfinance as yf
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

# Example FTSE 100 stocks
tickers = [
    "HSBA.L",   # HSBC
    "BP.L",     # BP
    "SHEL.L",   # Shell
    "AZN.L",    # AstraZeneca
    "ULVR.L"    # Unilever
]

# Download 5 years of daily data
data = yf.download(
    tickers,
    start="2021-01-01",
    end="2026-01-01",
    group_by="ticker",
    auto_adjust=True
)

data.to_csv("data/raw/ftse100_sample.csv")