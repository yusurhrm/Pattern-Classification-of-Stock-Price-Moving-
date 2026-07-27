import yfinance as yf
import pandas as pd
import os

# FTSE 100 companies selected to provide balanced sector representation

tickers = [

    # Banking & Financial Services
    "HSBA.L",   # HSBC
    "BARC.L",   # Barclays
    "LLOY.L",   # Lloyds Banking Group
    "NWG.L",    # NatWest Group
    "STAN.L",   # Standard Chartered

    # Energy & Mining
    "SHEL.L",   # Shell
    "BP.L",     # BP
    "RIO.L",    # Rio Tinto
    "GLEN.L",   # Glencore
    "AAL.L",    # Anglo American

    # Healthcare & Pharmaceuticals
    "AZN.L",    # AstraZeneca
    "GSK.L",    # GSK
    "HIK.L",    # was Haleon changed to Hikma Pharmaceuticals
    "SN.L",     # Smith & Nephew
    "CTEC.L",   # ConvaTec

    # Consumer Goods
    "ULVR.L",   # Unilever
    "DGE.L",    # Diageo
    "RKT.L",    # Reckitt
    "CCH.L",    # Coca-Cola HBC
    "IMB.L",    # Imperial Brands

    # Retail
    "TSCO.L",   # Tesco
    "SBRY.L",   # Sainsbury's
    "MKS.L",    # Marks & Spencer
    "BME.L",    # B&M European Value Retail
    "JD.L",     # JD Sports Fashion

    # Insurance
    "AV.L",     # Aviva
    "LGEN.L",   # Legal & General
    "ADM.L",    # Admiral Group
    "PRU.L",    # Prudential
    "BEZ.L",    # Beazley

    # Utilities
    "NG.L",     # National Grid
    "SSE.L",    # SSE
    "UU.L",     # United Utilities
    "SVT.L",    # Severn Trent
    "CNA.L",    # Centrica

    # Industrials & Technology
    "RR.L",     # Rolls-Royce
    "SMIN.L",   # Smiths Group
    "WEIR.L",   # Weir Group
    "AUTO.L",   # Auto Trader
    "REL.L"     # RELX
]

print(f"Downloading data for {len(tickers)} companies...")

# Download historical data
data = yf.download(
    tickers=tickers,
    start="2021-01-01",
    end="2026-01-01",
    group_by="ticker",
    auto_adjust=False,
    progress=True
)

# Save raw dataset
data.to_csv("data/raw/ftse100_40_companies.csv")

print("\nDownload complete!")
print(f"Dataset shape: {data.shape}")
print("Saved to: data/raw/ftse100_40_companies.csv")