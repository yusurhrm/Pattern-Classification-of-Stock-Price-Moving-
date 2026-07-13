import pandas as pd
import matplotlib.pyplot as plt

# Load the downloaded dataset
data = pd.read_csv(
    "data/raw/ftse100_40_companies.csv",
    header=[0, 1],
    index_col=0,
    parse_dates=True
)

# ============================================================
# Dataset overview
# ============================================================

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)

print(f"Rows (trading days): {len(data)}")
print(f"Columns: {data.shape[1]}")
print(f"Date range: {data.index.min()} to {data.index.max()}")

num_companies = len(data.columns.get_level_values(0).unique())
print(f"Number of companies: {num_companies}")

print("\nCompanies:")
print(data.columns.get_level_values(0).unique().tolist())

# ============================================================
# Dataset information
# ============================================================

print("\n" + "=" * 50)
print("DATASET INFO")
print("=" * 50)

data.info()

# ============================================================
# Missing values
# ============================================================

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)

missing_counts = data.isnull().sum()
missing_percentages = data.isnull().mean() * 100

missing_summary = pd.DataFrame({
    "Missing Count": missing_counts,
    "Missing Percentage": missing_percentages
})

print(missing_summary[missing_summary["Missing Count"] > 0])

# ============================================================
# Adjusted closing prices
# ============================================================

close_prices = data.xs("Adj Close", axis=1, level=1)

print("\n" + "=" * 50)
print("ADJUSTED CLOSE SUMMARY STATISTICS")
print("=" * 50)

print(close_prices.describe())

# ============================================================
# Plot raw adjusted closing prices
# ============================================================

plt.figure(figsize=(15, 7))

for company in close_prices.columns:
    plt.plot(
        close_prices.index,
        close_prices[company],
        alpha=0.5,
        label=company
    )

plt.title("Adjusted Closing Prices of 40 FTSE Companies")
plt.xlabel("Date")
plt.ylabel("Adjusted Closing Price (GBp)")
plt.tight_layout()
plt.show()

# ============================================================
# Daily returns
# ============================================================

print("\n" + "=" * 50)
print("DAILY RETURNS")
print("=" * 50)

daily_returns = close_prices.pct_change(fill_method=None)

print(daily_returns.head())

print("\nDaily return summary statistics:")
print(daily_returns.describe())

# ============================================================
# Correlation of daily returns
# ============================================================

print("\n" + "=" * 50)
print("DAILY RETURN CORRELATION")
print("=" * 50)

correlation = daily_returns.corr()

print(correlation)

plt.figure(figsize=(12, 10))
plt.imshow(
    correlation,
    vmin=-1,
    vmax=1,
    aspect="auto"
)

plt.colorbar(label="Correlation")
plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90,
    fontsize=7
)
plt.yticks(
    range(len(correlation.index)),
    correlation.index,
    fontsize=7
)

plt.title("Correlation Matrix of Daily Stock Returns")
plt.tight_layout()
plt.show()

# ============================================================
# Return distribution
# ============================================================

all_returns = daily_returns.stack().dropna()

plt.figure(figsize=(10, 6))
plt.hist(all_returns, bins=100)

plt.title("Distribution of Daily Stock Returns")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ============================================================
# Skewness and kurtosis
# ============================================================

print("\n" + "=" * 50)
print("SKEWNESS")
print("=" * 50)

print(daily_returns.skew().sort_values())

print("\n" + "=" * 50)
print("KURTOSIS")
print("=" * 50)

print(daily_returns.kurtosis().sort_values())

# ============================================================
# Volatility
# ============================================================

print("\n" + "=" * 50)
print("DAILY VOLATILITY")
print("=" * 50)

daily_volatility = daily_returns.std().sort_values()
print(daily_volatility)

print("\n" + "=" * 50)
print("ANNUALISED VOLATILITY")
print("=" * 50)

annualised_volatility = daily_returns.std() * (252 ** 0.5)
print(annualised_volatility.sort_values())