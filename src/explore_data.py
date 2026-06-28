import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(
    "data/raw/ftse100_sample.csv",
    header=[0, 1],
    index_col=0,
    parse_dates=True
)

print(data.head())

print("\nLast 5 rows:")
print(data.tail())

print("\nDate range:")
print(data.index.min())
print(data.index.max())

print("\nNumber of trading days:")
print(len(data))

print("\nColumn structure:")
print(data.columns)

print("\nMissing values:")
print(data.isnull().sum())

#############################################################################

# Select Shell data
shell = data["SHEL.L"]

print("\nShell Data:")
print(shell.head())

print("\nShell information:")
print(shell.info())

print("\nSummary statistics:")
print(shell.describe())

print("\nData types:")
print(shell.dtypes)

#############################################################################

import matplotlib.dates as mdates

plt.figure(figsize=(12,6))

plt.plot(shell.index, shell["Close"])

plt.title("Shell Closing Price")
plt.xlabel("Year")
plt.ylabel("Price (£)")

# Show one tick per year
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.grid(True)

plt.tight_layout()

plt.savefig("figures/shell_close_price.png")
plt.show()


#############################################################################

plt.figure(figsize=(12, 6))

plt.plot(shell.index, shell["Volume"])

plt.title("Shell Trading Volume")
plt.xlabel("Date")
plt.ylabel("Volume")

plt.grid(True)

plt.savefig("figures/shell_trading_volume.png")
plt.show()

#############################################################################
# find missing values for BP ~ one row only 
bp = data["BP.L"]

print("\nBP Missing Values:")
print(bp[bp.isnull().any(axis=1)])