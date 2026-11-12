
import yfinance as yf
import pandas as pd

# Download historical data
ticker = "RELIANCE.NS"

data = yf.download(
    ticker,
    start="2025-01-01",
    end="2026-10-01",
    interval="1d",
    auto_adjust=True
)

# Handle Yahoo Finance column format
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Keep required columns
data = data[["Open", "High", "Low", "Close", "Volume"]]

data = data.dropna()

# Saving data
data.to_csv("market_data.csv")

print("Market data downloaded successfully!")
print("Shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())

print("\nLast 5 rows:")
print(data.tail())

print("\nMissing values:")
print(data.isnull().sum())
