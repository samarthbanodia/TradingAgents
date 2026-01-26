"""Convert parquet to CSV with specified columns."""
import pandas as pd

# Load parquet
df = pd.read_parquet("ohlcv_5min.parquet")

# Ensure columns are in the correct order
columns = ["ticker", "timestamp", "open", "high", "low", "close", "volume"]
df = df[columns]

# Save to CSV
df.to_csv("ohlcv_5min.csv", index=False)

print(f"Saved ohlcv_5min.csv with {len(df)} rows")
print(f"\nColumns: {list(df.columns)}")
print(f"\nSample data:")
print(df.head(10).to_string())
print(f"\n\nBars per ticker:")
print(df.groupby("ticker").size().sort_values())
